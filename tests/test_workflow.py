import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from utils.geometry import mask_for, split_instance, merge_instances, resegment_candidates
from utils.annotation_workflow import create_crop, filter_proposals
from utils.classifier import (train_classifier, load_samples, load_model, classify_annotation,
                              write_json, metrics)
from utils.scale_experiment import run_scale_experiment, transform_instance
from utils.video import extract_frames, analyze_video


def rect(x,y,w,h,cls='unlabeled'):
    return {'id':0,'class':cls,'bbox':[x,y,w,h],'area':w*h,'score':.99,
            'polygons':[[[x,y],[x+w-1,y],[x+w-1,y+h-1],[x,y+h-1]]]}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def data(self):
        images=self.root/'images'; anns=self.root/'annotations'
        images.mkdir(); anns.mkdir()
        for j in range(6):
            image=np.full((64,96,3),j,np.uint8)
            image[5:25,5:25]=[10,10,220+j]
            image[30:55,40:70]=[10,220+j,10]
            cv2.imwrite(str(images/f'{j}.png'),image)
            write_json(anns/f'{j}.json',{'image':f'{j}.png','width':96,'height':64,
                'instances':[rect(5,5,20,20,'red'),rect(40,30,30,25,'green')]})
        return images,anns

    def test_training_split_augmentation_and_metrics(self):
        images,anns=self.data()
        result=train_classifier(anns,images,self.root/'models',test_size=.34)
        manifest=json.loads((self.root/'models/split.json').read_text())
        self.assertFalse({s['group'] for s in manifest['train']} & {s['group'] for s in manifest['test']})
        self.assertEqual(result['train_instances']+result['test_instances'],12)
        for name in ['baseline','augmented']:
            for scale in ['1','2','3']:
                self.assertEqual(result['models'][name][scale]['n'],result['test_instances'])
                self.assertGreaterEqual(result['models'][name][scale]['oa'],.5)
        self.assertEqual(metrics(['a','b'],['a','b'],['a','b'])['kappa'],1)
        bundle=load_model(self.root/'models/augmented.joblib')
        ann=json.loads((anns/'0.json').read_text()); ann['instances'][0]['class']='unlabeled'
        prediction=classify_annotation(cv2.imread(str(images/'0.png')),ann,bundle)
        self.assertEqual(prediction['instances'][0]['class'],'red')
        self.assertEqual(prediction['instances'][0]['label_source'],'model')
        self.assertEqual(prediction['instances'][1]['class'],'green')
        write_json(anns/'0.json',prediction)
        self.assertEqual(len(load_samples(anns,images)),11)  # never train on model labels

    def test_missing_labels_rejected(self):
        images=self.root/'images'; images.mkdir()
        anns=self.root/'anns'; anns.mkdir()
        with self.assertRaisesRegex(ValueError,'两个类别'):
            train_classifier(anns,images,self.root/'models')

    def test_split_preserves_remainder_and_holes(self):
        original=rect(2,2,25,25,'red'); candidate=rect(8,8,8,8)
        parts=split_instance(original,candidate,(32,32))
        self.assertEqual(len(parts),2)
        self.assertTrue(any(p['holes'] for p in parts))
        masks=[mask_for(p,(32,32)) for p in parts]
        self.assertEqual(int((masks[0]&masks[1]).sum()),0)
        np.testing.assert_array_equal(masks[0]|masks[1],mask_for(original,(32,32)))
        self.assertEqual(sum(p['area'] for p in parts),625)
        with self.assertRaises(ValueError): split_instance(original,original,(32,32))

    def test_resegment_preview_and_thin_remainder(self):
        original=rect(2,2,10,10,'red');original['label_source']='manual'
        # A nine-pixel-wide candidate leaves a one-pixel-wide strip.
        candidate=rect(2,2,9,10)
        parts=split_instance(original,candidate,(32,32))
        self.assertEqual(len(parts),2)
        rebuilt=np.zeros((32,32),np.uint8)
        for p in parts:rebuilt |= mask_for(p,(32,32))
        np.testing.assert_array_equal(rebuilt,mask_for(original,(32,32)))
        self.assertTrue(all(p['label_source']=='manual' for p in parts))
        # Legacy polygons can extend beyond their recorded bbox.
        legacy=rect(2,2,11,11);legacy["bbox"]=[2,2,10,10]
        rebuilt=np.zeros((32,32),np.uint8)
        for part in split_instance(legacy,candidate,(32,32)):rebuilt |= mask_for(part,(32,32))
        np.testing.assert_array_equal(rebuilt,mask_for(legacy,(32,32)))
        expanded=rect(1,1,12,12)
        previews=resegment_candidates(original,[original,rect(20,20,3,3),candidate,expanded],(32,32))
        self.assertEqual([p['edit_action'] for p in previews],['split','replace'])
        self.assertEqual(len(previews[0]['split_parts']),2)

    def test_merge_uses_union_area(self):
        a,b=rect(1,1,10,10,'red'),rect(6,1,10,10,'red')
        merged=merge_instances([a,b],(20,20))
        self.assertEqual(merged['area'],150)
        self.assertEqual(mask_for(merged,(20,20)).sum(),150)
        b['class']='green'
        with self.assertRaises(ValueError): merge_instances([a,b],(20,20))

    def test_crops_preserve_labels_and_source_groups(self):
        images,anns=self.data()
        original=(anns/'0.json').read_bytes()
        parent=json.loads(original)
        crop=create_crop(images/'0.png',[0,0,80,64],images,anns,parent,name='crop_test')
        self.assertEqual(len(crop['instances']),2)
        self.assertEqual(crop['instances'][0]['class'],'red')
        self.assertEqual(crop['source_image'],'0.png')
        self.assertEqual((anns/'0.json').read_bytes(),original)
        samples=load_samples(anns,images)
        groups={s['group'] for s in samples if s['image'] in ('0.png','crop_test.png')}
        self.assertEqual(len(groups),1)
        self.assertEqual(len(load_samples(anns,images,['crop_test.png'])),2)
        with self.assertRaises(ValueError): create_crop(images/'0.png',[0,0,80,64],images,anns,parent,name='crop_test')
        with self.assertRaises(ValueError): create_crop(images/'0.png',[0,0,10,10],images,anns,parent)

    def test_focus_profile_filters_fresh_proposals(self):
        small=rect(0,0,10,10); large=rect(0,0,90,90); clear=rect(10,10,25,25)
        items=filter_proposals([small,large,clear],(100,100),'focused')
        self.assertEqual(items,[clear])
        self.assertEqual(len(filter_proposals([small,large,clear],(100,100),'standard')),3)

    def test_scale_coordinates(self):
        inst=transform_instance(rect(10,20,30,40),.5,.25,100,200)
        self.assertEqual(inst['bbox'],[105,205,15,10])
        self.assertEqual(inst['polygons'][0][0],[105.,205.])

    def test_scale_pipeline(self):
        images,anns=self.data()
        train_classifier(anns,images,self.root/'models',test_size=.34)
        class Generator:
            def generate(self,image):
                h,w=image.shape[:2]; mask=np.zeros((h,w),bool); mask[1:-1,1:-1]=True
                return [dict(predicted_iou=.99,area=int(mask.sum()),segmentation=mask,bbox=[1,1,w-2,h-2])]
        result=run_scale_experiment(images/'0.png',self.root/'scales',
            [self.root/'models/baseline.joblib',self.root/'models/augmented.joblib'],
            anns/'0.json',reference_side=96,generator=Generator())
        self.assertEqual([r['instances'] for r in result['rows']],[1,1,4,4,9,9])
        self.assertTrue((self.root/'scales/report.md').exists())

    def test_video_reuses_edits_and_correct_metadata(self):
        training_images, training_annotations = self.data()
        train_classifier(training_annotations, training_images, self.root/'models')
        path=self.root/'video.mp4'
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5,(64,64))
        self.assertTrue(writer.isOpened())
        for i in range(10): writer.write(np.full((64,64,3),i*10,np.uint8))
        writer.release()
        frames,meta=extract_frames(str(path),2,1)
        self.assertEqual(meta['total_frames'],10)
        self.assertEqual(meta['decoded_frames'],1)
        with self.assertRaises(ValueError): extract_frames(str(path),0)
        images=self.root/'frames'; annotations=self.root/'annotations'
        with patch('utils.video.build_sam'),patch('utils.video.SamAutomaticMaskGenerator'),patch('utils.video.segment_image',return_value=[rect(2,2,20,20)]):
            result=analyze_video(str(path),str(self.root/'out'),stride=5,max_frames=1,
                                 image_dir=str(images),annotation_dir=str(annotations),
                                 model_path=self.root/'models/augmented.joblib')
        self.assertIn('无法判断',result['trend'])
        ann_path=annotations/'video_f00000.json'
        ann=json.loads(ann_path.read_text())
        self.assertEqual(ann['instances'][0]['label_source'], 'model')
        self.assertTrue(result['cls_total'])
        ann['instances'][0]['class']='building 建筑'
        ann['instances'][0]['label_source']='manual'; write_json(ann_path,ann)
        with patch('utils.video.segment_image',side_effect=AssertionError('must reuse')):
            result=analyze_video(str(path),str(self.root/'out'),stride=5,max_frames=1,
                                 image_dir=str(images),annotation_dir=str(annotations),
                                 model_path=self.root/'models/augmented.joblib')
        self.assertEqual(result['cls_total'],{'building 建筑':1})
        self.assertTrue(result['region_ranking'])
        self.assertTrue((images/'video_f00000.png').exists())


if __name__=='__main__': unittest.main()
