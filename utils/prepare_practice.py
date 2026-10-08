"""Prepare four independently sourced, visually checked study regions. Never overwrite user work."""
import argparse
import tempfile
import zipfile
from pathlib import Path

import cv2
import torch

from utils.annotation_workflow import create_crop, filter_proposals, PROFILES
from utils.classifier import write_json
from utils.sam_segment import build_sam, SamAutomaticMaskGenerator, segment_image, visualize
from utils.config import DATA_DIR

REGIONS = [('P0018',[399,108,614,614]), ('P0021',[190,87,671,671]),
           ('P0111',[644,1168,512,512]), ('P0142',[16,60,512,512])]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',default=str(DATA_DIR/'images/part1.zip'))
    ap.add_argument('--images',default='dataset/images');ap.add_argument('--results',default='results')
    args=ap.parse_args()
    settings=PROFILES['focused']; generator=None
    with zipfile.ZipFile(args.archive) as archive, tempfile.TemporaryDirectory(prefix='sam-practice-') as temp:
        for source,box in REGIONS:
            name='practice_'+source
            if (Path(args.images)/(name+'.png')).exists() or (Path(args.results)/(name+'.json')).exists():
                print('保留已有练习图与标注：',name);continue
            original=Path(temp)/(source+'.png');original.write_bytes(archive.read('images/'+source+'.png'))
            ann=create_crop(original,box,args.images,args.results,name=name,recommended=True)
            image=cv2.imread(str(Path(args.images)/ann['image']))
            if generator is None:
                generator=SamAutomaticMaskGenerator(build_sam('cuda' if torch.cuda.is_available() else 'cpu'),
                    points_per_side=settings['points_per_side'],pred_iou_thresh=settings['pred_iou_thresh'],
                    stability_score_thresh=settings['stability_score_thresh'],min_mask_region_area=settings['min_area'])
            ann['instances']=filter_proposals(segment_image(image,generator,min_area=settings['min_area']),image.shape[:2],'focused')
            for i,inst in enumerate(ann['instances']):inst['id']=i
            ann['segmentation_profile']='focused'
            write_json(Path(args.results)/(name+'.json'),ann)
            out=Path(args.results)/'local_run'/'practice';out.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(out/(name+'.jpg')),visualize(image,ann['instances']))
            print(name,'候选实例',len(ann['instances']),flush=True)


if __name__=='__main__':main()
