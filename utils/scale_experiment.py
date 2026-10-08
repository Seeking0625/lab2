"""1/4/9-area scale experiment: split into 1x1, 2x2, 3x3 crops, resize, segment, classify, count."""
import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from utils.classifier import classify_annotation, load_model, write_json, metrics
from utils.geometry import mask_for
from utils.sam_segment import build_sam, SamAutomaticMaskGenerator, visualize
from utils.report import generate_report


def transform_instance(inst, sx, sy, ox=0, oy=0):
    item=copy.deepcopy(inst)
    if not item.get('polygons') and item.get('polygon'):
        item['polygons'] = [item.pop('polygon')]
    for key in ('polygons','holes'):
        item[key]=[[[float(x*sx+ox),float(y*sy+oy)] for x,y in ring] for ring in item.get(key,[])]
    x,y,w,h=item['bbox']
    item['bbox']=[round(x*sx+ox),round(y*sy+oy),max(1,round(w*sx)),max(1,round(h*sy))]
    item['area']=max(1,round(item['area']*sx*sy))
    return item


def match_instances(reference, predictions, width, height, threshold=.5):
    # IoU rasterized at <=512 px long side to bound comparison memory.
    ratio=min(1.,512/max(width,height))
    shape=(max(1,round(height*ratio)),max(1,round(width*ratio)))
    def masks(items):
        return [mask_for(transform_instance(i,ratio,ratio),shape).astype(bool) for i in items]
    left,right=masks(reference),masks(predictions)
    edges=[]
    for i,a in enumerate(left):
        for j,b in enumerate(right):
            inter=np.count_nonzero(a&b)
            if not inter: continue
            iou=inter/max(np.count_nonzero(a|b),1)
            if iou>=threshold: edges.append((float(iou),i,j))
    used_a,used_b=set(),set(); pairs=[]
    for iou,i,j in sorted(edges,reverse=True):
        if i in used_a or j in used_b: continue
        used_a.add(i); used_b.add(j); pairs.append((i,j,iou))
    return pairs


def run_scale_experiment(image_path, out_dir, model_paths=None, annotation_path=None,
                         reference_side=1024, points_per_side=16, generator=None):
    image=cv2.imread(str(image_path))
    if image is None: raise ValueError('无法读取图片。')
    if reference_side<64 or points_per_side<1: raise ValueError('分辨率至少64，采样点数至少1。')
    H,W=image.shape[:2]
    if min(H,W)<3: raise ValueError('图片宽高至少为3。')
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    models={Path(p).stem:load_model(p) for p in model_paths or []}
    truth=[]
    group = None
    if annotation_path:
        ann=json.loads(Path(annotation_path).read_text(encoding='utf-8'))
        group = ann.get('source_video') or ann.get('source_image_hash')
        truth=[i for i in ann['instances'] if i.get('class','unlabeled')!='unlabeled'
               and i.get('label_source','manual')=='manual']
    digest=hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    if generator is None:
        import torch
        generator=SamAutomaticMaskGenerator(build_sam('cuda' if torch.cuda.is_available() else 'cpu'),
                  points_per_side=points_per_side,pred_iou_thresh=.86,stability_score_thresh=.9,min_mask_region_area=100)
    from utils.sam_segment import segment_image
    # Same target size for each grid cell: n times linear resolution relative to baseline.
    target=(max(3,round(W/max(W,H)*reference_side)),max(3,round(H/max(W,H)*reference_side)))
    rows=[]; baseline={}
    for n in (1,2,3):
        started=time.time(); outputs={name:[] for name in models or {'unclassified':None}}
        for row in range(n):
            y0,y1=round(row*H/n),round((row+1)*H/n)
            for col in range(n):
                x0,x1=round(col*W/n),round((col+1)*W/n)
                tile=cv2.resize(image[y0:y1,x0:x1],target,interpolation=cv2.INTER_LINEAR)
                instances=segment_image(tile,generator,min_area=100)
                ann={'image':Path(image_path).name,'width':target[0],'height':target[1],'instances':instances}
                for name in outputs:
                    classified=classify_annotation(tile,ann,models[name],False) if name in models else ann
                    for inst in classified['instances']:
                        outputs[name].append(transform_instance(inst,(x1-x0)/target[0],(y1-y0)/target[1],x0,y0))
        for name,instances in outputs.items():
            for i,inst in enumerate(instances): inst['id']=i
            ann={'image':Path(image_path).name,'width':W,'height':H,'instances':instances}
            stem=f'{name}_area{n*n}'
            write_json(out/f'{stem}.json',ann)
            cv2.imwrite(str(out/f'{stem}.jpg'),visualize(image,instances))
            (out/f'{stem}.md').write_text(generate_report(ann)['markdown'],encoding='utf-8')
            counts=dict(Counter(i['class'] for i in instances))
            item={'model':name,'area_multiplier':n*n,'grid':n,'instances':len(instances),
                  'counts':counts,'seconds':round(time.time()-started,2)}
            if n==1: baseline[name]=instances
            pairs=match_instances(baseline[name],instances,W,H)
            item['baseline_match_fraction']=len(pairs)/max(len(baseline[name]),1)
            item['count_change']=len(instances)-len(baseline[name])
            item['class_agreement_matched']=sum(baseline[name][i]['class']==instances[j]['class'] for i,j,_ in pairs)/len(pairs) if pairs and name in models else None
            if truth:
                pairs=match_instances(truth,instances,W,H)
                item['ground_truth']={'matched':len(pairs),'annotated':len(truth),
                      'recall_at_iou_05':len(pairs)/len(truth), 'unmatched_predictions':len(instances)-len(pairs),
                      'note':'未标注区域的预测不一定是假阳性；IoU 使用最长边不超过512像素的栅格近似。'}
                if name in models and pairs:
                    item['ground_truth']['classification_on_matches']=metrics(
                        [truth[i]['class'] for i,_,_ in pairs], [instances[j]['class'] for _,j,_ in pairs],
                        sorted({i['class'] for i in truth}|set(models[name]['labels'])))
                    item['ground_truth']['held_out_image']=(group or digest) not in models[name]['train_groups']
            rows.append(item)
            print(f"[scale] {name} area={n*n} instances={len(instances)}",flush=True)
    result={'image':str(image_path),'reference_side':reference_side,'target_crop_size':target,
            'method':'各网格裁块缩放到相同参考尺寸；结果映射回原图。网格边界切断的目标不做启发式合并，计数变化包含边界效应。',
            'has_classifier':bool(models),'has_manual_truth':bool(truth),'rows':rows}
    write_json(out/'summary.json',result)
    lines=['# 尺度鲁棒性实验','',result['method'],
           f'参考最长边：{reference_side}px；线性尺度1/2/3对应面积1/4/9倍。',
           '分类模型：'+(', '.join(models) or '尚未训练，当前仅为分割尺度实验。'),
           '有人工标注时提供匹配召回与匹配实例分类指标；训练图上的结果仅为探索性分析。','',
           '| 模型 | 面积倍数 | 实例数 | 相对原尺度计数变化 | 原尺度实例匹配率 |','|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['area_multiplier']} | {r['instances']} | {r['count_change']:+d} | {r['baseline_match_fraction']:.3f} |")
    lines+=['','## 分析','实例数、类别计数或匹配率随尺度变化，说明流水线对尺度/裁块边界敏感；一致性不等于准确率。']
    for name in outputs:
        subset=[r for r in rows if r['model']==name and r['area_multiplier']>1]
        for r in subset:
            lines.append(f"- {name} 面积{r['area_multiplier']}倍：计数变化{r['count_change']:+d}，匹配率{r['baseline_match_fraction']:.1%}；类别计数{r['counts']}。")
    if len(models)>1:
        lines.append('增强是否有效需对照同尺度的人工真值指标及 classifier/report.md 的固定测试集结果；不能仅用实例数量判断。')
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--image',required=True); ap.add_argument('--out',default='results/scales')
    ap.add_argument('--models',nargs='*',default=[]); ap.add_argument('--annotation')
    ap.add_argument('--reference-side',type=int,default=1024); ap.add_argument('--points-per-side',type=int,default=16)
    args=ap.parse_args()
    run_scale_experiment(args.image,args.out,args.models,args.annotation,args.reference_side,args.points_per_side)


if __name__=='__main__': main()
