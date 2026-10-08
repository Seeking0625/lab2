"""Manage independent small study regions without changing existing annotations."""
import copy
import hashlib
import math
from pathlib import Path

import cv2

from utils.classifier import write_json

PROFILES = {
    'standard': dict(points_per_side=16, pred_iou_thresh=.86, stability_score_thresh=.9,
                     min_area=100, max_fraction=1.),
    'focused': dict(points_per_side=24, pred_iou_thresh=.90, stability_score_thresh=.95,
                    min_area=500, max_fraction=.65),
}


def create_crop(image_path, box, image_dir, result_dir, parent_annotation=None, name=None, recommended=False):
    image=cv2.imread(str(image_path))
    if image is None: raise ValueError('无法读取原图。')
    if len(box)!=4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError('裁图框必须是四个有限数值。')
    x,y,w,h=map(lambda v:round(float(v)),box)
    if w<=0 or h<=0: raise ValueError('裁图框宽高必须为正。')
    H,W=image.shape[:2]
    x0,y0=max(0,x),max(0,y);x1,y1=min(W,x+w),min(H,y+h)
    if x1-x0<64 or y1-y0<64: raise ValueError('请选择宽高至少64像素的局部区域。')
    stem=name or f'crop_{Path(image_path).stem}_{x0}_{y0}_{x1-x0}_{y1-y0}'
    image_out=Path(image_dir)/(stem+'.png');annotation_out=Path(result_dir)/(stem+'.json')
    if image_out.exists() or annotation_out.exists():
        raise ValueError('这个局部图已经存在，请从图像列表打开或选择不同区域。')
    image_out.parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(image_out),image[y0:y1,x0:x1]): raise ValueError('保存裁图失败。')
    parent=parent_annotation or {}
    digest=parent.get('source_image_hash') or hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    parent_box=parent.get('source_box', [0,0,W,H])
    ann=dict(image=image_out.name,width=x1-x0,height=y1-y0,instances=[],
             source_image=parent.get('source_image',Path(image_path).name),source_image_hash=digest,
             source_box=[x0+parent_box[0],y0+parent_box[1],x1-x0,y1-y0],recommended=recommended)
    for key in ('source_video','frame_index'):
        if key in parent: ann[key]=parent[key]
    for inst in parent.get('instances',[]):
        bx,by,bw,bh=inst['bbox']
        # Only carry whole objects; clipped boundaries must be segmented afresh.
        if bx<x0 or by<y0 or bx+bw>x1 or by+bh>y1: continue
        item=copy.deepcopy(inst)
        item['bbox']=[bx-x0,by-y0,bw,bh]
        for key in ('polygons','holes'):
            item[key]=[[[px-x0,py-y0] for px,py in ring] for ring in item.get(key,[])]
        if item.get('polygon') and not item['polygons']:
            item['polygons']=[[[px-x0,py-y0] for px,py in item.pop('polygon')]]
        item['id']=len(ann['instances']);ann['instances'].append(item)
    write_json(annotation_out,ann)
    return ann


def filter_proposals(instances, shape, profile):
    settings=PROFILES[profile]
    total=shape[0]*shape[1]
    return [i for i in instances if i['area']>=settings['min_area']
            and i['score']>=settings['pred_iou_thresh'] and i['area']<=total*settings['max_fraction']]
