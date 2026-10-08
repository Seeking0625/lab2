"""Train from local DOTA v1.0 labels without user annotation; image-disjoint validation."""
import argparse
import hashlib
import json
import time
import zipfile
from collections import Counter,defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from utils.classifier import VERSION,features,metrics,region_patch,write_json
from utils.appearance import appearance_features
from utils.config import DATA_DIR, ROOT

LABELS={
 'plane':'airplane 飞机','ship':'ship 船舶','storage-tank':'storage-tank 储油罐',
 'baseball-diamond':'baseball-diamond 棒球场','tennis-court':'tennis-court 网球场',
 'basketball-court':'basketball-court 篮球场','ground-track-field':'ground-track-field 田径场',
 'harbor':'harbor 港口','bridge':'bridge 桥梁','large-vehicle':'large-vehicle 大型车辆',
 'small-vehicle':'small-vehicle 小型车辆','helicopter':'helicopter 直升机','roundabout':'roundabout 环岛',
 'soccer-ball-field':'soccer-ball-field 足球场','swimming-pool':'swimming-pool 泳池'}
OTHER='other 背景/其他'
RESERVED={'P0000','P0002','P1910','P0018','P0021','P0111','P0142'}


def parse_labels(text):
    objects=[]
    for lineno,line in enumerate(text.splitlines()):
        parts=line.split()
        if len(parts)!=10 or parts[8] not in LABELS:continue
        pts=np.array(list(map(float,parts[:8]))).reshape(4,2)
        x,y,w,h=cv2.boundingRect(np.rint(pts).astype(np.int32))
        if min(w,h)<4 or cv2.contourArea(pts.astype(np.float32))<12:continue
        objects.append({'line':lineno,'class':LABELS[parts[8]],'difficult':int(parts[9]),
                        'bbox':[x,y,w,h],'area':max(1,int(cv2.contourArea(pts.astype(np.float32)))),
                        'polygons':[pts.tolist()]})
    return objects


def catalogue(image_archives,labels_archive):
    images={};handles=[]
    for path in image_archives:
        z=zipfile.ZipFile(path);handles.append(z)
        for name in z.namelist():
            if name.lower().endswith('.png'):images[Path(name).stem]=(z,name,str(path))
    objects={}
    with zipfile.ZipFile(labels_archive) as z:
        for name in sorted(z.namelist()):
            stem=Path(name).stem
            if stem in images:objects[stem]=parse_labels(z.read(name).decode())
    return images,objects,handles


def split_images(objects,seed=42):
    names=sorted(objects);target=set(LABELS.values())
    y=[{o['class'] for o in objects[n] if not o['difficult']} for n in names]
    for a,b in GroupShuffleSplit(n_splits=1000,test_size=.2,random_state=seed).split(names,groups=names):
        train={names[i] for i in a}-RESERVED;val={names[i] for i in b}|(set(names)&RESERVED)
        if set().union(*(y[names.index(n)] for n in train))==target and set().union(*(y[names.index(n)] for n in val))==target:
            return train,val
    raise ValueError('无法让训练/验证都包含15类，请提供更多有标签的图像。')


def choose_objects(objects,groups,cap,rng):
    by_class=defaultdict(list)
    for name in sorted(groups):
        for obj in objects[name]:
            if not obj['difficult']:by_class[obj['class']].append((name,obj))
    selected=defaultdict(list)
    for label,rows in sorted(by_class.items()):
        for i in rng.permutation(len(rows))[:cap]:
            name,obj=rows[i];selected[name].append(obj)
    return selected


def negative_regions(image,objects,rng,n=3):
    H,W=image.shape[:2];result=[]
    for attempt in range(100):
        side=int(rng.integers(64,min(224,H,W)+1)) if min(H,W)>=64 else 0
        if not side:break
        x=int(rng.integers(0,W-side+1));y=int(rng.integers(0,H-side+1))
        if any(max(0,min(x+side,bx+bw)-max(x,bx))*max(0,min(y+side,by+bh)-max(y,by))>0
               for bx,by,bw,bh in (o['bbox'] for o in objects)):continue
        result.append({'class':OTHER,'line':-1,'bbox':[x,y,side,side],'area':side*side,
                       'polygons':[[[x,y],[x+side-1,y],[x+side-1,y+side-1],[x,y+side-1]]]})
        if len(result)>=n:break
    return result


def extract(images,objects,selected,split,rng,augment):
    X=[];y=[];manifest=[];hard=[]
    for k,name in enumerate(sorted(selected)):
        handle,entry,archive=images[name]
        raw=handle.read(entry);image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
        if image is None:raise ValueError(f'无法读取{name}')
        rows=selected[name]+negative_regions(image,objects[name],rng)
        regions=[];kept=[]
        for obj in rows:
            try:regions.append(region_patch(image,obj));kept.append(obj)
            except ValueError:continue
        if regions:
            X.append(appearance_features(regions));y.extend(o['class'] for o in kept)
            hard.extend(features(*r) for r in regions)
            for obj in kept:
                manifest.append(dict(image=name,sha256=hashlib.sha256(raw).hexdigest(),
                    annotation_line=obj['line'],label=obj['class'],bbox=obj['bbox'],
                    label_source='reference' if obj['class']!=OTHER else 'weak_background',split=split))
            if augment:
                X.append(appearance_features(regions,rotate=True));y.extend(o['class'] for o in kept)
        if (k+1)%20==0 or k+1==len(selected):print(f'[{split}] images {k+1}/{len(selected)} objects {len(manifest)}',flush=True)
    return np.concatenate(X),np.array(y),np.array(hard),manifest


def train(image_archives,labels_archive,out_dir,cap=1000,val_cap=300,seed=42):
    start=time.time();out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    images,objects,handles=catalogue(image_archives,labels_archive)
    try:
        train_groups,val_groups=split_images(objects,seed)
        rng=np.random.default_rng(seed)
        selected_train=choose_objects(objects,train_groups,cap,rng)
        selected_val=choose_objects(objects,val_groups,val_cap,rng)
        # Reserve all demo sources in validation, regardless of randomly capped object sampling.
        for name in RESERVED&val_groups:
            selected_val[name]=[o for o in objects[name] if not o['difficult']]
        print('Sources:',len(train_groups),'train,',len(val_groups),'validation; no shared images',flush=True)
        Xt,yt,Ht,mt=extract(images,objects,selected_train,'train',rng,True)
        Xv,yv,Hv,mv=extract(images,objects,selected_val,'validation',rng,False)
        np.savez_compressed(out/'features.npz',X_train=Xt,y_train=yt,X_val=Xv,y_val=yv,H_train=Ht,H_val=Hv)
        write_json(out/'split.json',dict(seed=seed,train_images=sorted(train_groups),validation_images=sorted(val_groups),
                    reserved_demo_sources=sorted(RESERVED),train=mt,validation=mv))
        labels=sorted(set(yt));print('Fitting classifier:',Xt.shape,'validation:',Xv.shape,flush=True)
        model=RandomForestClassifier(n_estimators=400,max_depth=28,min_samples_leaf=2,
                max_features=.25,class_weight='balanced_subsample',random_state=seed,n_jobs=4)
        model.fit(Xt,yt)
        prediction=model.predict(Xv);evaluation=metrics(yv,prediction,labels)
        baseline=RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=seed,n_jobs=4)
        baseline.fit(Ht,np.array([m['label'] for m in mt]))
        # Extraction stores base and rotation per image; Ht contains only base examples.
        base_metrics=metrics(yv,baseline.predict(Hv),labels)
        bundle=dict(version=VERSION,model=model,labels=labels,feature_backend='resnet18+handcrafted',
                    variant='dota-reference',train_groups=sorted({m['sha256'] for m in mt}),
                    seed=seed,reject_threshold=.35,min_margin=.04,source='DOTA v1.0 rotated boxes',
                    supported_categories=list(LABELS.values()),background_label=OTHER)
        joblib.dump(bundle,out/'model.joblib')
        result=dict(model='Frozen ImageNet ResNet18 features + supervised RandomForest',
            source='DOTA v1.0 labelTxt.zip; difficult=0; rotated rectangles are not iSAID pixel masks',
            seed=seed,train_images_used=len(selected_train),validation_images_used=len(selected_val),
            train_objects=len(mt),train_augmented_rows=len(yt),validation_objects=len(mv),
            train_counts=dict(Counter(m['label'] for m in mt)),validation_counts=dict(Counter(yv)),
            validation=evaluation,handcrafted_baseline=base_metrics,
            background_note='Negative crops avoid all provided target boxes. DOTA can omit objects; background labels are weak supervision.',
            inference_note='Scores are uncalibrated class votes; score<.35 or top-two gap<.04 leaves unlabeled and offers suggestions.',
            seconds=round(time.time()-start,1))
        write_json(out/'evaluation.json',result)
        write_json(out/'validation_predictions.json',[dict(m,prediction=str(p)) for m,p in zip(mv,prediction)])
        lines=['# 现成遥感标签分类器训练','',f"来源：{result['source']}",
            f"独立原图：训练{len(selected_train)}张，验证{len(selected_val)}张；两侧无重叠。演示图及其裁块来源全部留在验证侧。",
            f"原始训练目标{len(mt)}个，增强后{len(yt)}行；验证{len(mv)}个。",
            f"验证OA={evaluation['oa']:.4f}，Kappa={evaluation['kappa']:.4f}；手工特征基线OA={base_metrics['oa']:.4f}。",
            result['background_note'],result['inference_note'],
            '验证基于参考目标框裁块，不等于SAM区域分类精度或视频精度。建筑/农田/道路不在DOTA原始15类中，保留为人工自定义类别。','',
            '| 类别 | 训练数 | 验证数 | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|']
        for label in labels:
            m=evaluation['per_class'][label]
            lines.append(f"| {label} | {result['train_counts'].get(label,0)} | {result['validation_counts'].get(label,0)} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1-score']:.3f} |")
        (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        print(json.dumps({k:result[k] for k in ('train_objects','validation_objects','seconds')},ensure_ascii=False),flush=True)
        print('OA',evaluation['oa'],'Kappa',evaluation['kappa'],'model',out/'model.joblib',flush=True)
        return result
    finally:
        for handle in handles:handle.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archives',nargs='+',default=[str(p) for p in sorted((DATA_DIR/'images').glob('*.zip'))])
    ap.add_argument('--labels',default=str(DATA_DIR/'labelTxt-v1.0/labelTxt.zip'))
    ap.add_argument('--out',default=str(ROOT/'asset/dota_classifier'));ap.add_argument('--cap',type=int,default=1000)
    ap.add_argument('--val-cap',type=int,default=300);ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    if not args.archives:ap.error('没有找到图片压缩包，请设置 LAB2_DATA_DIR 或传入 --archives。')
    train(args.archives,args.labels,args.out,args.cap,args.val_cap,args.seed)


if __name__=='__main__':main()
