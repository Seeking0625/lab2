"""Supervised region classification; splits precede augmentation, predictions are not ground truth."""
import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from utils.geometry import mask_for

VERSION = 1


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as f:
        temporary = f.name
        f.write(text)
    os.replace(temporary, path)


def region_patch(image, inst):
    x,y,w,h = map(int, inst['bbox'])
    x0,y0 = max(0,x),max(0,y)
    x1,y1 = min(image.shape[1],x+w),min(image.shape[0],y+h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('实例 bbox 超出图像。')
    patch = image[y0:y1,x0:x1]
    mask = mask_for(inst, patch.shape[:2], (x0,y0))
    if not mask.any():
        raise ValueError('实例没有有效多边形像素。')
    return patch, mask


def features(patch, mask, scale=1.0, flip=False):
    # Scale augmentation changes both apparent size and resampling/texture.
    h, w = max(1, round(mask.shape[0]*scale)), max(1, round(mask.shape[1]*scale))
    # Large farmland/building masks must not allocate gigabytes during 3x augmentation.
    compute_scale = min(scale, 512/max(patch.shape[:2]))
    if compute_scale != 1:
        size = (max(1, round(patch.shape[1]*compute_scale)), max(1, round(patch.shape[0]*compute_scale)))
        patch = cv2.resize(patch, size, interpolation=cv2.INTER_LINEAR if compute_scale>1 else cv2.INTER_AREA)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    if flip:
        patch, mask = patch[:, ::-1].copy(), mask[:, ::-1].copy()
    pix = mask.astype(bool)
    if not pix.any():
        raise ValueError('缩放后实例为空。')
    result = [np.log1p(w), np.log1p(h), float(mask.mean()), w/max(h,1)]
    for space in (patch, cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)):
        values=space[pix].astype(np.float32)/255
        result.extend(values.mean(axis=0)); result.extend(values.std(axis=0))
        for channel in range(3):
            hist=np.histogram(values[:,channel],bins=8,range=(0,1))[0].astype(float)
            result.extend(hist/max(hist.sum(),1))
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0); gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    grad=np.sqrt(gx*gx+gy*gy)[pix]
    result.extend([float(grad.mean()),float(grad.std())])
    return np.asarray(result,dtype=np.float32)


def load_samples(annotation_dir, image_dir, image_names=None):
    samples=[]
    seen=set()
    for path in sorted(Path(annotation_dir).glob('*.json')):
        ann=json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(ann,dict) or 'instances' not in ann or 'image' not in ann:
            continue
        if image_names is not None and Path(ann['image']).name not in image_names:
            continue
        labeled=[(j,i) for j,i in enumerate(ann['instances'])
                 if i.get('class','unlabeled') != 'unlabeled' and i.get('label_source','manual') == 'manual']
        if not labeled:
            continue
        image_path=Path(image_dir)/Path(ann['image']).name
        image=cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'找不到标注对应图像：{image_path}')
        digest=hashlib.sha256(image_path.read_bytes()).hexdigest()
        group=ann.get('source_video') or ann.get('source_image_hash') or digest
        for j,inst in labeled:
            key=(digest,json.dumps(inst.get('polygons',inst.get('polygon')),sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            patch,mask=region_patch(image,inst)
            samples.append(dict(id=f'{path.name}:{j}',image=ann['image'],group=group,
                                label=inst['class'],patch=patch,mask=mask))
    return samples


def split_samples(samples, test_size=0.25, seed=42, split='image'):
    if not 0 < test_size < 1:
        raise ValueError('测试集比例必须在 0 和 1 之间。')
    y=np.array([s['label'] for s in samples]); indices=np.arange(len(samples))
    labels=set(y)
    if len(labels)<2 or min(Counter(y).values(),default=0)<2:
        raise ValueError('至少需要两个类别，每类至少两个已人工确认的实例；建议每类在多张图片中标注。')
    if split=='instance':
        try:
            train,test=train_test_split(indices,test_size=test_size,random_state=seed,stratify=y)
        except ValueError as e:
            raise ValueError(f'样本不足以分层划分，请增加标注或调整测试比例：{e}') from e
    else:
        groups=[s['group'] for s in samples]
        if len(set(groups))<2:
            raise ValueError('按图像划分至少需要两张不同图像（视频按源视频分组）。')
        train=test=None
        # Choose only by label coverage, never by model/test scores.
        for a,b in GroupShuffleSplit(n_splits=200,test_size=test_size,random_state=seed).split(indices,y,groups):
            if set(y[a])==labels and set(y[b])==labels:
                train,test=a,b
                break
        if train is None:
            raise ValueError('无法让训练集和测试集都包含全部类别，请在更多独立图像中标注每个类别。')
    return train,test


def metrics(y, pred, labels):
    kappa=float(cohen_kappa_score(y,pred,labels=labels))
    return {'oa':float(accuracy_score(y,pred)), 'kappa': kappa if np.isfinite(kappa) else None,
            'labels':list(labels),'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist(),
            'per_class':classification_report(y,pred,labels=labels,output_dict=True,zero_division=0),
            'n':len(y)}


def train_classifier(annotation_dir, image_dir, out_dir, test_size=.25, seed=42, split='image', image_names=None):
    samples=load_samples(annotation_dir,image_dir,image_names)
    train,test=split_samples(samples,test_size,seed,split)
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    labels=sorted({s['label'] for s in samples})
    manifest={'seed':seed,'split':split,'test_size_requested':test_size,
              'included_images':sorted(image_names) if image_names is not None else None,
              'warning':'同图像实例可能高度相关，不能代表跨图像泛化。' if split=='instance' else '',
              'train':[ {k:samples[i][k] for k in ('id','image','group','label')} for i in train],
              'test':[ {k:samples[i][k] for k in ('id','image','group','label')} for i in test]}
    write_json(out/'split.json',manifest)
    evaluation={'split':split,'seed':seed,'train_instances':len(train),'test_instances':len(test),
                'labels':labels,'warning':manifest['warning'],'models':{}}
    for variant,scales in [('baseline',[1.]),('augmented',[.5,1.,2.,3.])]:
        X=[]; y=[]
        for i in train:
            s=samples[i]
            for scale in scales:
                for flip in ([False,True] if variant=='augmented' else [False]):
                    X.append(features(s['patch'],s['mask'],scale,flip)); y.append(s['label'])
        model=RandomForestClassifier(n_estimators=160,class_weight='balanced',random_state=seed,n_jobs=2)
        model.fit(X,y)
        bundle={'version':VERSION,'model':model,'labels':labels,'variant':variant,
                'train_groups':sorted({samples[i]['group'] for i in train}), 'seed':seed}
        joblib.dump(bundle,out/f'{variant}.joblib')
        by_scale={}
        for scale in (1,2,3):
            Xtest=[features(samples[i]['patch'],samples[i]['mask'],scale) for i in test]
            pred=model.predict(Xtest); truth=[samples[i]['label'] for i in test]
            by_scale[str(scale)]=metrics(truth,pred,labels)
            write_json(out/f'predictions_{variant}_{scale}x.json',[
                {'id':samples[i]['id'],'truth':t,'prediction':str(p)} for i,t,p in zip(test,truth,pred)])
        evaluation['models'][variant]=by_scale
    write_json(out/'evaluation.json',evaluation)
    lines=['# 区域分类训练与测试','',f'训练 {len(train)} 实例；测试 {len(test)} 实例；划分方式 {split}。',
           manifest['warning'],'测试实例及其增强版本从不参与训练；增强只在划分后的训练集进行。',
           '以下尺度测试固定真实实例边界，仅评估分类器；完整 SAM 流程对比请运行尺度实验。','',
           '| 模型 | 面积倍率 | OA | Kappa |','|---|---:|---:|---:|']
    for variant,values in evaluation['models'].items():
        for scale,m in values.items():
            lines.append(f"| {variant} | {int(scale)**2} | {m['oa']:.4f} | {m['kappa']} |")
    base=evaluation['models']['baseline']; aug=evaluation['models']['augmented']
    for scale in ('2','3'):
        lines.append(f"\n面积 {int(scale)**2} 倍：基线相对原尺度 OA 变化 {base[scale]['oa']-base['1']['oa']:+.4f}；增强相对同尺度基线变化 {aug[scale]['oa']-base[scale]['oa']:+.4f}。")
    lines.append('\n上述数值只适用于本次标注与测试划分，数据增强不保证提升精度。')
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    return evaluation


def load_model(path):
    bundle=joblib.load(path)
    if bundle.get('version')!=VERSION:
        raise ValueError('分类模型特征版本不兼容，请重新训练。')
    return bundle


def classify_annotation(image, ann, bundle, preserve_manual=True):
    result=copy.deepcopy(ann)
    pending=[inst for inst in result['instances'] if not (preserve_manual and
        inst.get('class','unlabeled')!='unlabeled' and inst.get('label_source','manual')=='manual')]
    if not pending:return result
    regions=[region_patch(image,inst) for inst in pending]
    if bundle.get('feature_backend')=='resnet18+handcrafted':
        from utils.appearance import appearance_features
        X=appearance_features(regions)
    else:X=np.array([features(patch,mask) for patch,mask in regions])
    for inst,probs in zip(pending,bundle['model'].predict_proba(X)):
        order=np.argsort(probs)[::-1];idx=int(order[0]);score=float(probs[idx])
        margin=score-float(probs[order[1]]) if len(order)>1 else score
        accepted=score>=bundle.get('reject_threshold',0) and margin>=bundle.get('min_margin',0)
        inst.update({'class':str(bundle['model'].classes_[idx]) if accepted else 'unlabeled',
                     'class_score':score,'label_source':'model','class_candidates':[
                     {'class':str(bundle['model'].classes_[i]),'score':float(probs[i])} for i in order[:3]]})
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    tr=sub.add_parser('train')
    tr.add_argument('--annotations',default='results'); tr.add_argument('--images',default='dataset/images')
    tr.add_argument('--out',default='results/classifier'); tr.add_argument('--test-size',type=float,default=.25)
    tr.add_argument('--include-images', nargs='+', help='只使用列出的图像进行训练和测试')
    tr.add_argument('--seed',type=int,default=42); tr.add_argument('--split',choices=['image','instance'],default='image')
    pr=sub.add_parser('predict')
    pr.add_argument('--image',required=True); pr.add_argument('--annotation',required=True)
    pr.add_argument('--model',default='results/classifier/augmented.joblib'); pr.add_argument('--out',required=True)
    args=ap.parse_args()
    if args.command=='train':
        result=train_classifier(args.annotations,args.images,args.out,args.test_size,args.seed,args.split,args.include_images)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:
        from utils.sam_segment import visualize
        from utils.report import generate_report
        image=cv2.imread(args.image)
        if image is None: raise ValueError('无法读取图片。')
        ann=json.loads(Path(args.annotation).read_text(encoding='utf-8'))
        result=classify_annotation(image,ann,load_model(args.model))
        write_json(args.out,result)
        cv2.imwrite(str(Path(args.out).with_suffix('.jpg')),visualize(image,result['instances']))
        Path(args.out).with_suffix('.md').write_text(generate_report(result)['markdown'],encoding='utf-8')


if __name__=='__main__':
    main()
