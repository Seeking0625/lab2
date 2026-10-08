"""Frozen ImageNet ResNet18 appearance features plus region geometry/color features."""
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models import resnet18

_ENCODER=None
_DEVICE=None
from utils.config import APPEARANCE_WEIGHTS
WEIGHTS=APPEARANCE_WEIGHTS


def encoder():
    global _ENCODER,_DEVICE
    if _ENCODER is None:
        if not WEIGHTS.exists(): raise FileNotFoundError(f'分类外观权重缺失：{WEIGHTS}')
        _DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
        model=resnet18(weights=None)
        model.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True))
        _ENCODER=torch.nn.Sequential(*list(model.children())[:-1]).eval().to(_DEVICE)
        _ENCODER.requires_grad_(False)
    return _ENCODER,_DEVICE


def prepare_patch(patch,mask):
    # Letterbox avoids stretching long rotated aircraft/vehicles.
    h,w=mask.shape
    rgb=cv2.cvtColor(patch,cv2.COLOR_BGR2RGB).copy()
    rgb[mask==0]=[124,116,104]
    ratio=224/max(h,w);nw,nh=max(1,round(w*ratio)),max(1,round(h*ratio))
    resized=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA if ratio<1 else cv2.INTER_LINEAR)
    canvas=np.full((224,224,3),[124,116,104],np.uint8)
    x,y=(224-nw)//2,(224-nh)//2;canvas[y:y+nh,x:x+nw]=resized
    array=canvas.astype(np.float32)/255
    array=(array-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    return array.transpose(2,0,1)


def appearance_features(regions,batch_size=48,rotate=False):
    from utils.classifier import features
    model,device=encoder();output=[]
    for start in range(0,len(regions),batch_size):
        batch=regions[start:start+batch_size]
        arrays=[];hand=[]
        for patch,mask in batch:
            if rotate:patch=np.rot90(patch).copy();mask=np.rot90(mask).copy()
            arrays.append(prepare_patch(patch,mask));hand.append(features(patch,mask))
        with torch.inference_mode():
            embedding=model(torch.from_numpy(np.stack(arrays)).to(device)).flatten(1).cpu().numpy()
        embedding/=np.maximum(np.linalg.norm(embedding,axis=1,keepdims=True),1e-8)
        output.append(np.concatenate([embedding,np.array(hand)],axis=1))
    return np.concatenate(output) if output else np.empty((0,578),np.float32)
