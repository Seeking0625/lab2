"""Polygon/mask operations shared by editing, features and evaluation."""
import copy
import cv2
import numpy as np


def mask_for(inst, shape, offset=(0, 0)):
    mask = np.zeros(shape[:2], np.uint8)
    shift = np.array(offset)
    for ring in inst.get('polygons', [inst.get('polygon', [])]):
        if len(ring) >= 3:
            cv2.fillPoly(mask, [np.rint(np.array(ring) - shift).astype(np.int32)], 1)
    for ring in inst.get('holes', []):
        pts = np.rint(np.array(ring) - shift).astype(np.int32)
        hole = np.zeros_like(mask)
        cv2.fillPoly(hole, [pts], 1)
        cv2.polylines(hole, [pts], True, 0, 1)
        mask[hole != 0] = 0
    return mask


def instances_from_mask(mask, template, offset=(0, 0)):
    n, labels = cv2.connectedComponents(mask.astype(np.uint8))
    result = []
    for k in range(1, n):
        part = (labels == k).astype(np.uint8)
        contours, hierarchy = cv2.findContours(part, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        polygons, holes = [], []
        for i, contour in enumerate(contours):
            if len(contour) < 3:
                continue
            ring = (contour.reshape(-1, 2) + np.array(offset)).tolist()
            (polygons if hierarchy[0][i][3] < 0 else holes).append(ring)
        if not polygons:
            continue
        x, y, w, h = cv2.boundingRect(part)
        inst = copy.deepcopy(template)
        inst.update(polygons=polygons, holes=holes, bbox=[x+offset[0], y+offset[1], w, h],
                    area=int(part.sum()))
        inst.pop('polygon', None)
        result.append(inst)
    return result


def split_instance(original, candidate, shape):
    base, cut = mask_for(original, shape), mask_for(candidate, shape)
    chosen = base & cut
    rest = base & (1-cut)
    if not chosen.any() or not rest.any():
        raise ValueError('候选必须将原实例分成至少两部分，请换一个提示点。')
    parts = instances_from_mask(chosen, original) + instances_from_mask(rest, original)
    if len(parts) < 2:
        raise ValueError('分割区域过小，请换一个提示点。')
    # Preserve every pixel; contours of one-pixel fragments cannot encode a polygon.
    rebuilt = np.zeros(shape[:2], np.uint8)
    for part in parts:
        rebuilt |= mask_for(part, shape)
    if not np.array_equal(rebuilt, base):
        raise ValueError('候选含无法保存为多边形的细小碎片，请调整提示点。')
    return parts


def merge_instances(instances, shape):
    classes = {i['class'] for i in instances if i['class'] != 'unlabeled'}
    if len(classes) > 1:
        raise ValueError('请先将待合并实例改成相同类别。')
    union = np.zeros(shape[:2], np.uint8)
    for inst in instances:
        union |= mask_for(inst, shape)
    pieces = instances_from_mask(union, instances[0])
    if not pieces:
        raise ValueError('没有可合并的有效区域。')
    x, y, w, h = cv2.boundingRect(union)
    return dict(instances[0], **{'class': next(iter(classes), 'unlabeled'),
                'label_source': 'model' if any(i.get('label_source') == 'model' for i in instances) else 'manual',
                'polygons': [r for p in pieces for r in p['polygons']],
                'holes': [r for p in pieces for r in p['holes']],
                'area': int(union.sum()), 'bbox': [x,y,w,h]})
