"""SAM 遥感图像自动分割模块（实验四 步骤一）

对大尺寸遥感图像分块调用 SAM 自动掩码生成器，把结果映射回全图坐标，
去重后导出为多边形 JSON（实例 id / 类别 / bbox / 面积 / 置信度），
供步骤二标注修正 UI 直接读写。

用法：
    python -m utils.sam_segment --image dataset/images/P0000.png --vis
    python -m utils.sam_segment --dir dataset/images --max-images 5
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

CHECKPOINT = os.path.join(os.path.dirname(__file__), "..", "asset", "sam_vit_b_01ec64.pth")

# ---------------- 分割核心 ----------------

def build_sam(device: str = "cuda"):
    sam = sam_model_registry["vit_b"](checkpoint=os.path.abspath(CHECKPOINT))
    sam.to(device=device)
    return sam


def segment_image(image_bgr: np.ndarray, generator: SamAutomaticMaskGenerator,
                  tile_size: int = 1024, overlap: int = 64,
                  min_area: int = 100, score_thresh: float = 0.6) -> list:
    """分块自动分割，返回全图坐标下的实例列表（含多边形）。

    注意：每个掩码只保留多边形与 bbox，不缓存全图尺寸掩码数组，
    否则大图上千实例会撑爆内存（4000x4000 单掩码即 16MB）。
    """
    H, W = image_bgr.shape[:2]
    instances = []

    ys = list(range(0, max(H - tile_size, 0) + 1, tile_size - overlap))
    xs = list(range(0, max(W - tile_size, 0) + 1, tile_size - overlap))
    if ys[-1] != max(H - tile_size, 0):
        ys.append(max(H - tile_size, 0))
    if xs[-1] != max(W - tile_size, 0):
        xs.append(max(W - tile_size, 0))

    for y0 in ys:
        for x0 in xs:
            tile = image_bgr[y0:y0 + tile_size, x0:x0 + tile_size]
            for m in generator.generate(tile):
                if m["predicted_iou"] < score_thresh or m["area"] < min_area:
                    continue
                seg = m["segmentation"]
                poly = _mask_to_polygon(seg)
                if poly is None:
                    continue
                # 掩码坐标 + tile 偏移 -> 全图坐标
                poly = [[int(px + x0), int(py + y0)] for px, py in poly]
                bx, by, bw, bh = m["bbox"]
                instances.append({
                    "class": "unlabeled",
                    "bbox": [int(bx + x0), int(by + y0), int(bw), int(bh)],
                    "area": int(m["area"]),
                    "score": round(float(m["predicted_iou"]), 4),
                    "polygons": [poly],
                })

    instances = _nms_instances(instances, iou_thresh=0.6)
    for i, inst in enumerate(instances):
        inst["id"] = i
    return instances


def _nms_instances(instances: list, iou_thresh: float = 0.6) -> list:
    """按 bbox IoU 去除分块重叠区产生的重复实例（保留高分者）。"""
    instances = sorted(instances, key=lambda m: -m["score"])
    keep = []
    for m in instances:
        x, y, w, h = m["bbox"]
        ok = True
        for k in keep:
            kx, ky, kw, kh = k["bbox"]
            ix = max(0, min(x + w, kx + kw) - max(x, kx))
            iy = max(0, min(y + h, ky + kh) - max(y, ky))
            inter = ix * iy
            union = w * h + kw * kh - inter
            if union > 0 and inter / union > iou_thresh:
                ok = False
                break
        if ok:
            keep.append(m)
    return keep


def _mask_to_polygon(mask: np.ndarray, eps_ratio: float = 0.002) -> list | None:
    """掩码 -> 简化后的外轮廓多边形 [[x,y], ...]，失败返回 None。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 1:
        return None
    eps = eps_ratio * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
    return approx.tolist()

# ---------------- 可视化 ----------------

_PALETTE = np.array([
    [230, 25, 75], [60, 180, 75], [0, 130, 200], [255, 225, 25], [145, 30, 180],
    [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190], [0, 128, 128],
    [220, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
    [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128], [255, 255, 255],
], dtype=np.uint8)


def visualize(image_bgr: np.ndarray, instances: list, alpha: float = 0.45) -> np.ndarray:
    """实例多边形彩色叠加，用于快速质检。兼容 polygons 多环与旧版单环 polygon。"""
    canvas = image_bgr.copy()
    overlay = image_bgr.copy()
    for i, inst in enumerate(instances):
        color = _PALETTE[i % len(_PALETTE)].tolist()
        rings = inst.get("polygons") or [inst["polygon"]]
        pts_list = [np.array(r, dtype=np.int32) for r in rings]
        cv2.fillPoly(overlay, pts_list, color)
        for pts in pts_list:
            cv2.polylines(canvas, [pts], True, color, 2)
    return cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)

# ---------------- 命令行入口 ----------------

def main():
    ap = argparse.ArgumentParser(description="SAM 遥感图像批量自动分割")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", help="单张图像路径")
    g.add_argument("--dir", help="图像目录（批量）")
    ap.add_argument("--out", default="results", help="输出目录")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--points-per-side", type=int, default=16)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--max-images", type=int, default=0, help="批量时限制处理张数，0 为全部")
    ap.add_argument("--vis", action="store_true", help="同时输出可视化 PNG")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.vis:
        os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)

    images = [args.image] if args.image else sorted(
        os.path.join(args.dir, f) for f in os.listdir(args.dir)
        if f.lower().endswith((".png", ".jpg", ".tif")))
    if args.max_images > 0:
        images = images[:args.max_images]

    print(f"加载 SAM (vit_b) 到 {args.device} ...")
    sam = build_sam(args.device)
    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=args.points_per_side,
        pred_iou_thresh=0.86, stability_score_thresh=0.9,
        min_mask_region_area=args.min_area)

    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        t0 = time.time()
        image = cv2.imread(path)
        if image is None:
            print(f"[skip] 无法读取 {path}")
            continue
        instances = segment_image(image, generator, tile_size=args.tile_size,
                                  overlap=args.overlap, min_area=args.min_area)
        result = {"image": os.path.basename(path),
                  "width": image.shape[1], "height": image.shape[0],
                  "instances": instances}
        out_json = os.path.join(args.out, f"{name}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        msg = f"[{name}] {len(instances)} 实例, {time.time() - t0:.1f}s -> {out_json}"
        if args.vis:
            vis = visualize(image, instances)
            out_vis = os.path.join(args.out, "vis", f"{name}_vis.jpg")
            cv2.imwrite(out_vis, vis)
            msg += f" | vis -> {out_vis}"
        print(msg)


if __name__ == "__main__":
    main()
