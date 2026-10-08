"""实验二 步骤四：遥感视频自动分析

把步骤一~三的分割/报告能力扩展到视频：
  抽关键帧 -> SAM 分割 -> 逐帧标注(JSON) -> 跨帧聚合 -> 视频级 Markdown 报告
  + 类别叠加可视化视频 annotated.mp4

用法：
  # 合成一段航拍运动测试视频（网络视频不可得时）
  python -m utils.video --synth-from dataset/images/P0000.png --synth-out dataset/videos/synth.mp4
  # 分析视频
  python -m utils.video --video dataset/videos/synth.mp4 --stride 25
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from utils.classifier import classify_annotation, load_model, write_json
from utils.config import CLASSIFIER_MODEL

import cv2
import numpy as np

from utils.report import generate_report
from utils.sam_segment import SamAutomaticMaskGenerator, build_sam, segment_image, visualize

# ---------------- 视频合成（模拟无人机航拍：平移+变焦） ----------------

def synth_video(image_path: str, out_path: str, seconds: int = 12, fps: int = 25,
                out_size: tuple = (1280, 720)) -> str:
    """从一张大图合成平移+缩放航拍视频，用于验证视频分析流水线。"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像 {image_path}")
    if seconds * fps < 2 or fps <= 0:
        raise ValueError("合成视频至少需要两帧。")
    H, W = img.shape[:2]
    vw, vh = out_size
    n = seconds * fps
    win_w0, win_h0 = int(W * 0.35), int(W * 0.35 * vh / vw)      # 起始窗口(远)
    win_w1, win_h1 = int(W * 0.65), int(W * 0.65 * vh / vw)      # 结束窗口(近)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if not writer.isOpened():
        raise RuntimeError(f"无法写入视频 {out_path}")
    for i in range(n):
        t = i / (n - 1)
        ww = int(win_w0 + (win_w1 - win_w0) * t)
        wh = int(win_h0 + (win_h1 - win_h0) * t)
        x = int((W - ww) * (0.15 + 0.7 * t))   # 从左往右平移
        y = int((H - wh) * (0.30 + 0.4 * t))
        crop = img[y:y + wh, x:x + ww]
        frame = cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
        writer.write(frame)
    writer.release()
    print(f"[synth] {out_path}  {n}帧 {fps}fps {out_size[0]}x{out_size[1]}")
    return out_path

# ---------------- 关键帧抽取 ----------------

def extract_frames(video_path: str, stride: int, max_frames: int = 0) -> list:
    """按间隔抽关键帧，返回 [(frame_idx, t_sec, bgr)]。"""
    if stride < 1 or max_frames < 0:
        raise ValueError("stride 至少为1，max_frames 不能为负数。")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频 {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    declared_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        current = idx
        idx += 1
        if current % stride == 0:
            frames.append((current, current / fps, frame))
            if max_frames and len(frames) >= max_frames:
                break
    # 注意：CAP_PROP_FRAME_* 在部分编解码下返回 -1，以首帧实际尺寸为准
    meta = {"fps": fps, "total_frames": max(declared_total, idx), "decoded_frames": idx,
            "width": frames[0][2].shape[1] if frames else 0,
            "height": frames[0][2].shape[0] if frames else 0}
    cap.release()
    if not frames:
        raise ValueError("视频没有可读取的关键帧。")
    return frames, meta

# ---------------- 视频分析主流程 ----------------

def analyze_video(video_path: str, out_dir: str, stride: int = 25,
                  max_frames: int = 0, points_per_side: int = 16,
                  model_path=None, image_dir="dataset/images", annotation_dir="results") -> dict:
    name = os.path.splitext(os.path.basename(video_path))[0]
    out = os.path.join(out_dir, f"video_{name}")
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    os.makedirs(os.path.join(out, "overlays"), exist_ok=True)

    generator = None  # 已有修正标注时无需重新分割
    model = load_model(model_path) if model_path else None
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(annotation_dir, exist_ok=True)

    frames, meta = extract_frames(video_path, stride, max_frames)
    print(f"视频 {meta['width']}x{meta['height']} @ {meta['fps']:.1f}fps, "
          f"抽 {len(frames)} 个关键帧 (stride={stride})")

    vw, vh = meta["width"], meta["height"]
    writer = cv2.VideoWriter(os.path.join(out, "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             max(meta["fps"] / stride, 1), (vw, vh))

    if not writer.isOpened():
        raise RuntimeError("无法创建输出视频。")

    frame_reports = []       # 每关键帧的 report dict
    cls_frames = defaultdict(set)   # 类别 -> 出现的关键帧序号集合
    cls_total = defaultdict(int)    # 类别 -> 累计实例次数
    region_votes = defaultdict(float)

    t0 = time.time()
    for k, (fidx, tsec, frame) in enumerate(frames):
        stem = f"{name}_f{fidx:05d}"
        image_name = stem + ".png"
        cv2.imwrite(os.path.join(image_dir, image_name), frame)
        editable = Path(annotation_dir) / (stem + ".json")
        if editable.exists():
            ann = json.loads(editable.read_text(encoding="utf-8"))
            if ann.get("width") != vw or ann.get("height") != vh:
                raise ValueError(f"旧关键帧标注尺寸不匹配：{editable}")
        else:
            if generator is None:
                print("加载 SAM ...")
                generator = SamAutomaticMaskGenerator(build_sam(_device()),
                    points_per_side=points_per_side, pred_iou_thresh=.86,
                    stability_score_thresh=.9, min_mask_region_area=100)
            ann = {"image": image_name, "width": vw, "height": vh,
                   "instances": segment_image(frame, generator)}
        ann.update(image=image_name, source_video=os.path.abspath(video_path), frame_index=fidx)
        if model:
            ann = classify_annotation(frame, ann, model, preserve_manual=True)
        instances = ann["instances"]
        write_json(editable, ann)
        write_json(os.path.join(out, "frames", f"f{fidx:05d}.json"), ann)

        rep = generate_report(ann)
        Path(out, "frames", f"f{fidx:05d}.md").write_text(rep["markdown"], encoding="utf-8")
        rep["t_sec"] = round(tsec, 2)
        frame_reports.append(rep)
        for c, cnt in rep["cls_count"].items():
            cls_frames[c].add(k)
            cls_total[c] += cnt
        for r, s in rep["region_scores"].items():
            region_votes[r] += s

        vis = visualize(frame, instances)
        cv2.imwrite(os.path.join(out, "overlays", f"f{fidx:05d}.jpg"), vis)
        writer.write(vis)
        print(f"  [{k + 1}/{len(frames)}] t={tsec:.1f}s 实例 {len(instances)}"
              f" 区域={rep['region_type']}")

    writer.release()

    # ---- 跨帧聚合 ----
    counts = [r["total"] for r in frame_reports]
    labeled = [r["labeled"] for r in frame_reports]
    ranked_regions = sorted(region_votes.items(), key=lambda kv: -kv[1])
    half = max(len(counts) // 2, 1)
    trend = ("前段平均实例数 %.0f -> 后段 %.0f，目标数量%s"
             % (np.mean(counts[:half]), np.mean(counts[half:] or counts),
                "呈增多态势" if np.mean(counts[half:] or counts) > np.mean(counts[:half]) * 1.1
                else "呈减少态势" if np.mean(counts[half:] or counts) < np.mean(counts[:half]) * 0.9
                else "基本稳定"))

    if len(counts) == 1:
        trend = "仅一个关键帧，无法判断时序趋势"

    md = ["# 遥感视频自动分析报告", "",
          "类别累计数为各关键帧实例次数，不是跨帧去重后的目标数量。",
          "分类来源：" + ("机器学习预测与保留的人工修正；预测类别仍需核验。" if model else "已保存的关键帧标注；未标注实例不参与类别判断。"),
          f"- 视频：{os.path.basename(video_path)}"
          f"（{meta['width']}x{meta['height']}，{meta['fps']:.0f}fps，"
          f"{meta['total_frames']} 帧 / {meta['total_frames'] / meta['fps']:.1f}s）",
          f"- 分析关键帧：{len(frames)} 个（每 {stride} 帧 1 帧，"
          f"处理耗时 {time.time() - t0:.0f}s）",
          f"- 单帧实例数：均值 {np.mean(counts):.0f}，范围 {min(counts)}~{max(counts)}",
          "",
          "## 区域类型判定（全视频投票）", ""]
    if ranked_regions:
        for r, s in ranked_regions[:3]:
            mark = " **<- 全片主判定**" if r == ranked_regions[0][0] else ""
            md.append(f"- {r}（累计评分 {s:.0f}）{mark}")
    else:
        md.append("- 关键帧中无已标注类别，无法判定。建议对若干关键帧先做人工标注。")
    md += ["", "## 类别出现统计", "",
           "| 类别 | 出现帧数 | 累计实例数 |", "|---|---|---|"]
    for c in sorted(cls_total, key=lambda c: -cls_total[c]):
        md.append(f"| {c} | {len(cls_frames[c])}/{len(frames)} | {cls_total[c]} |")
    if not cls_total:
        md.append("| (无已标注类别) | - | - |")
    md += ["", "## 时序变化分析", "",
           f"- {trend}。"]
    # 每帧明细表
    md += ["", "## 关键帧明细", "",
           "| 帧号 | 时间(s) | 实例数 | 已标注 | 主判定区域 |", "|---|---|---|---|---|"]
    for k, r in enumerate(frame_reports):
        md.append(f"| f{frames[k][0]:05d} | {r['t_sec']} | {r['total']} | "
                  f"{r['labeled']} | {r['region_type'] or '-'} |")
    md += ["", "## 逐帧地貌描述（首/中/末关键帧）", ""]
    for k in sorted({0, len(frame_reports) // 2, len(frame_reports) - 1}):
        r = frame_reports[k]
        md.append(f"**t={r['t_sec']}s**：{r['markdown'].split('## 地貌与区域分析')[-1].strip()}")
        md.append("")

    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    summary = {"video": os.path.basename(video_path), "meta": meta,
               "keyframes": len(frames), "stride": stride,
               "avg_instances": round(float(np.mean(counts)), 1),
               "region_ranking": [(r, round(s, 1)) for r, s in ranked_regions],
               "cls_total": dict(cls_total), "trend": trend,
               "outputs": {"report": os.path.join(out, "report.md"),
                           "annotated_video": os.path.join(out, "annotated.mp4")}}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print(f"[done] 报告 -> {summary['outputs']['report']}")
    print(f"       可视化视频 -> {summary['outputs']['annotated_video']}")
    return summary


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser(description="遥感视频自动分析（实验二 步骤四）")
    ap.add_argument("--video", help="待分析视频路径")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", default=str(CLASSIFIER_MODEL), help="已训练的区域分类器")
    ap.add_argument("--image-dir", default="dataset/images", help="导出关键帧供 UI 标注")
    ap.add_argument("--annotation-dir", default="results", help="读取并保存可修正的关键帧标注")
    ap.add_argument("--synth-from", help="用单张遥感大图合成航拍测试视频")
    ap.add_argument("--synth-out", default="dataset/videos/synth.mp4")
    ap.add_argument("--stride", type=int, default=25, help="关键帧间隔")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--points-per-side", type=int, default=16)
    args = ap.parse_args()

    if args.synth_from:
        os.makedirs(os.path.dirname(args.synth_out) or ".", exist_ok=True)
        synth_video(args.synth_from, args.synth_out)
        if not args.video:
            return
    if not args.video:
        ap.error("需要 --video 或 --synth-from")
    analyze_video(args.video, args.out, stride=args.stride,
                  max_frames=args.max_frames, points_per_side=args.points_per_side,
                  model_path=args.model, image_dir=args.image_dir, annotation_dir=args.annotation_dir)


if __name__ == "__main__":
    main()
