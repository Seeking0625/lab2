"""实验四 步骤四：遥感视频自动分析

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

import cv2
import numpy as np

from utils.report import generate_report
from utils.sam_segment import SamAutomaticMaskGenerator, build_sam, segment_image, visualize

# ---------------- 视频合成（模拟无人机航拍：平移+变焦） ----------------

def synth_video(image_path: str, out_path: str, seconds: int = 12, fps: int = 25,
                out_size: tuple = (1280, 720)) -> str:
    """从一张大图合成平移+缩放航拍视频，用于验证视频分析流水线。"""
    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    vw, vh = out_size
    n = seconds * fps
    win_w0, win_h0 = int(W * 0.35), int(W * 0.35 * vh / vw)      # 起始窗口(远)
    win_w1, win_h1 = int(W * 0.65), int(W * 0.65 * vh / vw)      # 结束窗口(近)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频 {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append((idx, idx / fps, frame))
            if max_frames and len(frames) >= max_frames:
                break
        idx += 1
    # 注意：CAP_PROP_FRAME_* 在部分编解码下返回 -1，以首帧实际尺寸为准
    meta = {"fps": fps, "total_frames": idx,
            "width": frames[0][2].shape[1] if frames else 0,
            "height": frames[0][2].shape[0] if frames else 0}
    cap.release()
    return frames, meta

# ---------------- 视频分析主流程 ----------------

def analyze_video(video_path: str, out_dir: str, stride: int = 25,
                  max_frames: int = 0, points_per_side: int = 16) -> dict:
    name = os.path.splitext(os.path.basename(video_path))[0]
    out = os.path.join("results", f"video_{name}")
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    os.makedirs(os.path.join(out, "overlays"), exist_ok=True)

    print("加载 SAM ...")
    sam = build_sam(_device())
    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=points_per_side, pred_iou_thresh=0.86,
        stability_score_thresh=0.9, min_mask_region_area=100)

    frames, meta = extract_frames(video_path, stride, max_frames)
    print(f"视频 {meta['width']}x{meta['height']} @ {meta['fps']:.1f}fps, "
          f"抽 {len(frames)} 个关键帧 (stride={stride})")

    vw, vh = meta["width"], meta["height"]
    writer = cv2.VideoWriter(os.path.join(out, "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             max(meta["fps"] / stride, 1), (vw, vh))

    frame_reports = []       # 每关键帧的 report dict
    cls_frames = defaultdict(set)   # 类别 -> 出现的关键帧序号集合
    cls_total = defaultdict(int)    # 类别 -> 累计实例次数
    region_votes = defaultdict(float)

    t0 = time.time()
    for k, (fidx, tsec, frame) in enumerate(frames):
        instances = segment_image(frame, generator)
        ann = {"image": f"{name}_f{fidx:05d}.jpg", "width": vw, "height": vh,
               "instances": instances}
        with open(os.path.join(out, "frames", f"f{fidx:05d}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(ann, f, ensure_ascii=False)

        rep = generate_report(ann)
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
             % (np.mean(counts[:half]), np.mean(counts[half:]),
                "呈增多态势" if np.mean(counts[half:]) > np.mean(counts[:half]) * 1.1
                else "呈减少态势" if np.mean(counts[half:]) < np.mean(counts[:half]) * 0.9
                else "基本稳定"))

    md = ["# 遥感视频自动分析报告", "",
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
    ap = argparse.ArgumentParser(description="遥感视频自动分析（实验四 步骤四）")
    ap.add_argument("--video", help="待分析视频路径")
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
    analyze_video(args.video, "results", stride=args.stride,
                  max_frames=args.max_frames, points_per_side=args.points_per_side)


if __name__ == "__main__":
    main()
