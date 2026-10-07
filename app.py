"""实验四 步骤二：SAM 半自动标注修正 UI —— Flask 后端

启动（需 GPU，沙箱外运行）：
    /home/seeking/aisj/lab1/.venv/bin/python app.py
浏览器打开 http://127.0.0.1:5000
"""
import json
import os

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from segment_anything import SamPredictor

from utils.sam_segment import _mask_to_polygon, build_sam, segment_image

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE, "dataset", "images")
RESULT_DIR = os.path.join(BASE, "results")
CLASS_FILE = os.path.join(BASE, "classes.json")
CHECKPOINT = os.path.join(BASE, "asset", "sam_vit_b_01ec64.pth")

# iSAID 15 类作为默认类别表
DEFAULT_CLASSES = [
    {"name": "airplane 飞机", "color": "#e6194b"},
    {"name": "small-vehicle 小型车辆", "color": "#3cb44b"},
    {"name": "large-vehicle 大型车辆", "color": "#4363d8"},
    {"name": "storage-tank 储油罐", "color": "#ffe119"},
    {"name": "swimming-pool 泳池", "color": "#42d4f4"},
    {"name": "harbor 港口", "color": "#911eb4"},
    {"name": "bridge 桥梁", "color": "#f032e6"},
    {"name": "building 建筑", "color": "#469990"},
    {"name": "farmland 农田", "color": "#9a6324"},
    {"name": "road 道路", "color": "#808000"},
    {"name": "helicopter 直升机", "color": "#800000"},
    {"name": "roundabout 环岛", "color": "#aaffc3"},
    {"name": "soccer-ball-field 足球场", "color": "#808fff"},
    {"name": "basketball-court 篮球场", "color": "#ffd8b1"},
    {"name": "tennis-court 网球场", "color": "#e6beff"},
]

app = Flask(__name__, static_folder="static", static_url_path="/static")

_sam = None
_predictor = None


def get_predictor():
    global _sam, _predictor
    if _predictor is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"加载 SAM 到 {device} ...")
        _sam = build_sam(device)
        _predictor = SamPredictor(_sam)
    return _predictor


def _ensure_classes():
    if not os.path.exists(CLASS_FILE):
        with open(CLASS_FILE, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CLASSES, f, ensure_ascii=False, indent=1)
    with open(CLASS_FILE, encoding="utf-8") as f:
        return json.load(f)


def _ann_path(name):
    return os.path.join(RESULT_DIR, f"{os.path.splitext(name)[0]}.json")


def _image_path(name):
    base = os.path.splitext(name)[0]  # 兼容带/不带扩展名
    for ext in (".png", ".jpg", ".tif"):
        p = os.path.join(IMAGE_DIR, base + ext)
        if os.path.exists(p):
            return p
    return None


def _crop_with_margin(image, box, margin=0.25, min_size=768):
    """按 bbox 外扩裁剪，返回 (crop, offset_x, offset_y)。"""
    x, y, w, h = box
    H, W = image.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    if min_size:
        mx = max(mx, (min_size - w) // 2)
        my = max(my, (min_size - h) // 2)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    return image[y0:y1, x0:x1], int(x0), int(y0)


def _candidates_from_masks(crop, masks, scores, ox, oy):
    """SAM 掩码 -> 候选实例（全局坐标多边形），按分数降序。"""
    cands = []
    for m, s in zip(masks, scores):
        if s < 0.5:
            continue
        poly = _mask_to_polygon(m)
        if poly is None:
            continue
        poly = [[int(px + ox), int(py + oy)] for px, py in poly]
        bx, by, bw, bh = cv2.boundingRect(m.astype(np.uint8))
        cands.append({
            "class": "unlabeled",
            "bbox": [int(bx + ox), int(by + oy), int(bw), int(bh)],
            "area": int(m.sum()),
            "score": round(float(s), 4),
            "polygons": [poly],
        })
    cands.sort(key=lambda c: -c["score"])
    return cands


# ---------------- 页面与静态资源 ----------------

@app.get("/")
def index():
    return send_from_directory("static", "index.html")


# ---------------- 数据 API ----------------

@app.get("/api/images")
def api_images():
    names = sorted(f for f in os.listdir(IMAGE_DIR)
                   if f.lower().endswith((".png", ".jpg", ".tif")))
    return jsonify(names)


@app.get("/api/image/<name>")
def api_image(name):
    p = _image_path(name)
    if p is None:
        return jsonify({"error": "image not found"}), 404
    return send_file(p, conditional=True)


@app.get("/api/annotation/<name>")
def api_annotation(name):
    p = _ann_path(name)
    if not os.path.exists(p):
        return jsonify({"error": "annotation not found"}), 404
    return send_file(p)


@app.post("/api/annotation/<name>")
def api_save_annotation(name):
    data = request.get_json(force=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(_ann_path(name), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return jsonify({"saved": True, "instances": len(data.get("instances", []))})


@app.post("/api/segment/<name>")
def api_segment(name):
    """对该图运行 SAM 自动分割并保存（较慢，约 0.5~1.5 分钟/张）。"""
    from utils.sam_segment import SamAutomaticMaskGenerator  # 延迟导入
    p = _image_path(name)
    if p is None:
        return jsonify({"error": "image not found"}), 404
    image = cv2.imread(p)
    predictor = get_predictor()
    generator = SamAutomaticMaskGenerator(
        predictor.model, points_per_side=16,
        pred_iou_thresh=0.86, stability_score_thresh=0.9, min_mask_region_area=100)
    instances = segment_image(image, generator)
    result = {"image": os.path.basename(p), "width": image.shape[1],
              "height": image.shape[0], "instances": instances}
    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(_ann_path(name), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return jsonify(result)


# ---------------- 类别 API ----------------

@app.get("/api/classes")
def api_classes():
    return jsonify(_ensure_classes())


@app.post("/api/classes")
def api_add_class():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    color = data.get("color") or "#%06x" % (hash(name) & 0xFFFFFF)
    if not name:
        return jsonify({"error": "name required"}), 400
    classes = _ensure_classes()
    if any(c["name"] == name for c in classes):
        return jsonify({"error": "class exists"}), 400
    classes.append({"name": name, "color": color})
    with open(CLASS_FILE, "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=1)
    return jsonify(classes)


# ---------------- SAM 交互式分割 API ----------------

@app.post("/api/sam/point")
def api_sam_point():
    """点提示分割：返回候选实例（多环 polygons，全局坐标）。

    body: {image, points: [[x, y, label]], bbox: [x,y,w,h](可选)}
    """
    data = request.get_json(force=True)
    p = _image_path(data["image"])
    if p is None:
        return jsonify({"error": "image not found"}), 404
    image = cv2.imread(p)
    points = data["points"]
    box = data.get("bbox")
    if box is None:
        cx, cy = points[0][0], points[0][1]
        r = 512
        box = [cx - r, cy - r, 2 * r, 2 * r]
    crop, ox, oy = _crop_with_margin(image, box)
    predictor = get_predictor()
    predictor.set_image(crop)
    coords = np.array([[pt[0] - ox, pt[1] - oy] for pt in points], dtype=np.float32)
    labels = np.array([pt[2] for pt in points], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=coords, point_labels=labels,
        multimask_output=True)
    return jsonify(_candidates_from_masks(crop, masks, scores, ox, oy))


@app.post("/api/sam/box")
def api_sam_box():
    """框提示分割：框内新目标 -> 候选实例。

    body: {image, box: [x,y,w,h]}
    """
    data = request.get_json(force=True)
    p = _image_path(data["image"])
    if p is None:
        return jsonify({"error": "image not found"}), 404
    image = cv2.imread(p)
    box = data["box"]
    crop, ox, oy = _crop_with_margin(image, box)
    predictor = get_predictor()
    predictor.set_image(crop)
    local_box = np.array([box[0] - ox, box[1] - oy, box[2], box[3]], dtype=np.float32)
    # 框提示 + 中心正点，稳定性更好
    coords = np.array([[box[0] - ox + box[2] / 2, box[1] - oy + box[3] / 2]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=coords, point_labels=labels,
        box=local_box, multimask_output=True)
    return jsonify(_candidates_from_masks(crop, masks, scores, ox, oy))


@app.get("/api/report/<name>")
def api_report(name):
    """由标注结果生成分析报告，同时落盘 results/report_<name>.md"""
    from utils.report import generate_report
    p = _ann_path(name)
    if not os.path.exists(p):
        return jsonify({"error": "annotation not found"}), 404
    with open(p, encoding="utf-8") as f:
        ann = json.load(f)
    rep = generate_report(ann, _ensure_classes())
    os.makedirs(RESULT_DIR, exist_ok=True)
    md_path = os.path.join(RESULT_DIR, f"report_{os.path.splitext(name)[0]}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(rep["markdown"])
    rep["markdown_file"] = os.path.basename(md_path)
    return jsonify(rep)


@app.get("/api/status")
def api_status():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
