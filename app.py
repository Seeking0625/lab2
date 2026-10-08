"""实验二 步骤二：SAM 半自动标注修正 UI —— Flask 后端

启动（GPU 可选）：
    conda activate mLearning
    python app.py
浏览器打开 http://127.0.0.1:5000
"""
import json
import os
import threading
import uuid
import zipfile
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, g, jsonify, request, send_file, send_from_directory
from segment_anything import SamPredictor

from utils.sam_segment import _mask_to_polygon, build_sam, segment_image
from utils import config

BASE = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = str(config.IMAGE_DIR)
RESULT_DIR = str(config.RESULT_DIR)
CLASS_FILE = os.path.join(BASE, "classes.json")
CHECKPOINT = str(config.SAM_CHECKPOINT)

def classification_model_path():
    reference=app.config.get('REFERENCE_MODEL_PATH', str(config.CLASSIFIER_MODEL))
    if reference and os.path.exists(reference):return reference
    legacy=os.path.join(RESULT_DIR,'classifier','augmented.joblib')
    return legacy if os.path.exists(legacy) else None

# 遥感目标类别及可自定义的场景类别
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
    {"name": "ship 船舶", "color": "#0088aa"},
    {"name": "baseball-diamond 棒球场", "color": "#aa6644"},
    {"name": "ground-track-field 田径场", "color": "#cc8800"},
    {"name": "other 背景/其他", "color": "#888888"},
]

app = Flask(__name__, static_folder="static", static_url_path="/static")
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

_sam = None
_predictor = None
_work_lock = threading.RLock()
_executor = ThreadPoolExecutor(max_workers=1)
_jobs = {}


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
    if Path(name).name != name:return None
    base = os.path.splitext(name)[0]  # 兼容带/不带扩展名
    for ext in (".png", ".jpg", ".tif"):
        p = os.path.join(IMAGE_DIR, base + ext)
        if os.path.exists(p):
            return p
    entry = archive_images().get(base)
    if entry:
        archive, member = entry
        import tempfile
        destination = Path(IMAGE_DIR) / (base + '.png')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            with tempfile.NamedTemporaryFile(dir=IMAGE_DIR, delete=False) as f:
                temporary = f.name
                f.write(z.read(member))
        os.replace(temporary, destination)
        return str(destination)
    return None


@lru_cache(maxsize=1)
def _archive_images():
    entries = {}
    for archive in sorted((config.DATA_DIR / 'images').glob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                if member.lower().endswith('.png'):
                    entries.setdefault(Path(member).stem, (str(archive), member))
    return entries


def archive_images():
    return _archive_images() if app.config.get('ARCHIVE_IMAGES_ENABLED', True) else {}


def _crop_with_margin(image, box, margin=0.25, min_size=256):
    """按 bbox 外扩裁剪，返回 (crop, offset_x, offset_y)。"""
    x, y, w, h = map(lambda v: round(float(v)), box)
    if w <= 0 or h <= 0:
        raise ValueError("框的宽高必须为正。")
    H, W = image.shape[:2]
    mx, my = int(w * margin), int(h * margin)
    if min_size:
        mx = max(mx, (min_size - w) // 2)
        my = max(my, (min_size - h) // 2)
    x0, y0 = max(0, x - mx), max(0, y - my)
    x1, y1 = min(W, x + w + mx), min(H, y + h + my)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("提示区域超出图像。")
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
    names = sorted(set(names) | {stem+'.png' for stem in archive_images()})
    return jsonify(names)


@app.get('/api/image-catalog')
def api_image_catalog():
    items = []
    for path in sorted(Path(IMAGE_DIR).glob('*')):
        if path.suffix.lower() not in ('.png', '.jpg', '.tif'): continue
        ann_path = Path(_ann_path(path.name))
        ann = json.loads(ann_path.read_text(encoding='utf-8')) if ann_path.exists() else {}
        kind = 'practice' if ann.get('recommended') else ('video' if ann.get('source_video') or '_f' in path.stem
                else 'crop' if ann.get('source_image') else 'original')
        items.append(dict(name=path.name, kind=kind, source=ann.get('source_image', ''),
                          count=len(ann.get('instances', [])), width=ann.get('width'), height=ann.get('height')))
    present = {Path(item['name']).stem for item in items}
    for stem, (archive, member) in archive_images().items():
        if stem not in present:
            items.append(dict(name=stem+'.png', kind='original', source=Path(archive).name,
                              count=0, width=None, height=None))
    return jsonify(sorted(items, key=lambda item:item['name']))


@app.post('/api/crop')
def api_crop():
    from utils.annotation_workflow import create_crop
    data = request.get_json()
    path = _image_path(data['image'])
    if path is None: raise ValueError('原图不存在。')
    # Use the current client annotation so unsaved corrections are represented accurately.
    parent = data.get('annotation') or {}
    return jsonify(create_crop(path, data['box'], IMAGE_DIR, RESULT_DIR, parent)), 201


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
    from utils.classifier import write_json
    data = request.get_json(force=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    write_json(_ann_path(name), data)
    return jsonify({"saved": True, "instances": len(data.get("instances", []))})


@app.post("/api/segment/<name>")
def api_segment(name):
    """对该图运行 SAM 自动分割并保存（较慢，约 0.5~1.5 分钟/张）。"""
    from utils.sam_segment import SamAutomaticMaskGenerator  # 延迟导入
    p = _image_path(name)
    if p is None:
        return jsonify({"error": "image not found"}), 404
    image = cv2.imread(p)
    from utils.annotation_workflow import PROFILES, filter_proposals
    from utils.classifier import write_json
    profile = (request.get_json(silent=True) or {}).get('profile', 'standard')
    if profile not in PROFILES:
        raise ValueError('未知分割策略。')
    settings = PROFILES[profile]
    predictor = get_predictor()
    generator = SamAutomaticMaskGenerator(predictor.model,
        points_per_side=settings['points_per_side'], pred_iou_thresh=settings['pred_iou_thresh'],
        stability_score_thresh=settings['stability_score_thresh'], min_mask_region_area=settings['min_area'])
    instances = filter_proposals(segment_image(image, generator, min_area=settings['min_area']), image.shape[:2], profile)
    for i, inst in enumerate(instances): inst['id'] = i
    result = {"image": os.path.basename(p), "width": image.shape[1],
              "height": image.shape[0], "instances": instances, "segmentation_profile": profile}
    previous = Path(_ann_path(name))
    if previous.exists():
        old = json.loads(previous.read_text(encoding='utf-8'))
        # Keep original labels reviewable when the user explicitly resegments.
        write_json(Path(RESULT_DIR) / 'backups' / f'{previous.stem}_{uuid.uuid4().hex}.json', old)
        for key in ('source_image', 'source_image_hash', 'source_box', 'source_video', 'frame_index', 'recommended'):
            if key in old: result[key] = old[key]
    model_path=classification_model_path()
    if model_path:
        from utils.classifier import classify_annotation, load_model
        result=classify_annotation(image,result,load_model(model_path))
    write_json(previous, result)
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
    predictor.set_image(crop, image_format="BGR")
    coords = np.array([[pt[0] - ox, pt[1] - oy] for pt in points], dtype=np.float32)
    labels = np.array([pt[2] for pt in points], dtype=np.int32)
    local_box = None
    if data.get('use_box'):
        local_box = np.array([box[0]-ox, box[1]-oy, box[0]-ox+box[2], box[1]-oy+box[3]], dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=coords, point_labels=labels, box=local_box,
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
    predictor.set_image(crop, image_format="BGR")
    # SAM 接收 XYXY；前端提交的是 XYWH。
    local_box = np.array([box[0] - ox, box[1] - oy,
                          box[0] - ox + box[2], box[1] - oy + box[3]], dtype=np.float32)
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



@app.errorhandler(ValueError)
def bad_input(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(FileNotFoundError)
def missing_file(error):
    return jsonify(error=str(error)), 404


@app.before_request
def serialize_inference():
    if request.path.startswith(('/api/sam/', '/api/segment/', '/api/classify/')) or (
            request.method == 'POST' and request.path.startswith('/api/annotation/')):
        if not _work_lock.acquire(blocking=False):
            return jsonify(error='另一个分析任务正在运行，请稍后重试。'), 409
        g.work_locked = True


@app.teardown_request
def release_inference(error):
    if getattr(g, 'work_locked', False):
        _work_lock.release()


@app.post('/api/edit/split')
def api_split():
    from utils.geometry import split_instance
    data = request.get_json()
    image = cv2.imread(_image_path(data['image']) or '')
    if image is None:
        raise ValueError('图像不存在。')
    return jsonify(split_instance(data['original'], data['candidate'], image.shape[:2]))


@app.post('/api/edit/merge')
def api_merge():
    from utils.geometry import merge_instances
    data = request.get_json()
    image = cv2.imread(_image_path(data['image']) or '')
    if image is None or len(data['instances']) < 2:
        raise ValueError('需要有效图像及至少两个实例。')
    return jsonify(merge_instances(data['instances'], image.shape[:2]))


@app.post('/api/classify/<name>')
def api_classify(name):
    from utils.classifier import classify_annotation, load_model
    model_path = classification_model_path()
    if not model_path:
        raise ValueError('分类模型不存在，请运行参考标签训练脚本。')
    image = cv2.imread(_image_path(name) or '')
    if image is None:
        raise ValueError('图像不存在。')
    ann = request.get_json()
    return jsonify(classify_annotation(image, ann, load_model(model_path)))


@app.get('/api/videos')
def api_videos():
    directory = Path(BASE) / 'dataset' / 'videos'
    return jsonify(sorted(p.name for p in directory.glob('*') if p.suffix.lower() in ('.mp4', '.avi', '.mov')))


@app.get('/api/artifacts/<path:name>')
def api_artifact(name):
    return send_from_directory(RESULT_DIR, name)


@app.get('/api/tasks/<job_id>')
def api_task(job_id):
    if job_id not in _jobs:
        return jsonify(error='任务不存在'), 404
    return jsonify(_jobs[job_id])


@app.post('/api/tasks')
def start_task():
    data = request.get_json()
    kind = data.get('kind')
    if kind not in ('train', 'scale', 'video'):
        raise ValueError('未知任务。')
    if any(j['status'] in ('queued', 'running') for j in _jobs.values()):
        return jsonify(error='已有分析任务运行中。'), 409
    job_id = uuid.uuid4().hex
    _jobs[job_id] = {'status': 'queued', 'kind': kind}

    def work():
        try:
            with _work_lock:
                _jobs[job_id]['status'] = 'running'
                model_dir = Path(RESULT_DIR) / 'classifier'
                if kind == 'train':
                    from utils.classifier import train_classifier
                    image_names = data.get('image_names')
                    if image_names is not None:
                        image_names = [Path(n).name for n in image_names]
                    result = train_classifier(RESULT_DIR, IMAGE_DIR, model_dir, image_names=image_names)
                    report = 'classifier/report.md'
                elif kind == 'scale':
                    from utils.scale_experiment import run_scale_experiment
                    p = _image_path(Path(data.get('image', '')).name)
                    if not p:
                        raise ValueError('请先选择图像。')
                    models = [model_dir / f'{v}.joblib' for v in ('baseline', 'augmented')]
                    if not all(m.exists() for m in models):
                        raise ValueError('完整尺度实验需要先训练分类器。')
                    out = Path(RESULT_DIR) / 'scales' / Path(p).stem
                    annotation = _ann_path(Path(p).name)
                    result = run_scale_experiment(p, out, models, annotation if os.path.exists(annotation) else None)
                    report = f'scales/{Path(p).stem}/report.md'
                else:
                    from utils.video import analyze_video
                    name = Path(data.get('video', '')).name
                    video = Path(BASE) / 'dataset' / 'videos' / name
                    if not video.is_file():
                        raise ValueError('请选择视频。')
                    model = classification_model_path()
                    result = analyze_video(str(video), RESULT_DIR, stride=int(data.get('stride', 25)),
                        model_path=model,
                        image_dir=IMAGE_DIR, annotation_dir=RESULT_DIR)
                    report = f'video_{video.stem}/report.md'
                _jobs[job_id].update(status='done', result=result, report=report)
        except Exception as e:
            _jobs[job_id].update(status='failed', error=str(e))
    _executor.submit(work)
    return jsonify(id=job_id), 202


if __name__ == "__main__":
    app.run(host=os.environ.get("LAB2_HOST", "127.0.0.1"),
            port=int(os.environ.get("LAB2_PORT", "5000")), debug=False)
