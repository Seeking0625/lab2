"""实验四 步骤三：标注结果 -> 自动分析报告

输入标注 JSON（utils.sam_segment 输出格式），输出：
- 类别与数目统计、面积占比
- 区域类型规则推断（机场/港口/油库/城镇/农业区/体育场馆等）
- 地貌描述 Markdown（可选接入 OpenAI 兼容 LLM 润色，环境变量：
  LLM_BASE_URL / LLM_API_KEY / LLM_MODEL，缺省走纯规则模板）
"""
import json
import os
from collections import defaultdict

# 区域类型推断规则：类型名 -> (构成权重表, 命中所需最低分)
REGION_RULES = {
    "机场/航空枢纽": {"airplane 飞机": 5, "helicopter 直升机": 4},
    "港口/码头": {"harbor 港口": 5, "large-vehicle 大型车辆": 1},
    "油库/储运区": {"storage-tank 储油罐": 5},
    "体育场馆/学校": {"ground_track_field": 4, "soccer-ball-field 足球场": 4,
                      "basketball-court 篮球场": 3, "tennis-court 网球场": 3,
                      "swimming-pool 泳池": 2},
    "农业种植区": {"farmland 农田": 4},
    "城镇居民区": {"building 建筑": 3, "small-vehicle 小型车辆": 2,
                    "road 道路": 2},
    "交通枢纽/立交": {"roundabout 环岛": 4, "bridge 桥梁": 3,
                       "large-vehicle 大型车辆": 1},
}


def _region_scores(cls_count: dict, cls_area: dict) -> dict:
    """按类别数量与面积占比给各候选区域类型打分。"""
    total_area = sum(cls_area.values()) or 1
    scores = {}
    for region, weights in REGION_RULES.items():
        s = 0.0
        for cls, w in weights.items():
            s += w * cls_count.get(cls, 0)
            s += w * 8 * (cls_area.get(cls, 0) / total_area)  # 面积权重
        if s > 0:
            scores[region] = round(s, 1)
    return dict(sorted(scores.items(), key=lambda kv: -kv[1]))


def _describe(ann: dict, cls_count: dict, cls_area: dict, unlabeled: int,
              spread: float) -> str:
    """规则模板生成地貌描述段落。"""
    total_area = sum(cls_area.values()) or 1
    ranked = sorted(cls_count.items(), key=lambda kv: -kv[1])
    area_ranked = sorted(cls_area.items(), key=lambda kv: -kv[1])

    parts = []
    n = len(ann.get("instances", []))
    parts.append(f"本影像共分割出 {n} 个实例，其中 {n - unlabeled} 个已完成类别标注"
                 f"（未标注 {unlabeled} 个）。")

    if ranked:
        top3 = "、".join(f"{k}({v}个)" for k, v in ranked[:3])
        parts.append(f"数量最多的地物为：{top3}。")
    if area_ranked:
        k, v = area_ranked[0]
        parts.append(f"占据面积最大的类别是{k}，约占已标注面积的 {v / total_area:.0%}。")

    if unlabeled > n * 0.5:
        parts.append("注意：超过半数实例尚未标注，区域判断置信度有限，建议先在标注"
                     "界面完成主要类别确认。")

    density = "空间分布集中" if spread < 0.25 else \
              "空间分布较分散" if spread < 0.45 else "空间分布离散"
    parts.append(f"实例质心归一化标准差为 {spread:.2f}，呈{density}态势。")
    return "".join(parts)


def _centroid_spread(instances: list) -> float:
    """实例质心的归一化平均标准差（0 集中 ~ 1 均匀铺满）。"""
    pts = []
    for it in instances:
        x, y, w, h = it["bbox"]
        pts.append((x + w / 2, y + h / 2))
    if len(pts) < 2:
        return 0.0
    W, H = max(p[0] for p in pts), max(p[1] for p in pts)
    W, H = max(W, 1), max(H, 1)
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sx = (sum((p[0] - mx) ** 2 for p in pts) / len(pts)) ** 0.5 / W
    sy = (sum((p[1] - my) ** 2 for p in pts) / len(pts)) ** 0.5 / H
    return (sx + sy) / 2


def _llm_polish(prompt: str) -> str | None:
    """可选：OpenAI 兼容接口润色。未配置环境变量时返回 None。"""
    base = os.environ.get("LLM_BASE_URL")
    key = os.environ.get("LLM_API_KEY")
    if not (base and key):
        return None
    try:
        import urllib.request
        body = json.dumps({
            "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            base.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[report] LLM 调用失败，使用规则模板: {e}")
        return None


def generate_report(ann: dict, classes: list | None = None) -> dict:
    """标注 JSON -> 报告数据（统计 + 区域类型 + Markdown 文本）。"""
    cls_count = defaultdict(int)
    cls_area = defaultdict(int)
    labeled_area = 0
    for it in ann.get("instances", []):
        cls_count[it["class"]] += 1
        cls_area[it["class"]] += it["area"]
        if it["class"] != "unlabeled":
            labeled_area += it["area"]
    unlabeled = cls_count.pop("unlabeled", 0)
    cls_area.pop("unlabeled", None)  # 面积统计仅保留已标注类别

    img_area = ann.get("width", 0) * ann.get("height", 0) or 1
    spread = _centroid_spread(ann.get("instances", []))
    desc = _describe(ann, cls_count, cls_area, unlabeled, spread)
    scores = _region_scores(cls_count, cls_area)
    region = next(iter(scores), None) if scores else None

    # LLM 可选润色（把规则结论交给它扩展成自然段落）
    llm_text = None
    if region:
        prompt = (
            "你是遥感影像分析专家。基于以下统计写出 150 字以内的地貌与区域类型"
            f"分析段落：统计={dict(cls_count)}；推断区域类型={region}；"
            f"评分={scores}。直接输出结论性描述。")
        llm_text = _llm_polish(prompt)

    # Markdown 报告
    lines = [
        f"# 遥感影像分割分析报告",
        "",
        f"- 影像：{ann.get('image', 'unknown')}"
        f"（{ann.get('width', '?')} x {ann.get('height', '?')} px）",
        f"- 实例总数：{len(ann.get('instances', []))}（已标注 "
        f"{len(ann.get('instances', [])) - unlabeled}，未标注 {unlabeled}）",
        f"- 已标注面积占全图：{labeled_area / img_area:.1%}",
        "",
        "## 类别统计",
        "",
        "| 类别 | 数目 | 面积(px) | 面积占比(已标注) |",
        "|---|---|---|---|",
    ]
    for cls in sorted(cls_count, key=lambda c: -cls_area[c]):
        lines.append(f"| {cls} | {cls_count[cls]} | {cls_area[cls]:,} | "
                     f"{cls_area[cls] / (labeled_area or 1):.1%} |")
    if not cls_count:
        lines.append("| (暂无已标注类别) | - | - | - |")
    lines += ["", "## 区域类型推断", ""]
    if scores:
        for r, s in list(scores.items())[:3]:
            mark = " **<- 最可能**" if r == region else ""
            lines.append(f"- {r}（评分 {s}）{mark}")
    else:
        lines.append("- 标注类别不足，无法推断区域类型。")
    lines += ["", "## 地貌与区域分析", ""]
    lines.append(llm_text or desc)
    if llm_text:
        lines += ["", f"（规则模板参考：{desc}）"]

    return {
        "image": ann.get("image"),
        "total": len(ann.get("instances", [])),
        "labeled": len(ann.get("instances", [])) - unlabeled,
        "cls_count": dict(cls_count),
        "cls_area": {k: round(v, 1) for k, v in cls_area.items()},
        "region_type": region,
        "region_scores": scores,
        "spread": round(spread, 3),
        "used_llm": llm_text is not None,
        "markdown": "\n".join(lines),
    }


if __name__ == "__main__":
    import sys
    ann = json.load(open(sys.argv[1], encoding="utf-8"))
    rep = generate_report(ann)
    print(rep["markdown"])
