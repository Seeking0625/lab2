# 基于 SAM 的遥感图像半自动标注与分析工具

AI 综合实践·实验四。以 Segment Anything Model（SAM）为核心，对 iSAID 遥感图像完成
**自动分割 → 人工修正 → 自动报告 → 视频扩展** 的完整半自动标注流水线。

## 功能总览（对应实验四步骤）

| 步骤 | 内容 | 代码入口 |
|---|---|---|
| 1. SAM 自动分割实例 | 1024px 分块 + 64px 重叠 + bbox IoU 去重，输出多边形 JSON | `utils/sam_segment.py` |
| 2. 标注修正 Web UI | Canvas 交互：改类 / 合并 / 点选再分割 / 框选新增 | `app.py` + `static/` |
| 3. 自动报告 | 类别统计、面积占比、区域类型规则推断、可选 LLM 润色 | `utils/report.py` |
| 4. 遥感视频分析 | 抽帧 → 复用流水线 → 跨帧聚合 → 报告 + 分割叠加视频 | `utils/video.py` |

## 目录结构

```
lab2/
├── app.py               # Flask 后端（标注 UI + 全部 API）
├── classes.json         # 类别表（默认 iSAID 15 类，可在 UI 中增删）
├── asset/               # SAM 权重（不入库，见「数据准备」）
├── dataset/
│   ├── images/          # iSAID 训练图像（不入库）
│   └── videos/          # 测试视频（含合成的航拍运动视频 synth.mp4）
├── static/
│   ├── index.html       # 标注 UI 页面
│   └── js/main.js       # Canvas 交互逻辑
├── utils/
│   ├── sam_segment.py   # 步骤1：分块自动分割（含可视化）
│   ├── report.py        # 步骤3：统计 + 区域推断 + Markdown 报告
│   └── video.py         # 步骤4：视频合成/抽帧/分析
├── results/             # 分割 JSON、可视化、报告、视频分析输出
└── train/               # iSAID 原始压缩包（不入库）
```

## 数据准备

仓库不含大文件，运行前需自行放置：

1. **SAM 权重** `sam_vit_b_01ec64.pth`（375MB）
   下载：<https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth>
   放到 `asset/sam_vit_b_01ec64.pth`
2. **iSAID 图像**：将训练图像（PNG）放入 `dataset/images/`。
   没有数据也可以直接用 `utils/video.py` 从任意一张遥感大图合成测试视频。

## 环境安装

Python 3.10+，GPU 可选（CPU 也能跑，只是慢）：

```bash
pip install torch torchvision segment-anything==1.0 opencv-python flask matplotlib pycocotools
```

## 快速开始

```bash
# 1) 对单张图自动分割（结果 -> results/P0000.json，可视化 -> results/vis/）
python -m utils.sam_segment --image dataset/images/P0000.png --vis

# 2) 批量分割
python -m utils.sam_segment --dir dataset/images --max-images 5

# 3) 启动标注修正 UI（浏览器打开 http://127.0.0.1:5000）
python app.py

# 4) 生成报告：UI 中点击「生成报告」按钮，
#    或命令行调用 utils/report.py 的 generate_report()

# 5) 视频分析：先合成测试视频，再分析
python -m utils.video --synth-from dataset/images/P0000.png --synth-out dataset/videos/synth.mp4
python -m utils.video --video dataset/videos/synth.mp4 --stride 25
# 输出在 results/video_<名称>/：frames/*.json、overlays/*.jpg、annotated.mp4、report.md
# 也可直接传入任意真实航拍视频文件
```

## 标注 UI 操作说明

顶部按钮切换模式，左栏选图像和类别，右栏实例列表点击可定位：

| 操作 | 方式 |
|---|---|
| 选择 / 改类 | 模式1 点击实例选中，数字键 `1-9` 或右栏改类 |
| 合并实例（过分割） | 模式2 依次点击多个同类实例，`Enter` 合并 |
| 点选再分割（欠分割） | 模式3 点击目标内部，SAM 给 3 个候选，数字键 `1-3` 采纳 |
| 框选新增实例 | 模式4 拖出矩形框（可再配合中心点），回车确认 |
| 平移 / 缩放 | 滚轮缩放；中键或「空格+左键」拖拽 |
| 撤销 / 保存 | `Ctrl+Z`（30 步）；`Ctrl+S` 保存到 `results/*.json` |

## 自动报告说明

- 统计各类别实例数与面积占比（未标注不计入）；
- 内置 7 类区域规则（机场/港口/油库/体育场馆/农业/城镇/交通枢纽）按类别数量与面积评分推断地貌；
- 实例质心离散度描述空间分布；
- 可选 LLM 润色：设置环境变量 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`（OpenAI 兼容接口），未配置则用纯规则模板。

## 视频分析说明

按 `stride` 抽关键帧 → 每帧复用步骤1 分割 → 逐帧 JSON 与报告 → 跨帧聚合：
全片区域类型投票、类别出现帧数与累计实例数、前后半段目标数量趋势、
首/中/末帧地貌描述，并产出彩色分割叠加视频 `annotated.mp4`。

## 实测性能（RTX 4060 Laptop 8GB, vit_b）

| 场景 | 规模 | 耗时 | 显存峰值 |
|---|---|---|---|
| P0002（2557×2086） | 313 实例 | 24s | 1.6GB |
| P0000（3875×5502） | 613 实例 | 48s | 1.78GB |
| 合成视频 12s×25fps | 12 关键帧 | 47s | - |
