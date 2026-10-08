# 实验二：SAM 遥感分割与自动分类修正

按实验要求 PDF 第6页的四个实验步骤实现：SAM实例分割、UI类别与边界修正、图片分析报告、真实航拍视频分析。无需从零人工标注训练集。

## 组员克隆后启动

本机已验证 Linux、Python3.10 与 CPU/CUDA；提供 Windows 的 Conda 配置及命令说明，Windows 尚未实测。建议使用：

```bash
git clone <仓库地址>
cd <仓库目录>
git switch feat/automatic-classification-portable
conda env create -f environment.yml
conda activate mLearning
```

也可用 Python3.10 的虚拟环境后运行 `python -m pip install -r requirements.txt`。
依赖固定为本机验证版本；共享的随机森林模型使用 scikit-learn1.7.2，避免随意升级后加载不兼容。
默认自动选择 CUDA 或 CPU；没有 NVIDIA GPU 也能启动和推理，CPU 分割会较慢。
需要特定 CUDA 安装时，先安装匹配的 PyTorch2.6.0 / torchvision0.21.0，再安装其余依赖。

### 推荐：导入组内运行资源包

Git 不包含大模型和原始图片。向本项目维护者领取 `lab2-runtime.zip`，包含 SAM权重、ResNet18权重、已训练分类器、4张练习图及其预测、真实演示视频。
将资源包放到任意位置，在仓库根目录运行（路径换成实际路径）：

```bash
python scripts/resources.py import /path/to/lab2-runtime.zip
python scripts/resources.py check
python app.py
```

Windows 示例：`python scripts/resources.py import "D:\Downloads\lab2-runtime.zip"`。
导入检查 SHA256 和文件大小，不覆盖内容不同的现有文件。只导入组内提供的模型资源包。
打开 http://127.0.0.1:5000 。启动目录不影响网页、模型与默认数据路径。
导入后即可分割、自动分类、修正和分析视频，组员无需重新训练。

### 不使用资源包

```bash
python scripts/resources.py download
```

该命令从官方地址下载 SAM vit_b 与 ImageNet ResNet18 权重，已存在时跳过；不会下载训练后的分类器或 DOTA 数据。
把自己的 PNG/JPG/TIF 图片放到 `dataset/images/`。
若需要本项目分类器，复制 `asset/dota_classifier/model.joblib`，或准备下一节的数据后训练。
只有 SAM 时仍能分割和人工改类；“自动分类”会提示模型缺失。没有图片时网页显示准备数据提示，不会启动失败。

### 全部 DOTA 图片及自定义路径

运行包仅含练习图，完整数据集需另行共享/下载。推荐目录：

```text
dataset/dota/
  images/part1.zip
  images/part3.zip
  labelTxt-v1.0/labelTxt.zip
```

放好后重启服务。也兼容仓库旁边的 `../train/` 目录。
本机两个压缩包共937张原图；组员实际能看到的数量取决于自己的数据包。
压缩包内原图按点击解压缓存，左侧“原始图片”或“全部图像”可浏览。

以下可选环境变量支持绝对路径或相对仓库根目录的路径：

| 变量 | 用途 |
|---|---|
| `LAB2_DATA_DIR` | DOTA根目录，包含images与labelTxt-v1.0 |
| `LAB2_IMAGE_DIR` | 已解压图片及裁块、关键帧目录 |
| `LAB2_RESULT_DIR` | 标注与报告目录 |
| `LAB2_SAM_CHECKPOINT` | SAM vit_b权重路径 |
| `LAB2_APPEARANCE_WEIGHTS` | ResNet18权重路径 |
| `LAB2_CLASSIFIER_MODEL` | 已训练分类器路径 |
| `LAB2_HOST` / `LAB2_PORT` | 服务监听地址/端口，默认127.0.0.1:5000 |

Linux/macOS：`export LAB2_DATA_DIR=/path/to/dota`；PowerShell：`$env:LAB2_DATA_DIR="D:\data\dota"`。

维护者生成组内资源包：`python scripts/resources.py export --out results/local_run/lab2-runtime.zip`。
资源包不包含完整937张原图，完整原始数据按目录另行共享。

## 使用

1. 选择图片，点击“分割并自动分类”。SAM产生区域，分类器随后预测类别；重新分割会提示替换并备份旧标注。
2. 核对预测，点选实例改类别、新建类别、Shift多选合并、点选重分割或框选新增。正负点修边支持目标内正点和背景负点。已有人工确认的类别不会被自动分类覆盖。
点选重分割的候选卡会注明“拆分为几块，保留剩余区域”或“修正边界，替换原实例”。选中前先核对预览；Esc取消。候选已预检，无变化或不相交的结果不会显示，细小残片和旧多边形外接框之外的边缘保留。

3. 保存后生成报告，查看各类数量及可能区域。报告区分机器预测、人工确认和未分类；区域结论是规则推断。
4. 选择真实 `VisDrone_0011.mp4` 运行视频分析。自动抽关键帧、分割分类、生成叠加视频与报告；关键帧也可在图片分组中修正后重新分析。

首页默认四张局部练习图。左侧分组切换到“原始图片”可浏览本地压缩包中的937张原图，“全部图像”包含原图、练习图与视频帧；点击原图时才解压缓存，未处理图片需点击“分割并自动分类”。原图、局部图、视频帧分别分组。筛选和掩码开关仅影响显示，不删除数据。可拖框裁成局部图。
置信分来自随机森林投票，未经概率校准；低于0.35或前两类差小于0.04的区域保留未分类并显示候选。SAM分割分数与分类置信分分开显示。

## 分类训练与验证

```bash
python -u -m utils.train_reference
```

默认读取配置数据目录下 `images/*.zip` 和 `labelTxt-v1.0/labelTxt.zip`。也可传 `--archives`、`--labels`、`--out` 明确路径。
这是 DOTA 的旋转目标框标签，并非 iSAID 的像素掩码。训练覆盖15个官方目标类别，另加入避开参考目标框的弱监督背景裁块。
每类最多1000个训练目标；按原图划分训练/验证，训练侧做90度旋转增强。演示图片及其裁块的原图全部预留在验证侧。
训练报告、每类指标、OA与Kappa见 [report.md](asset/dota_classifier/report.md) 和 `evaluation.json`。
验证基于参考框内目标裁块，其分数不等于SAM区域或真实视频的分类准确率。
建筑、农田、道路不在原始15类中，可作为自定义类别手动修正；背景训练标签也不意味着能准确识别所有未见类别。

旧的人工标签训练与尺度实验代码保留为可选工具，不作为四步实验的前置要求。
视频来源见 [SOURCES.md](dataset/videos/SOURCES.md)，检查记录见 [VALIDATION.md](VALIDATION.md)。
