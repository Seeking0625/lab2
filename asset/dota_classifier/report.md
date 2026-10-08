# 现成遥感标签分类器训练

来源：DOTA v1.0 labelTxt.zip; difficult=0; rotated rectangles are not iSAID pixel masks
独立原图：训练721张，验证192张；两侧无重叠。演示图及其裁块来源全部留在验证侧。
原始训练目标10771个，增强后21542行；验证3590个。
验证OA=0.9123，Kappa=0.9018；手工特征基线OA=0.8518。
Negative crops avoid all provided target boxes. DOTA can omit objects; background labels are weak supervision.
Scores are uncalibrated class votes; score<.35 or top-two gap<.04 leaves unlabeled and offers suggestions.
验证基于参考目标框裁块，不等于SAM区域分类精度或视频精度。建筑/农田/道路不在DOTA原始15类中，保留为人工自定义类别。

| 类别 | 训练数 | 验证数 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| airplane 飞机 | 1000 | 311 | 0.968 | 0.987 | 0.978 |
| baseball-diamond 棒球场 | 155 | 40 | 0.974 | 0.950 | 0.962 |
| basketball-court 篮球场 | 256 | 55 | 0.932 | 0.745 | 0.828 |
| bridge 桥梁 | 181 | 37 | 0.561 | 0.622 | 0.590 |
| ground-track-field 田径场 | 142 | 17 | 0.591 | 0.765 | 0.667 |
| harbor 港口 | 1000 | 300 | 0.879 | 0.823 | 0.850 |
| helicopter 直升机 | 31 | 1 | 0.000 | 0.000 | 0.000 |
| large-vehicle 大型车辆 | 1000 | 547 | 0.898 | 0.898 | 0.898 |
| other 背景/其他 | 2163 | 576 | 0.978 | 1.000 | 0.989 |
| roundabout 环岛 | 140 | 57 | 1.000 | 0.526 | 0.690 |
| ship 船舶 | 1000 | 300 | 0.797 | 0.813 | 0.805 |
| small-vehicle 小型车辆 | 1000 | 412 | 0.845 | 0.850 | 0.847 |
| soccer-ball-field 足球场 | 118 | 27 | 0.826 | 0.704 | 0.760 |
| storage-tank 储油罐 | 1000 | 300 | 0.920 | 0.997 | 0.957 |
| swimming-pool 泳池 | 585 | 301 | 0.986 | 0.970 | 0.978 |
| tennis-court 网球场 | 1000 | 309 | 0.965 | 0.987 | 0.976 |