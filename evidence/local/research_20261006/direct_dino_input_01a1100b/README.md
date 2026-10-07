# 当前直接 DINOv3 输入与处理

用户最新要求用 DINOv3 特征处理参考/查询。当前入口只接受冻结 DINOv3 的 patch 表征、已知完整参考掩码的面积权重及图像几何；没有 FoRIS 分数、Part1 位置去偏、MEAN 基场或 native 掩码输入。服务器按用户关机状态保持关闭，没有远端连接或开机动作。

## 已实际完成

本地四例真实 RGB128 whole-image DINO 输出已从此前封存的 `tokens.npz` 提取为独立输入包。只读取 `reference_rgb/query_rgb` 两个表征、参考 mask 面积权重和几何；不读取原工作特征、gray、旧 margin、base 或 query GT。原生 encoder 输出为 FP32 final-LN，原保存过程随后做单位归一化并转 FP64；**没有 Part1**，不是未经任何归一化的 residual state。

默认处理是一个直接 prototype 基线：对参考表征按完整 mask 的 FG/BG 面积加权取单位均值，查询特征做与物理视图一致的映射，判别 `cos(q,mu_FG)-cos(q,mu_BG)>0`。输出完整 1024 与原尺寸掩码。它仅用于落实正确的 DINO 输入/完整输出路线，不计为一个新的科学方法。

四例已运行并封存；算子核对摘要见[checks.json](checks.json)。输入manifest、feature packs、seal和逐例掩码留在本地工作区，未纳入本次提交：manifest含机器绝对路径，feature packs是二进制输入。解析程序不接受 GT 输入。查询目标移动的独立合成检查、正交特征基不变性、half-pixel 插值解析 ramp 均通过；四例 direct-RGB margin 与原同 RGB 的独立算子核对误差最大 `8.9e-16`。这个核对只用 RGB margin 验证数学运算，没有在预测时使用它。

| 真实已有表征 | 原尺寸完整输出 |
|---|---|
| 0_0_72，DINO128，64 tokens | 640×640 |
| 1_0_73，DINO128，64 tokens | 375×500 |
| 2_0_74，DINO128，64 tokens | 640×480 |
| 3_0_75，DINO128，64 tokens | 375×500 |

**这不是 1024 DINO 实验或质量提升结果。** 本次已检查的项目资产中未找到上述样本的完整查询 GT 或 RGB1024 直接 DINO 特征包；所用本地 Python 环境没有 Torch，1024 checkpoint/runtime 未就绪；未计算新输出 mIoU，不能挪用原四例 bounded-MEAN 报告评分。128 输入的 8×8 patch 表征尤其不能代替目标的 1024 输入、64×64 native patch 表征。

## 目标 1024 的独立入口

[run_direct_dino_features.py](../../../../scripts/run_direct_dino_features.py) 提供 `extract` 与 `infer`：默认只做两张 whole-R/Q RGB1024 的冻结 CPU 前向，直接取 FP32 final-LN 的 4096 个 patch；MR nearest1024 后按16×16面积标注 reference tokens。checkpoint/config 在前后逐 SHA 核对。extract 不调用 Part1，也不接受 FoRIS 输出。

首四例 v3 seal 绑定的 runner SHA256 为 6deb426d32a65359b07035890d99a450a04403d378ba6a41247d7ad1f35cea2b，当前 runner 为 cce023cc558451ae307bd047338994c646f32bb4ef9c223aebce86e0e038e733；方法模块 SHA256 两者均为 15adc1086a135b4a0f0a92bb54661bce7bca3ae411ffe6d3d3f7cdd6967c469a。二者 runner 差异只在 extract 分支增加 architecture 校验与记录；已封存的 infer 分支相同。当前 extract 尚未运行。

`infer` 只读取该明确绑定的 DINO-only pack 和 MR 权重，独立产生 signed margin 与完整二级掩码。NumPy decoder 为 half-pixel bilinear、border/no-antialias、strict `>.5`；已核算子数学，未声称与所有 Torch 实现 bitwise 相同。没有阈值选择、GT 调参、模型下载、服务器开启或 GPU 行为。

1024 `extract` 分支尚未执行；真实权重/runtime与数据可用后仍必须检查实际 forward/source/shape，完成完整评分才能谈方法效果。当前只保留这一个直接 DINO 处理入口，旧 FoRIS/MEAN 修补队列不再是当前待办。
