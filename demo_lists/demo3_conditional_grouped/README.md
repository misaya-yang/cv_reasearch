# 条件正交分组解码：PixNerd T2I1024

实验负责人：本任务。独立于demo1 SAM及demo2 metric decoding，不修改它们的实验代码。

目标是检验：相同紧凑语义与原始像素信息下，内容决定的可逆正交组织，能否优于较大固定组、原始邻域和廉价动态滤波。首轮是预训练模型的配对velocity MSE/成本试验，不宣称SOTA。

## 已落实

- 作者方案原文保留在 `review_plan.md`。
- 官方代码固定 `6da060bd4b11a4a0ac2443a31b79701090e23c23`，不改源码。
- 主模型 `MCG-NJU/PixNerd-XXL-P16-T2I` 固定revision `9fa8836b553976b88a3b27649ed1628499e02bf9`，官方SHA及bytes在 `assets/manifests/download_public.json`。
- 官方conditioner是 `Qwen/Qwen3-1.7B`，revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`。代码MIT，两个模型卡Apache-2.0。
- 1024图片 / 512语义读取 / patch16语义stem / patch32像素头：1024语义tokens，P1024。默认插值nearest。T2I forward只有 `(x,t,y)`；没有C2I的s/mask参数。
- α=t、σ=1−t、velocity目标x−noise；时间采样sigmoid-normal后timeshift4。各分支共同省略REPA，不下载DINO。

## 方法与强对照

`cgd/model.py`保留官方语义计算及旧field。显式接入分析/合成，无调试hooks。

- original：仅同范围旧field继续训练。
- conditional：h/t融合状态生成全部butterfly角度，group2。
- fixed：只用相同时间嵌入，角度头同参数数，不读取图像语义；同近恒等初始化。
- group32：固定32系数读取，自己的RGB始终位于首个槽位。
- neighborhood：同patch内6×6原始邻域、replicate边界、原RGB第一槽，新增投影零初始化。
- dynamic：h生成共享RGB的6×6动态邻域特征，通过独立投影加到field输入，保留原始RGB直接路径，再接group2 field。当前是低成本动态滤波对照，尚未证明实测成本与候选匹配，也不冒充原论文DDF复现。
- full_stem：旧patch16权重沿nearest相位精确散布进patch32，新增列零初始化。新增列梯度保留，冻结Transformer仍参与反传；encoder checkpointing减少峰值。

所有分支训练旧像素embedder、NerfBlocks与final head；其他旧权重冻结。角度在RGB间共享，合成使用同角度逆序反号。位置编码标记变换后的系数。未新增GMM/能量模型/损失。

## 执行与资源

```bash
# 本机CPU随机小模块：正交、逆、gradcheck、全分支恒等恢复、stem梯度
assets/runtime/cpu/bin/python tests/check_correctness.py
# 服务器资源实测（预训练权重+文本缓存完成后）
PYTHONPATH=runtime/python_packages:assets/source/PixNerd:. /root/miniconda3/bin/python tools/pilot.py --mode conditional --phase resource --out results/resource_v1/conditional
# 队列：下载完成后文本缓存 -> 全分支资源检查 -> 匹配128步短训练 -> 配对汇总
bash tools/run_queue.sh --train
```

FP32 / TF32关闭 / batch1首测。完整forward/backward/AdamW计时，首次和后续更新分开。共享GPU下的耗时有资源竞争，不能作为独占吞吐结论。最多并行两个本项目分支，并依据实测峰值与当前剩余显存收缩；其他任务可能另占GPU。

不带--train仅完成资源检查；完整恢复点空间已通过用户授权的临时分块清理落实。128步仅初次受控额度。依据训练曲线、留出分层误差及实际成本决定延长/改法，不能把未收敛判成统计机制失败。保留分支日志、失败、可恢复权重与optimizer。首轮不自动展开大训练。

## 数据与解释

复用服务器COCO官方train/val ZIP和captions，不重复下载。固定256 train/64 val图像、固定caption，图像级分离；排除可识别的demo1小集ID。沿官方Resize(1024)+CenterCrop+[-1,1]处理。COCO大多原始分辨率低于1024：这是上采样受控pilot，不证明原生1024细节质量；后续须增加合适的高分辨率caption资产。

固定caption的Qwen表示可以缓存；噪声相关h每步重算。完整stem分支不detach。评价在t=.2/.5/.8上保留图像级配对数据，以图像bootstrap，单训练seed的区间不能涵盖训练随机性。小样本不测FID。

`results/cpu_correctness.json`的PASS仅属于随机小模块；预训练数值检查由GPU resource脚本另行报告。不会因为下载/CPU检查完成而宣布GPU可训练或质量通过。
