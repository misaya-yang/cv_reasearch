# SAM 共享解码器：CPU 复现包

日期：2026-10-01。包含四方法实现、随机正确性测试、固定版本的最小官方源码和结果。无需权重或网络即可运行包内随机测试；不含预训练权重、完整图像编码器及真实图像资产。

## 结论先行

这里的 **exact 指实数算术下的图改写等价**，不保证浮点逐位一致、分割质量不变或 GPU 加速。

- 随机权重 SAM：原始因子诊断 10/10 通过（FP64、FP32 各 5）；独立 SAM2 扩展 5/5 通过
- SAM 执行基线：9 个固定输入 × 6 个变体，54/54 通过；投影因子候选：同样 9 个输入 × 2 种注意力实现，18/18 通过
- 随机 FP32 的最大 mask-logit 误差：执行基线 3.12924e-7；投影因子候选 4.61936e-7；阈值均为 5e-5
- 已训练 SAM 的同一真实图像、两个点提示：缓存、关联顺序和投影因子共 6 个变体的 mask-logit 误差都超过 5e-5，因此全部失败。阈值没有放宽
- 这 6 个变体在该样本的低分辨率二值掩码和 IoU argmax 上均未观察到变化；这不构成一般决策或质量保证
- 另对较早的原始 split-projection 因子诊断实现执行官方完整后处理：点提示批次有 4 个、框提示批次有 15 个二值像素不同。每批比较 8 张 1200×1800 掩码，共 17,280,000 像素。因此最终分辨率掩码并不完全相同
- 后处理的最低单掩码 IoU 为 0.9997528423（点）和 0.9999571533（框）；micro IoU 为 0.9999959750 和 0.9999942182。这里的 IoU 比较两个实现，不是对真实标注的质量评分
- 上述完整后处理检查不覆盖后来的 6 个执行基线变体。它们的真实数据检查限于低分辨率输出
- 未运行 GPU；未运行优化编译器。已有 compile/backend=eager 检查仅验证接口和完整图执行，不代表优化编译性能

可核对的聚合数值、原始结果哈希和实验范围见 `results_summary.json`；简表见 `RESULTS.md`。

## 包内四种方法

`sam_shared_decoder/execution_baselines/benchmark.py` 默认比较：

1. `official`：固定版本、未经修改的官方完整 MaskDecoder
2. `cached`：普通稠密图像状态，缓存首层图像 K/V/Q
3. `dense_assoc`：保留稠密图像状态，按张量尺寸选择投影与聚合的关联顺序；选择依据是运算量模型，不是实测延迟
4. `factor_projected`：投影因子 + 稠密伴随状态，后者计算真实 LayerNorm 统计量；保留所有 mask token、四个超网络、非线上采样和 IoU 头

更便宜的 implicit-factor attention 调度仅做了分析，**本包没有实现或测量它**。旧版 `validate_official.py` 是带中间诊断的原始候选，不能当作无诊断开销的性能实现。

## 依赖

使用你已配置好的 Python 环境。已验证环境版本列在 `requirements-tested.txt`：Python 3.12.14、PyTorch 2.14.1+cpu、NumPy 2.5.3；SAM2 的最小官方模块还在导入时需要 Pillow 12.3.0 和 tqdm 4.70.1。

此包不自动安装软件、不下载权重或源码，也不附带 Python 运行环境。依赖版本用于记录已验证配置，不表示其他版本已被验证。CUDA/低精度/TF32 选项的存在不表示其数值或速度已被验证。

## 运行

解压后进入含本 README 的目录。以下 `python` 指你已配置好依赖的解释器。

最小四方法 CPU 检查（所有输入均为随机权重和合成的已编码张量）：

```bash
python sam_shared_decoder/execution_baselines/benchmark.py --device cpu --dtype float32 --prompt-batch 3 --microbatch 2 --grid 4 --tokens 7 --warmup 1 --repetitions 1 --output smoke.json
```

先运行普通基线，再运行因子候选完整随机套件：

```bash
python sam_shared_decoder/execution_baselines/validate.py
python sam_shared_decoder/execution_baselines/validate_factor.py
```

原始官方模块逐阶段对照，以及独立 SAM2 扩展：

```bash
python sam_shared_decoder/official_cpu_validation/validate_official.py
python sam_shared_decoder/official_cpu_validation/sam2_extension/validate_sam2.py
```

两条官方模块测试命令可加 `--quick`，仅跑一个小 FP64 样本。完整 SAM 测试含 N4096，SAM2 测试也有一个 N4096 FP32 样本。执行基线使用 D256、H8、两层、MLP2048，测试所有四个 mask token；54 次对比包括 N4096 和不同 token 数。

正确性脚本会写回各自目录的结果 JSON；官方模块测试还会从本包的固定源码快照重建相同模块。因此请保留原 ZIP，或另解压一份用于重跑。`validate_factor.py` 读取同目录的 `validation_results.json` 来核对模型/输入种子，故不要删除该文件。

## 如何理解 benchmark

默认运行四种方法，在计时前按相同 dtype 和 microbatch 对照官方输出。测量边界是完整低分辨率解码器；不包括图像编码、提示编码、完整分辨率后处理或跨 microbatch 输出拷贝/拼接。相同微批调度用于所有方法。

- 一张图像和完全相同的 dense prompt 可以共享缓存；不同 dense prompt 不能复用该缓存
- 固定 PE 缓存、每图缓存构建、冷图解码和热缓存解码分开报告；“冷”不表示清空硬件缓存
- 权重、图像、dense prompt、PE、device、dtype 或相关执行计划改变时，需要重建缓存
- SDPA 表示调用 PyTorch API，不证明实际使用 FlashAttention 或某个融合后端
- `--mode compile --compiler-backend eager` 是诊断接口检查；不能作为优化编译的速度证据
- 包内已有的微型 CPU smoke 仅证明流程能运行，不能据此给方法排速度名次或外推 GPU 提速

## 证据边界与文件

四份详细随机结果 JSON 与当时执行的代码保持字节一致。`results_summary.json` 从这些记录及真实图像验证结果提取聚合值；真实图像的权重/输入不在包内，所以真实图像结果只可查阅，不能仅凭本包重跑。完整分辨率后处理使用官方 `Sam.postprocess_masks` 的 resize → crop → resize → threshold 路径，没有据此增加训练质量结论。

`package_provenance.json` 记录原样复制的文件与 SHA256。`SHA256SUMS.txt` 校验交付文件；测试重跑后会自然改变输出 JSON。官方源码精确固定到：

- SAM：`dca509fe793f601edb92606367a655c15ac00fdf`
- SAM2：`2b90b9f5ceec907a1c18123530e92e794ad901a4`

所需官方文件与源快照均附带，原始来源链接和内容哈希见 `sam_shared_decoder/source_audit/source_provenance.json`。上游许可证见 `THIRD_PARTY_NOTICES.md` 与相应 `LICENSE` 文件。
