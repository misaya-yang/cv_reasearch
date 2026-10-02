# 两组 post-LayerNorm 投影合并

本模块优化现有 `execution_baselines/factor_candidate.py`，复用原
`FactorState.update` 和稠密 LayerNorm 辅助状态，保留完整四个 mask 和 IoU。
三个主实现文件未修改。CPU 正确性已验证，GPU 收益尚未测量。

## 等价式与调用数

对因子状态 `X=s*C+U@V`，同一状态的独立投影
`P_j=s*(C@W_j.T)+U@(V@W_j.T)+PE_j+b_j` 可按输出维拼接：

```
W = cat([W_1, W_2, W_3], dim=0)
P = s*(C@W.T) + U@(V@W.T) + cat(PE_j) + cat(b_j)
P_1, P_2, P_3 = P.split(widths, dim=-1)
```

| 状态 | 输出分块 | 合并权重 | 合并基底 | 提示相关展开 |
|---|---|---|---|---|
| 第一次图像 LN 后 | 次层 read K、read V、write Q | `[384,256]` | `[1,N,384]` | `[B,N,r1]@[B,r1,384]` |
| 第二次图像 LN 后 | final K、final V、第一上采样 | `[512,256]` | `[1,N,512]` | `[B,N,r2]@[B,r2,512]` |

K/Q 加对应 PE 投影，V/上采样的 PE 分块为零。bias 各加一次。
因子路线保留“PE 后 bias”，普通缓存保留“bias 后 PE”的原运算顺序。
首层 `ImageCache` 保留原生 `(image+dense+PE)` 联合 K/Q 投影，V 没有 PE。

无重叠 `ConvTranspose2d(kernel=2,stride=2)` 权重 `[256,64,2,2]` 排列成
`[256,256]` phase 线性映射，拼入第二组。输出按
`(h,w,phase_y,phase_x,channel)` 还原，后续 LayerNorm2d、GELU、第二上采样、
hypernetwork 和 IoU head 原样执行。

六次 `U@(V@W)` 展开变两次，内部 `V@W` 也从六次变两次；乘加数不变。
独立 CPU profiler 在完整 decoder 中确认了六次/两次 `aten::bmm`，形状见
结果 JSON。其他 GEMM、attention、LN 和上采样未计入“6→2”；计数不是测速。

## 公平基线和缓存成本

普通缓存使用同类两组合并。强基线依据实际 shape plan，对 projected 路线
执行 eligible 的合并（包括 sparse-output write）；associated 路线继续调用
原强基线，不补回已消除的图像投影，上采样保持原生。N=4096、T=7/9 的 auto
强基线两组为空，输出与原实现逐位相同。混合 read qk/value plan 尚不支持，
会显式报错；现有 auto/forced planner 对这两项选择一致。

`MergeCache.build` 直接计算两块合并基底，不先建立旧 `FactorCache` 或重复
独立 K/V/Q/conv 基底。plain/strong 不计算因子基底。首层 `ImageCache` 仍保留。
合并权重、完整宽度 PE、bias、因子基底都是实际持久张量，须计入缓存耗时和
显存。拼接 PE 会增加缓存空间；不能隐藏这项代价。

推理路径不重建权重/PE，没有误差统计、计时器或 profiler。可选 `trace` 仅供
验证，正式计时保持 `None`。模型、图像、dense prompt、PE、dtype/device 改变
后重建缓存；strong 缓存还绑定实际提示 T 和 contraction plan。

## CPU 验证

```bash
cd /Users/yang/projects/CVPR2027/demo_lists/demo1_sam
python3 research/projection_merge/validate_merge.py
```

Python 3.9.6、PyTorch 2.8.0、CPU、两线程；两个 TF32 开关均关闭。
先小型 FP64/FP32，再原全部 9 个 FP32 fixture，覆盖 N=64/4096、T=5/7/9/17、
不同 seeds、共享常量/空间 dense prompt。每个 fixture 测 factor、cached、
strong auto、forced projected+sparse、associated+sparse，各测 explicit/SDPA。
检查官方四-mask/IoU、LN 状态、公共单/多 mask 切片、原实现、microbatch 1/2。

110 variant-cases 全通过，门槛仍为 `5e-5`。FP32 最坏 mask 误差 `5.0664e-7`，
IoU `1.9372e-7`，所有状态/输出/分批对照最坏 `3.3379e-6`；小型 FP64 最坏
`4.4409e-15`。小型分块代数专项 FP64/FP32 最大差异均为零。

**保留了一次二值阈值变化**：seed=1、N=4096、T=9、cached+SDPA 的 524288
个 mask logits 中有 1 个发生零阈值符号变化，参考最小绝对值 `8.3966e-8`。
logit 门槛通过不能称为所有二值输出完全相同。所有 IoU argmax 均一致。
完整结果为 `cpu_results.json`，stdout 为 `cpu_validation.log`；运行时长只记录
验证预算，不能当性能结论。

不覆盖预训练、真实图片、GPU、端到端 predictor、SAM2。已有预训练 FP32
约 `1e-3` 的差异和原门槛失败状态仍有效，本次随机 CPU 通过未撤销那条证据。

## GPU harness 接口

现有 baseline import 路径可用后，把本目录加入 `sys.path`：

```python
from baselines import PositionCache, shape_plan
from projection_merge import MergeCache, factor_merge_predict, cached_merge_predict

pc = PositionCache.build(model, pe)
fc = MergeCache.build(model, image, dense, pc, method="factor")
factor_call = lambda prompts: factor_merge_predict(model, fc, prompts, backend)

pc = PositionCache.build(model, pe)
cc = MergeCache.build(model, image, dense, pc, method="cached")
cached_call = lambda prompts: cached_merge_predict(model, cc, prompts, backend)

plans = shape_plan(model, image.shape[-2]*image.shape[-1], sparse.shape[1]+5,
                   method="dense_assoc")
pc = PositionCache.build(model, pe, plans)
sc = MergeCache.build(model, image, dense, pc, method="dense_assoc", plans=plans)
strong_call = lambda prompts: cached_merge_predict(model, sc, prompts, backend)
```

分别计时 PositionCache/合并缓存建立、首次完整解码、复用解码；报告峰值显存。
compile 应编译完整 callable 并使用真正优化 backend；所有比较保持相同
precision、TF32、batch、完整输出范围。SDPA 不自动等于 FlashAttention。
