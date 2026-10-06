# Pro M1：实际块接口的可执行准备

已实现：`src/ics/methods/pro_common_context.py` 和独立 `scripts/run_pro_common_context.py`。固定四臂ctx/plain/mean-unit/uniform，共享ρ=1、λ=16、20-NN互近邻图、τ=.07、每角色最多4真实参考探针、原始block20初值、block21–24续算、正式3×3局部及prefix键排除。两次双线性/阈值 renderer 输出1024及原尺寸完整mask。没有query GT、类别/fold参数、额外学习模型或网络下载。

适配器调用**实际模型的block.forward**并保留native前向传入的RoPE和其他args/kwargs；只在实际SDPA点换成原生Q的只读K/V。LN、QK norm、QKV bias、attention输出norm/投影、LayerScale、MLP及residual都由实际块执行。原始H20来自block21的输入pre-hook，绝不使用`get_intermediate_layers(...,norm=True)`。模型/块/attention的真实源码路径与SHA运行时写入receipt。非fused、非标准等宽Q/K/V、mask/dropout/causal、非NLC或多SDPA调用均显式拒绝，不猜实现。

根代理提供的资产审计上下文：服务器实际为timm1.0.30 `Eva`，源码`/root/demo4_cache/env/timm/models/eva.py`，24块/D1024/16头/CLS+4register，权重SHA `45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941`。这些是团队传来的运行时资产信息，不是本子代理在服务器核验的结果；本地timm和真实权重均缺失，因此**尚未完成真实Eva数值一致性**。运行时原生self审核未通过会在输出前失败。

本地有限核验：

```sh
python3 scripts/run_pro_common_context.py --self-check --out evidence/local/research_20261006/pro_context_preparation_01a1100b
```

`toy_check.json`记录：五块/16维/四头/36patch/2prefix的确定性含RoPE、QK norm、bias、LayerScale小模型；最后三块续算。无排除键self与native最大差4.76837158203125e-7；正式7位置和36位置attention分块差相同；memory不改写；损坏K/V导致审核拒绝。四臂图求解和完整renderer均通过；参考全前景与空前景分支通过。这不是24块真实DINO验证，不是分割收益或服务器耗时测量。

真实CPU入口（由根代理绑定当前授权的资源，不在此启动）：

```sh
python3 scripts/run_pro_common_context.py \
  --reference REF_RGB --reference-mask REF_BINARY_MASK --query QUERY_RGB \
  --weights /root/demo4_cache/models/dinov3-vitl16-timm \
  --host-packet HOST_SCORE_NPZ --host-receipt HOST_RECEIPT_JSON \
  --basis EXISTING_NATIVE_BASIS_PT --threads 4 --chunk-size 64 \
  --audit-only --out FRESH_REAL_NATIVE_AUDIT_DIR
```

审核通过后，去掉`--audit-only`运行固定四臂。新输出目录存完整field、1024/原尺寸mask、源/权重/host身份、原生self差及时间、RSS原始平台计量；本脚本不读取query GT或计分。各臂推断结束后另由评价入口封存、计分。

`HOST_SCORE_NPZ`仅打开`score`，形状64×64，必须是完整FoRIS连续证据链的二值化/CRF之前响应。本脚本没有实现FoRIS RGB producer；用外部已冻结、无query GT的host缓存组合，不将其生产成本计作0或宣称此脚本实现了整个FoRIS RGB链。Host receipt必须明确绑定：

- `reference_sha256`, `reference_mask_sha256`, `query_sha256`；
- `weights_sha256`, `config_sha256`, `host_packet_sha256`；
- `score_stage="complete_FoRIS_before_binarize_CRF"`, `query_gt_used=false`；
- `producer_recipe`, `producer_source_sha256`；
- `image_transform="PIL_RGB_bilinear1024_ImageNet_FP32"`，来自实际producer核验；
- `apply_native_projection`，若true还要求`basis_sha256`与已存在normalized-black原生basis一致。

参考mask必须显式0/1或0/255二值PNG，不能传类别号掩码。R/Q使用PIL双线性到1024、ImageNet归一化，mask最近邻到1024再area到64。模型FP32、autocast/TF32关闭；当前入口仅CPU。基底原矩阵保留，没有QR或列缩放。投影计算明确为`x-(x@U)@U.T`，在全部臂中一致；真实host历史数值parity需另检，不能自动等同已有FP16/APD缓存。旧241 q/r不足以提供H20/KV。

资源边界：当前适配器为正确性优先方案，各轨迹整图通过真实paired block，attention矩阵按查询位置分块；因此会重新计算未用的参考QKV/MLP和query K/V。它完整符合只读memory合同，但尚未实现选择性query投影优化。不得套用Pro的3.5图像等效承诺它的墙钟；真实CPU耗时未知。缓存还保存四层paired raw block inputs/调用args用于真实块重放，内存高于只存Q KV的128MiB。CPU32核/60GiB预算由根代理统一安排，本子代理不并行争用服务器。

下一步只有一个：根代理在实际timm Eva、真实绑定RGB/host资产上执行`--audit-only`；若缺资产或一致性失败，保留精确错误，不启动变体搜索。通过后才能讨论实际四臂成本/完整分割实验。当前方法效果未测，准备完成不增加已验证方法数。

## 新46466端点真实CPU审核：2026-10-06

收到本轮明确资源授权后，已在独立`code_pro_context_work_v1`复制中运行，未覆盖上传snapshot。4线程、8GiB RSS守卫、没有GPU/下载/query GT。输入为`bound600_v2/smoke4.json`首例`0_0_72`，原始COCO参考/查询RGB真实1024编码；没有用末层缓存替代原生状态。

`real_native_audit_v1.json`：实际timm Eva权重318键全部匹配；raw H20=[2,4101,1024]，Q K/V=[1,16,4101,64]。无prefix/局部排除审核模式的native self最大差**0.0**，memory未改写。真实配对编码47.177144秒、审核后缀8.861705秒、加载6.768471秒、总墙钟64.367770秒。Linux ru_maxrss=3055880KiB（约2.914GiB）；200ms采样RSS峰值2777894912bytes，采样值不是精确峰值替代。

随后在独立`runs/pro_context_predict4_v1`启动四臂前4例；新模型完整FP32编码取得H20/KV，宿主score和Π gate绑定既有packet/feature，按**缓存宿主＋新FP32 native图**版本报告，不宣称历史MEAN或完整图像到FoRIS生产链数值复现。完成mask封存后由独立评价阶段计分；当前无质量结论。运行PID和路径见`real_predict4_launch_v1.json`，它们仅为本次新作业记录，不是可继承队列或资源授权。

首例完整四臂已封存，见`real_predict_first_v1.json`：pair编码48.045896秒、ctx九轨迹续算71.001650秒、uniform九轨迹续算61.488021秒；包括原生self审核、四臂图求解及渲染保存的首例墙钟192.943620秒。原生self仍为0误差，四臂CG相对残差均<1e-7，采样RSS峰值2911813632bytes。四臂完整工作mask前景像素plain155378、mean-unit223432、ctx144814、uniform115754；这些只是预测面积，未读GT，不是效果/正确性证据。首4例均复用已暴露样本，不能作独立确认。

**首4例已经全部完成并退出。** `real_predict4_v1.json`及`real_predict4_seal_v1.json`记录14个输出文件的hash，封存前后未读取query GT。远端完整输出保留在`/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/pro_context_predict4_v1`，没有向本机传raw H20/QKV。

| key | 完整四臂首例墙钟秒 | 原生pair秒 | ctx后缀秒 | uniform后缀秒 | self maxabs |
|---|---:|---:|---:|---:|---:|
| 0_0_72 | 192.943620 | 48.045896 | 71.001650 | 61.488021 | 0 |
| 1_0_73 | 195.470383 | 46.698979 | 72.359604 | 64.931434 | 0 |
| 2_0_74 | 202.913031 | 50.048363 | 74.934955 | 65.544712 | 0 |
| 3_0_75 | 201.876019 | 48.935539 | 77.497929 | 63.509041 | 0 |

含模型加载的总墙钟801.550024秒（13.36分钟），4线程，Linux ru_maxrss3201520KiB（约3.05GiB），采样RSS峰值3243982848bytes；8GiB守卫未触发。四例都用9轨迹、同固定参数和实际raw状态，memory只读审核通过；没有因成本/预测面积调参数或开新变体。质量仍未评分，不能把native数值一致性认作语义收益。下一步由根代理独立计分或安排下一cohort共享capture，不自动扩展到600。

共享capture的exact字段/producer/内存核对见`capture_sharing_review.md`：当前M1/M5只有部分对象互补，下一cohort可做一次联合paired forward；核心张量未别名672.8203MiB、严格只读别名576.7031MiB，另计模型/投影/RoPE/临时状态。M4可接raw未投影原Q finalLN，但batch2与当前M4单Qbatch1数值核对尚未运行，不承诺bitwise或省掉八/十六干预编码。

## 首4独立QUALITY评分完成：当前结果为负的活动检查

根代理新授权后，独立1线程`score_real4.py`先验证全部14文件SHA、全部field/mask形状、实际JPEG原HW、两次renderer逐位对应、同evaluation manifest/control hashes，写`score4_v1/pre_GT_audit.json`后才打开packet.truth。没有重新编码，没有重做预测或更改参数。复用`ics.experiment.summarize`的class-summed I/U、照片连通簇2000次RandomState(0)配对bootstrap。仅1024同packet GT/同像素口径，**没有原尺寸GT评分**。

| 行 | 首4 class-summed mIoU |
|---|---:|
| ctx | 25.112759 |
| plain.control | 32.410145 |
| mean-unit.control | 24.975686 |
| uniform.control | 35.135329 |
| full native | 36.316536 |
| stored_mean.control | 35.684576 |
| rcg.control | 35.473585 |
| fine16.control | 34.911639 |
| fine64.control | 35.907872 |

ctx−plain为−7.297386，描述性配对区间[−12.382629,−2.212143]；ctx−uniform为−10.022571，[−14.476247,−5.568894]，逐例0胜/4负；ctx−native为−11.203777，[−16.304637,−6.102918]，0胜/3负/1平。ctx−mean-unit为+0.137073，[−7.443417,7.717563]，1胜/3负。四例/四类别/四照片簇且样本已暴露，这些区间仅描述活动检查，不作总体显著性或整类方法不可能性结论。

相对native，ctx新增TP16348、删除FP131898、误删TP52493、新增FP283309；相对plain，ctx新增TP9017、删除FP157787、误删TP29873、新增FP255846。这显示当前四例中错误新增和真实目标损伤未被纠错抵消，不以native数值一致性冒充语义改善。完整逐例I/U、原始像素账本和每基线配对增删在`score4_v1/episode_metrics.json`与`pairwise_edits.json`；每臂card与report已回传。本次评分约0.926秒、无encoder前向。

当前首4观察不支持直接开600，也不能预测固定24的结果。是否继续有限24由根代理依据已测质量和每例约193–203秒四臂成本决定；本子任务不自动派生机制/常数/层位变体。远端评分目录`runs/pro_context_score4_v1`，本地完整报告`score4_v1/report.json`和`report.md`。
