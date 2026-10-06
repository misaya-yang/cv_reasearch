# Pro M5 可部署实现与真实绑定缺口

已实现 `src/ics/methods/pro_message_extrapolation.py` 和独立CPU入口 `scripts/run_pro_message_extrapolation.py`。toy合同核对通过；随后在实际冻结DINO原生接口上做过一次CPU审核，但第6个参考区域的block22 beta=1差异0.000244140625，完整预测未封存，未读取GT或评分。实际部分运行已超过60秒继续门槛。当前源码随后有改动，尚未重审；质量结果仍未测。详见[审核记录](actual_audit_v1.md)。

## 已固定的执行合同

- 相邻Ward取4/8/16/32完整query分区，覆盖4N，不按宿主得分过滤ROI；局部tie按有序最小leaf ID。参考完整coverage≥.5为FG，非空原mask但粗FG空时取首个最大coverage token；BG最多8分区。BG连通分量超过8时先完成组件内Ward，再按未归一化组件特征均值的平方距离合并，报告非空间合并次数。
- 真实H20之后21–24块；实际native QK固定，prefix CLS/register全属外侧；每层区域内V只计算实际LN1与V投影，外部V读native。按query chunk分组logsumexp/softmax，无分母下限，无`1-p_out`消减。β=.75正式，β=0 hard控制，β=1逐层audit。不得将干预分支的QK作为attention权重。
- 内侧质量增加与外侧削弱同时发生；输出projection、gamma/LayerScale、真实MLP和残差保留。支持标准block、Functional注入适配器及实际timm Eva no-qkv-bias/gamma布局。未知布局拒绝，不静默替换。
- 先final LN、固定原episode的线性Pi、区域pool，再unit。4倍差分、.5相对范数限幅；η<1不称β0恢复。参考FG/BG同样处理。四分区类别margin差平均，`base+tanh(delta/.07)`，不clamp、minmax或重求图。
- 完整extrapolate/endpoint/hard_channel/native_region/zero五行，另有response/mass objectness诊断；native_region是old margin，绝不是zero。统一bilinear64→1024、严格>.5，再bilinear二值work→原H/W、严格>.5。
- paper前报告另要求的无内侧重分配、真正重算QK masked-attention、crop/灰底与FG/BG交换控制尚未集成为完整比较行。本次没有把这些控制冒充已完成，亦未开启新增研究机制。

## 与实际Eva/M1的接口

`capture_native_pair(model, images, project, producer=...)`直接捕获实际完整前向的原生SDPA输入Q/K/V（已包含实际QKnorm/RoPE），真实H20与四层真实残差输出。不重建RoPE频率，不依赖旧末层缓存。调用共享`pro_common_context._SDPA_LOCK`，禁止同时全局patch SDPA；没有修改M1文件。默认实际Eva分支仅分解LN/V与gamma/MLP残差，必须在每个ROI的β1审核中对照捕获的native状态；超过固定atol=rtol=5e−5即失败。

根代理可直接复用同一个已加载冻结模型和原episode固定Pi：

```python
from ics.methods.pro_message_extrapolation import capture_native_pair, predict
r_cache, q_cache, capture = capture_native_pair(
    model, images, project, producer=producer)  # images: FP32 [R,Q,3,1024,1024]
result = predict(q_cache, r_cache,
    q_cache.projected_native.cpu().numpy(), r_cache.projected_native.cpu().numpy(),
    coverage64, mean_field64, query_original_hw,
    reference_has_foreground=bool(reference_binary_mask.any()))
# result: fields, work_masks, original_masks, info
```

额外native replay不必重做前20层。`capture_native_pair`自己执行一次paired RGB完整forward（两张图），调用数在capture receipt单列；主法新增suffix工作量20N token-layers。当前包含β1 audit及hard控制，实际局部replay为主法的3倍，不把控制开销或native重新编码隐去。QKV双图四层约384MiB，另有H20、四层native residual审核状态约128MiB、最终场/模型/临时attention；没有保存四层heads×N²全attention。

## 独立入口与真实资源

默认离线factory `load_local_eva_pipeline`使用已有本地timm config和safetensors，`pretrained=False`、strict state加载、CPU frozen eval。assets模板已写现存服务器权重/源码/config/basis路径及审计提供的hash；**projection_enabled=null和原episode身份占位必须绑定，模板不能直接作为已解析run recipe。** Factory仅原episode的固定布尔gate决定Pi，不重开gate。

```text
PYTHONPATH=/root/demo4_cache/env:<repo>/src <python> scripts/run_pro_message_extrapolation.py \
  --assets resolved_assets.json --input actual_episode.pt --out fresh_run
```

支持两种真实包：

1. `images` FP32 `[2,3,1024,1024]` 已按原native RGB变换归一化；`cov/base` 64×64；query_image_hw、reference_has_foreground和`producer`。默认factory会从实际完整forward捕获QKV，Ward特征直接取final LN后固定Pi（Ward再单位化）。
2. 真实FP32 q_h20/r_h20 `[4101,1024]`、q/r `[4096,1024]`、q_patch_ids/r_patch_ids=`arange(5,4101)`，及同上字段；自定义factory返回四个真实BlockAdapter和final_norm/project。旧FP16 q/r缓存不得代替这包。

实际输入包`producer`必须逐字段等于resolved assets的native_state_producer；weights/source hashes、after_block20、1024/64grid、FP32、projection_frozen被检查。建议绑定原RGB变换hash、Pi gate/basis、episode身份和MEAN host producer收据。这里不接受query GT作为算法输入，不进行评分；fields和work/original masks写出后生成seal，后续评测单独读GT。

精确缺口：本地没有timm/实际DINO权重或真实状态包，因此未做真实load/forward/β1审核；根代理已确认服务器实际Eva资产存在且负责新CPU编码。仍需绑定每例原gate/身份、同producer complete MEAN输入场、参考完整mask与original query H/W；然后先做真实β1/native数值与一次成本检查。当前代码不是已获得完整成绩的结果。

## 本地必要检查

`PYTHONDONTWRITEBYTECODE=1 python3 evidence/local/research_20261006/pro_message_preparation_01a1100b/check_toy.py`通过，见toy_check.json：

- 独立dense softmax vs grouped β1最大差2.22e−16；四层native状态最大差6.66e−16。
- 合成Eva gamma残差布局使用独立SDPA参考，最大差2.78e−16。
- 分组质量和=1，空外侧精确0，极端±1000 logits无分母截断。
- 干预续算不调用native QK构造；随机及全零tie例四尺度Ward分区与当前M3逐leaf一致。
- 覆盖4N，参考覆盖N，离散BG按最近组件均值合并。
- η=.125触发限幅时没有虚称β0；δ=0场、work mask、原尺寸mask全部逐点一致，native-region控制确实非zero。

这些是计算合同证据，实际DINO例=0，不作真实分割、原创、校准或运行时证据。
