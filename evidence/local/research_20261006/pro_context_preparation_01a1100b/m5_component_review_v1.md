# M5第6个参考ROI：SDPA消息漂移是当前定位点

结论：一次有界组件诊断精确复现block22失败；在该ROI，pointwise padding64→ceil64/min256无效。用正确native hidden和**真实native attention message**分别隔离时，V投影、输出投影/gamma/residual/norm/MLP均逐值复现。最早可见漂移在ROI query gather/chunk SDPA消息，之后被后缀放大；尚未精确分离query长短、最后2行尾块、B1/B2及gather头stride中的哪一项。

本子任务没有改M5文件或活跃代码/状态。使用新46466端点现存`outputs_v2/episode_00_0_0_72/native_pair.pt`；只在独立`runs/pro_context_m5_diag_v1`写诊断输出。2 CPU线程、4GiB RSS守卫，wall10.440536秒、采样RSS1824768000bytes（约1.70GiB），没有query GT、RGB重新编码或完整encoder前向。仅meta构造真实Eva结构并给blocks[20:22]加载本地已存在的对应权重，在缓存的两层native边界上调用真实块作组件参考。原native_pair文件SHA前后相同，原cache未改写。原始JSON为`m5_component_diag_v1.json`，可执行只读诊断为`diag_m5_components.py`。

## 精确复现对象

参考区域的token数按当前实际mask/正确coverage为 `[1707,239,181,717,188,130,410,144,380]`，包括FG与8个BG区域；第6区域为130 token。不能用旧packet cov重建ROI：当前binding报告与旧cov最大差0.08203125。本诊断重读**reference** annotation，按原case标签生成mask、nearest1024→area64，不读query标签。

原生模型源码、状态与配置来自M5不可变第二次副本`code_48e506444426`，而不是正在修复的模块。Fresh实际paired EvaBlock在已有native H20/H21输入上，两层输出与saved native_states均maxabs0。因此本诊断2线程相对原capture4线程没有引入这两层参考输出差。

## 分离后的结果

| 检查 | block21 maxabs | block22 maxabs | 判断 |
|---|---:|---:|---|
| Fresh真实paired block vs cache | 0 | 0 | source/权重/native边界匹配 |
| captured saved V vs fresh actual V | 0 | 0 | 保存没有改变V数值 |
| 正确native hidden→实际combined QKV的V | 0 | 0 | 该ROI逐点投影不是初始误差源 |
| finish(正确native hidden,真实full-native attention message) | 0 | 0 | 该ROI输出投影/gamma/norm/MLP不是初始误差源 |
| ROI SDPA message，saved V stride | 1.049041748e−5 | 7.688999176e−6 | 消息开始偏离native |
| ROI SDPA message，saved V.contiguous | 同上 | 同上 | layout改变未修复 |
| ROI SDPA message，actual native V stride | 同上 | 同上 | 原V stride未修复 |
| β1完整两层传播，base pad64 | 9.155273438e−5 | 0.000244140625 | block22原容差失败 |
| β1完整两层传播，ceil64/min256逐点padding | 同上 | 同上 | padding不修此ROI |

所有上述V投影/finish的pad64与ceil64/min256隔离结果都是0。此结论局限于这一个130-token ROI和前两suffix块，不能说所有其他尺寸的pointwise GEMM都无风险。

## 容差与dispatch的准确含义

正式代码以 `atol=rtol=5e−5`调用torch.allclose，检查阈值是 `5e−5+5e−5*abs(expected)`，不是仅比较maxabs是否大于5e−5。block21传播maxabs9.155e−5仍通过，0违反元素、最大阈值比0.802668；block22有5个违反元素、最大阈值比1.208173，确实失败。不能仅以raw residual最大值推断gate，也不能据“只有5元素”放宽容差。

CPU profiler实际记录 `aten::scaled_dot_product_attention` 和 `aten::_scaled_dot_product_flash_attention_for_cpu`。仅观察到同一种dispatch名字，不足以认定其内部query tile/尾块策略完全相同。没有GPU计算；PyTorch CPU profiler的GPU可用性探测发出无GPU警告，不是GPU作业。

Saved Q/K stride为 `(262464,64,1)`；saved V为`(64,1024,1)`，fresh实际V为`(64,3072,1)`。虽然clone改变V storage layout，三种布局的ROI消息误差逐值相同，因此这次不是V layout导致的差异。原生query序列为4101，ROI130通过advanced gather后按64分块，最后query块为2；其head stride与连续native-slot slice不同。以上形状/stride差和B1/B2值得下一次**有限同合同**检查，当前数据没有把它们进一步唯一定位。

## 给独占M5修复代理的建议

1. 优先在已有native_pair上比较**attention queries**的形状/slot布局，而非继续调pointwise padding。Padding只加无输出的dummy query行，不能加入K/V、ROI成员、区域池化或mask；Q/K始终使用native缓存。
2. 一份可靠数值参考是用完整native Q形状及实际 `V_mixed`重新计算SDPA，再取ROI输出。这仍真实计算干预，适用于β1/.75/0；不能把β1直接返回cache/native结果。它可能昂贵，只用于有限组件对齐和确定后续可等价优化的调用形状。
3. 若query-only padding/保持native-slot head stride通过，可保持同消息合同作为计算优化；kernel内部tile假说必须用实际结果确认，不能凭“CPU flash”名称保证padding256必然匹配native。必要时与M1无排除self的实际block/SDPA调用形状比较，但不能借M1通过就略过M5全ROI审核。
4. 保留现有容差和所有ROI检查。此次只定位同合同数值问题，不恢复4/24/600队列，不宣称主分割结果或整类机制被否定。后续变更及验证由`audit_4_6`独占完成。

源码指针：M5 `EvaBlockAdapter.values/finish`在本地`src/ics/methods/pro_message_extrapolation.py:343-367`；`replay_region`的native Q/K＋mixed V与分块SDPA在`581-620`，逐层allclose在`623-631`。当前实际诊断绑定的不可变源SHA见JSON的`method_source_sha256`；之后代理编辑会改变本地行号，不能用更新后的模块冒充该诊断源。
