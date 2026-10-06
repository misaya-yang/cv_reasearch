# Pro M5 首例真实审核：未封存完整预测（2026-10-06）

结论：真实FP32原生pair/QKV capture已完成；同合同计算修复后，两图4个预定probe的β=1在block21–24逐层maxabs均为0。**完整主链仍在第6个参考区域的β=1审核失败，block22 maxabs=0.000244140625；没有完整掩码封存，没有打开query GT或评分。** 原生capture加已经执行的部分预测耗时下界83.99秒，已超过60秒扩展门槛，因此未展开后续4/24例队列。活动进程均已因审核异常退出，当前无M5计算作业。

证据：`actual_audit_v1.json`保存两个真实尝试的原始state、第二次capture/binding及对应远端路径/SHA。所有远端操作仅在用户新授权的46466端点，CPU4线程、RSS守卫8GiB；没有GPU、下载/安装新资源、Git或改动其他代理文件。

## 真实输入与宿主身份

- 实际case：smoke4首项`0_0_72`，reference `000000486122`、query `000000507081`。
- 实际固定gate从该row的`feature_export`中读取`debiased=True`，没有猜测/重开gate。native basis SHA=`9b9b20755a796cbda11bb7220d246ee540106e40cb024e249a5f63f884b6a116`。
- source Eva SHA=`a8c9807ef5e8dabc725c1e2a89439760a900d6af6cdafefea18fc121173643a2`，权重SHA=`45172f209c9583c40538afc26b60a07033e6fcc2e8c30228338e6b2e932e7941`，configSHA=`a71f705b0074e173540d0bdbd3aa940fa8d7d3c6c7f020a683004c46ca605b24`。
- 两图按PIL RGB bilinear1024/ImageNet FP32，raw native H20/21–24QKV从实际SDPA捕获；reference类别完整mask转nearest1024再area coverage。
- 宿主读取已封存nine_public600_v2 `fields/000000.npz`的`mean.control`，明确标注 **existing_FP16_conditional_cache_MEAN_graph_float64**。M5新分支为fresh FP32 native；不宣称这个host-cache变体是Pro纯FP32独立图像主链，也不宣称历史baseline像素复现。

## 两次执行及停止条件

| 尝试 | 模型加载 | paired capture | 当前结果 | 整次wall / peak RSS |
|---|---:|---:|---|---|
| 初版 e1affd2f964b |10.54s|47.18s|预定probe在block22失败，maxabs1.430511474609375e−4|59.38s / 2.84GiB|
| 等价SDPA实现 48e506444426 |10.74s|52.76s|两图四probe四层均0；完成5个reference区域三分支后，后一个reference ROI在block22失败，maxabs2.44140625e−4|98.65s / 2.81GiB|

第二次probe：reference四层`[0,0,0,0]`、query四层`[0,0,0,0]`，各耗时0.0846s与0.0803s。完整阶段已完成5个区域的β1/.75/0三分支，进度时间31.2233s；还没进入全部60个query ROI，也没生成whole work/original masks。完整主链秒数/RSS和mIoU因此仍未测，不把部分阶段计时称完整成本。

修复保持Pro β=.75、四分区、差分4、限幅.5、temperature .07、阈值和审核atol/rtol=5e−5不变。消息用代数等价的`β*SDPA_native(Q,K,V_mixed)+(1−β)*SDPA_inside(Q,K_inside,V_inside)`，其内部系数严格为`p_in+(1−β)p_out`，外部为`βp_out`；分组logsumexp仍统计质量，没有重新计算QK。Toy两式差1.11e−16。小ROI逐点Eva模块使用64行计算批形状来避免小矩阵FP32派发差异；padding不进入attention keys、区域pool或掩码。

当前具体难点：固定QK与native callback在小probe上已经逐层一致，但**不同实际ROI批形状下的原生数值一致性还没有通过全部区域**。尚未分清中/大ROI点运算kernel差异和注意力收缩中的具体贡献。不能临时放宽容差、略过该ROI或只用通过probe替代全区域审核。可复用第二次保存的真实579MiB `native_pair.pt`做一次有界组件定位，避免第三次原生编码；这不是授权展开新方法/参数搜索。

远端目录：

```text
/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/pro_message_audit_v1/
  outputs/state.json                         # 初版失败
  outputs_v2/state.json                      # 第二次失败与probe全层0
  outputs_v2/episode_00_0_0_72/native_audit.json
  outputs_v2/episode_00_0_0_72/native_pair.pt  # 真实双图raw状态/QKV/rope，579MiB
  outputs_v2/episode_00_0_0_72/binding.json
  code_e1affd2f964b/                          # 原审核副本，保留
  code_48e506444426/                          # 第二次实际副本，保留
```

## 下一cohort共享capture接口（本地代码已补，尚未做共享真实测试）

同case/source/transform/gate/basis与M1的描述吻合。`capture_native_pair`现在保留actual block args/kwargs、Q四层KV、raw paired H20、actual raw final LN，不从projected/unit/FP16表示重建；没有改M1活跃文件。下次一次paired forward可返回M5两份cache，并通过`shared_m1_cache(q_cache)`取得M1已支持的cache schema：

```python
r_cache,q_cache,receipt = capture_native_pair(model, images, fixed_linear_Pi, producer=producer)
m1_cache = shared_m1_cache(q_cache)
# m1_cache: layers[index]{args,kwargs,k,v,sdpa_kwargs,heads,calls},
#           raw_h20 [2,4101,1024], final [2,4101,1024], prefix=5, side=64,
#           source, memory_hash, capture_seconds, producer
# final = actual final LN, no Pi/no unit/no FP16；M4原R/Q可读该张量。
```

M5完整QKV约384MiB、H20约32MiB、四层native outputs约128MiB、projected patch outputs约32MiB；新增raw finalLN约32MiB。共享args用已有原生boundary tensor引用，Q的KV是双图QKV的view，不重复该128MiB。仅capture张量约609MiB再加RoPE/少量metadata；模型/临时attention另计。共享cache应顺序消费，继续使用M1/M5共用SDPA lock；两个独立process无法自动共享普通Python tensor，需要根代理在同一episode worker内组织。

该共享API添加晚于第二次远端快照，不能据当前结果宣称它已在真实M1/M5联合链通过。当前已完成的是实际capture与有界审核失败定位，完整分割与质量证据尚缺。
