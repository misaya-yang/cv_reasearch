# 统一任务冲突完整组合6200：结果、机制边界与实验资产

2026-10-11。本批已完成：6200例、三新完整臂、CLI及original共37200张最终Mask全部先封存后评分，直接I/U、四类编辑、实际同预算位置和官方class聚合核验通过。**当前冻结配方未达到多数据集稳定领先，不采用；研究goal继续active。**这是完整组合结果，未以裸cue分数淘汰方法，未按数据集、GT、case或frame选赢家。

机器为本机M4 Max 36GiB，branch `codex_m4`，无可用服务器。本批仅消费已有whole R/Q O24；没有新编码、下载、样本、views、训练、CRF或大dense归档。Pro200、九窗1300、自然hardMask别名库存保持完成／关闭，不重复。

## 同条件完整输出

唯一协议[protocol.json](../../evidence/local/unified_task_conflict_composition_20261011/protocol.json) SHA `4fd6b2cb3a9ff3ef7b56eeb22833f16c2f6d051f296df9db81d83d5134a7e7cd`，设计见[统一组合规格](UNIFIED_TASK_CONFLICT_COMPOSITION_20261011.md)。完整FoRIS B为默认；原MEAN G和有界参考R0定义新增／删除争议域。合法参考软FG/BG各128个面积quantile多模bank、温度.07，query预测共同FG仅提供外观中心；保留原.5有界宿主修正。配去整个query-anchor步骤的完整消融和**同frame实际增／删数量**的FoRIS原margin位置控制。

下面均为百分制mIoU；官方fold/class固定槽位COCO40、PACO303、LVIS920、PASCAL56、SUIM7，缺槽0，class pooled I/U→每fold class均值→fold均值。各域单列，不取跨域平均。6200是已经曝光的固定队列，不是完整官方测试集或独立新图确认。

### 主口径：CLI1024

| 队列 | FoRIS B | MEAN G | R0 | 旧C | 旧N | 旧K | 完整主臂 | 去query anchor | 新同预算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO200 | 63.170230 | 63.827315 | 62.316424 | 63.888679 | 63.962037 | 63.949968 | 62.913079 | 63.184992 | 62.950982 |
| PACO600 | 38.438053 | 39.230131 | 40.441563 | 39.492237 | 39.517861 | 39.510933 | 38.362589 | 39.161624 | 38.322917 |
| LVIS1400 | 43.762466 | 43.806129 | 42.679952 | 44.398755 | 44.317619 | 44.319063 | 41.208266 | 42.028370 | 41.218415 |
| PASCAL原生2000 | 54.788956 | 55.249268 | 55.785157 | 55.508974 | 55.677669 | 55.659050 | 55.525897 | 55.665384 | 55.477743 |
| SUIM原生2000 | 59.182775 | 59.260172 | 58.426355 | 59.419885 | 59.632460 | 59.623091 | 59.845815 | 59.838209 | 59.829132 |

### 次口径：original独立读出

| 队列 | FoRIS B | MEAN G | R0 | 旧C | 旧N | 旧K | 完整主臂 | 去query anchor | 新同预算 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COCO200 | 63.358900 | 64.006821 | 62.459822 | 64.080288 | 64.140667 | 64.127715 | 63.029863 | 63.299341 | 63.067123 |
| PACO600 | 38.460868 | 39.352045 | 40.774944 | 39.664615 | 39.684486 | 39.678407 | 38.297529 | 39.166088 | 38.268380 |
| LVIS1400 | 43.912229 | 43.872534 | 42.771402 | 44.519548 | 44.438002 | 44.435575 | 41.298864 | 42.113680 | 41.309201 |
| PASCAL原生2000 | 55.496178 | 55.715834 | 56.534176 | 56.255810 | 56.394769 | 56.386686 | 56.236795 | 56.100911 | 56.190965 |
| SUIM原生2000 | 59.660311 | 59.549329 | 59.169070 | 59.907918 | 60.095403 | 60.083051 | 60.271474 | 60.405745 | 60.248221 |

主臂CLI相对B为−.257151／−.075464／−2.554200／+.736941／+.663040；前三域下降。PASCAL仍低R0和旧N，SUIM点数高于旧N但query-photo区间跨0，不能宣称稳定优于全部强控制。去query-anchor完整消融也在LVIS明显低B；不能把所有失败都归于中心平移。

## 成员级补真、补假、删真、删假

这是主臂相对B的original真实gross；不是净TP/FP，也没有把像素线性换成mIoU。

| 队列 | 补真 | 补假 | 删真 | 删假 | 放行D_add真比例 | 放行D_add假比例 |
|---|---:|---:|---:|---:|---:|---:|
| COCO200 | 229,190 | 447,654 | 315 | 749 | 79.61% | 69.05% |
| PACO600 | 68,259 | 230,535 | 309 | 16,274 | 96.24% | 88.83% |
| LVIS1400 | 595,165 | 2,673,109 | 225 | 7,420 | 98.10% | 96.81% |
| PASCAL原生2000 | 714,068 | 532,252 | 18,316 | 8,248 | 86.38% | 85.84% |
| SUIM原生2000 | 6,465,194 | 8,448,493 | 439,332 | 892,099 | 85.45% | 84.80% |

LVIS新增域606,677个真点与2,761,089个背景点，主臂分别放行98.10%／96.81%；新增595,165真点时同时加2,673,109误报，仅删225真点／7,420误报。输出接近在弱宿主范围内放行现有proposal，而非可靠的成员语义裁决。

相对旧C/N，确实恢复其删掉的一些真目标，也恢复大量已经被拒绝的误报。例如LVIS主臂相对N补793,774TP／3,893,899FP，同时删4,521TP／13,099FP。保住B-only真覆盖本身不能证明完整组合收益；需要同时拒绝B-only和其它proposal里的假目标。[逐例双frame账本](../../evidence/local/task_conflict_execution6200_20261011/episodes.jsonl)保存全部三新臂对B、主臂对旧C/旧N/去anchor/同预算的四编辑与TP/FP闭合。

original主臂减同预算为−.037260／+.029149／−.010337／+.045830／+.023253点。预算严格同增同删但位置不同，这一净差很小且正负不一；不能把完整新增覆盖的收益全部归为新参考任务识别。CLI对应净差与区间如下。

## 必要配对区间

复用原query-photo5000与connected R/Q-photo2000，seed0；仅读已封存整数账本，未再次读GT/Mask/raw。以下均CLI百分制差，RQ区间条件于原图池；重采样缺失官方slot仍0，尤其稀疏class/大连通组会使区间偏离固定point，不能当新图泛化证明。

| 队列 | 主臂−B | query-photo95% | R/Q95% | 主臂−同预算 | query-photo95% |
|---|---:|---|---|---:|---|
| COCO200 | -0.257151 | [-1.1898, +0.5462] | [-1.2561, +0.6792] | -0.037904 | [-0.0833, +0.0164] |
| PACO600 | -0.075464 | [-0.2499, +0.0707] | [-0.2643, +0.1837] | +0.039672 | [+0.0234, +0.0525] |
| LVIS1400 | -2.554200 | [-2.8923, -2.4222] | [-2.2396, -1.6354] | -0.010149 | [-0.0162, -0.0090] |
| PASCAL原生2000 | +0.736941 | [+0.4806, +0.9875] | [+0.1575, +0.9207] | +0.048154 | [+0.0280, +0.0720] |
| SUIM原生2000 | +0.663040 | [+0.2714, +1.0617] | [+0.6320, +2.0674] | +0.016683 | [-0.0214, +0.0641] |

[完整六项对比及两frame区间](../../evidence/local/task_conflict_execution6200_20261011/sensitivity_intervals.json)保留main−B/C/N/source/samebudget及source−B。未重新bootstrap旧／旧组合，没有用区间选择方法或frame。

## 同R/Q只换Mask：内部确实变化，任务判断仍不够可靠

原先固定的PASCAL bus WINDOWS `3/312` 与VEHICLE_SIDE `3/422`沿用相同R/Q RGB、crop和O24编码身份；不同合法R Mask不是新造Mask。FG质量MF为809.71875／3301.82422，source尺度²为.596958／.681103，query anchor质量MA为1340.30859／3005.74609，角色bank index hash也不同。参考指定的任务确实进入角色bank、尺度、anchor和输出。

主臂原图最终Mask XOR为53,535像素（B为54,281），CLI为412,502；这不是Mask-blind。两任务原图IoU为72.815846／84.314173，分别低于B的73.852966／86.603783；S_A都约.8629并通过。**发生响应不能推出按参考完成了任务。**两GT交42845、并94757，WINDOWS独有95、SIDE独有51817，是层级重叠任务，不按互斥类解释。[固定pair小诊断](../../evidence/local/task_conflict_execution6200_20261011/pas_mask_switch_diagnostic.json)只读两个新最终packet和已有小账本，不重生成cue、不读新GT/raw；这一个pair也不是跨数据集因果复现。

## 能成立的条件反例与尚未成立的理论

在理想实数算子中，source-only的相对中心在距离中相消：`||(q−mu_RF)/s−(r−mu_RF)/s||²=||q−r||²/s²`。合法FG/BG bank与Mask条件尺度仍有作用；单纯写成“相对中心”没有额外增加跨图信息。主臂等价将全部Q平移`−(mu_QA−mu_RF)`，同时移动FG和BG，没有验证两角色共用该位移。

考虑合法单模特例：source标准化FG bank全0、BG bank全b。此时`logLR=(||b||²−2u·b)/(2*tau)`。若错误的query anchor恰为一致BG外观，其中心化后u=0，`S_A=tanh(||b||²/(4*tau))>0`，仍通过兼容检查。例如query BG与source BG外观恰相同，source-only可给负分；把它当query FG中心后则给正分，可传播到争议新增区。这个条件反例直接说明：来自预测anchor的自洽性不能证明参考任务正确。

实际6184个可用anchor无一次incompatible；15个empty、1个reference缺BG，恰按冻结规则回B。实际近union扩张、弱同预算位置净收益与该风险吻合，但**不证明所有多模case都必然正分**：平均logLR正不能一般推出平均tanh正。source-only本身也有LVIS损失，故不能仅归罪平移。

这些结论定位当前anchor／角色核／有界组合的判别不足；源特征、B/G/R0和合法R Mask仍保留，不是整个计算图不可逆地丢失任务信息。尚未得到支撑原创算法的普遍跨域信息损失定理，也未宣称本配方原创或有效。后续应直接研究“错误anchor是否能被参考BG反证、合法BG怎样约束新增成员”，同时保留FoRIS有效链和完整强控制；不靠重新调tau、alpha、按域阈值或复跑失败裸头推进。

## 实际成本与保留资产

成功生产871.670547s wall／776.133071s CPU，峰值RSS1.657GiB；6200例全部MPS32，无CPU fallback。密集角色核178.795293s，已有raw读／验SHA440.203450s，两frame组合／预算／无GT核对186.348077s；这些是阶段内成本，不重复加总。实际12399 raw loads（12400逻辑R/Q请求中1次同key复用），8016不同key，重复请求container合计202,687,259,905 bytes；库存不同container为130,210,686,119 bytes，二者不是新磁盘写入。MPS sampled allocated最大71,406,080 bytes，driver最大1,101,791,232 bytes（约1.026GiB），采样值不是系统全程精确峰值。

完整score/directverify 215.150541s wall／225.803756s CPU、RSS1.144GiB；queryGT读取6200次，37200新mask直接I/U，旧六臂I/U直接复用atlas。root仅整数独立核验557,158项、0.267590s，0新GT/Mask/raw。六对区间2.659207s wall／2.585197s CPU，0GT/Mask/raw。冷编码成本并未重新测量，不由缓存时间推断冷启动或FoRIS速度。

一次首例启动因MPS联合CPU/FP64转换报错，0最终mask／record；已修为先CPU copy再FP64 cast，实现原冻结契约。失败源码、log/config保存在`failed_start_mps_exit_dtype/`，失败wall／CPU没有成功保存，记unknown，不隐去。三次执行代理stream中断未终止任务，主Sol唯一启动producer/scorer/CI，design补评分代码，signals只读核代码。成功生产没有resume或重复worker；旧CPU64协议未执行、快照保留。

QueryTruthGuard实际阻断5417个唯一query-mask路径，整个生产attempts为0；当前producer没有annotation读取。这个guard是列出路径的访问保护，不能表述为对任意annotation文件的通用隔离。root另核当前输入／代码与冻结SHA一致，源码在整个成功run中未改。

主要复现入口均在[执行目录](../../../evidence/local/task_conflict_execution6200_20261011/)：

- `inputs.jsonl`／`score_bindings.jsonl`／`raw_metadata.json`／`preflight_receipt.json`：原6200身份、crop、合法R/Q和raw绑定；所有大raw仅链接原`../cv_data`，不复制。
- `producer.py`／`dense_core.py`／`protocol.frozen.json`／`config.json`：实际生产代码、精度、无GT保护与source SHA。`predictions/000000.npz`到`006199.npz`保存仅三臂CLI/original最终packbits；`records/`保存参数统计、编辑数、来源、成本与前三例数值参照。
- `sealed.json`：所有6200 packet SHA、先mask后GT证据；seal SHA `99d4b644416dc85822c36acd3905331c61c3494dc68b4150931dfb77b14591a9`。
- `score_and_verify.py`／`episodes.jsonl`／`report.json`／`verification.json`：全部直接I/U、有限gross、固定官方class贡献与同预算位置／padding检验。
- `intervals_continue.py`／`sensitivity_intervals.json`、`root_scalar_verification.json`及[scripts/sol_verify_task_conflict6200_scalars.py](../../scripts/sol_verify_task_conflict6200_scalars.py)：所需区间与独立整数终点复核。
- `production_receipt.json`／`runtime_census.json`／`root_live_binding_receipt.json`／`pas_mask_switch_diagnostic.json`及三个运行log／失败目录：实际资源、路径、阶段和机制边界。

原始DINO缓存、dataset、权重、原六臂最终Mask/IU、九窗1300和Pro200完整失败证据均保留。本批终点是完成并拒绝当前配方，不能把批次收尾当主研究完成。家中Opus入口见[当前基本状态](OPUS_RESEARCH_STATUS_20261011.md)，执行以[PLAN最新首节](PLAN.md)及主state为准。
