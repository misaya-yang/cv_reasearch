# QK family实质修订v3：pre-RoPE两向角色一致性的小残差

结论：数学定义、可执行完整field/renderer、强简单对照、同一冻结teacher的物理可实现正负例及真实第一case的无GT可行性均已完成。**独立方法增量0**，这是已有QK family修订v3，不为1000目标拆头/层/参数计数。质量未验证，未启动4/24/600、未改M5活跃文件、shared、PLAN或Git。

## 新观测量与旧路线边界

旧`evidence/local/results/reference_qk_v2/REPORT.md:1-42,65-87`记录post-QKnorm/post-RoPE conditional bank、3072维augmentation进入完整FoRIS，包括CRF。旧DEV40主法58.704对native65.161、main−Q-only −0.042，说明该完整旧构造没有实用增益，不能把−6.457点移植到当前MEAN62.93。

本修订固定**block21、全部16头等权**，在实际raw H20上重新调用actual `block.norm1/attn.qkv/attn.q_norm/attn.k_norm`取pre-RoPE Q/K；不猜逆RoPE，不从norm=True中间层替代H20。增加两个cross-image有向角色margin `(a_i,b_i)`及其分歧统计；不再把QK扩维bank接进FoRIS改变分组几何。参考监督只有完整mask的patch coverage，query labels/classes/fold都不进`predict`。

这不是新增外部语义信息：两向量由同一个原生H20和同一冻结teacher映射确定，并不统计独立。相比末层缓存，它需要未存的raw H20与teacher Q/K权重；相比直接H20，它是固定预训练非对称双线性视图的读出，不是信息论上创造了新信息。旧descriptor精确源码未在本次恢复，不能宣称论文原创；保守family计数明确0。未进行新的head/layer/自IoU选择。

## 冻结完整定义

令pre-RoPE每头查询/键为`Q^h(x),K^h(x)`，按每头通道unit（零向量为零），再flatten除以√16。两向相似度：

\[
S^{\to}_{ij}=\frac1{16}\sum_h
 \langle\mathrm{unit}(Q^h_Q(i)),\mathrm{unit}(K^h_R(j))\rangle,
\quad
S^{\leftarrow}_{ij}=\frac1{16}\sum_h
 \langle\mathrm{unit}(K^h_Q(i)),\mathrm{unit}(Q^h_R(j))\rangle.
\]

参考coverage m定义归一角色权重 `w_F=m/sum(m)`、`w_B=(1−m)/sum(1−m)`。固定τ=.07：

\[
G(S)_i=\tau\log\sum_jw_{F,j}e^{S_{ij}/\tau}
       -\tau\log\sum_jw_{B,j}e^{S_{ij}/\tau},\quad
a=G(S^\to),\ b=G(S^\leftarrow).
\]

令`x=tanh(a/.07), y=tanh(b/.07)`，固定最弱一致方向：

\[
c_i=\begin{cases}
\operatorname{sign}(x_i+y_i)\min(|x_i|,|y_i|),&x_iy_i>0,\\
0,&x_iy_i\le0.
\end{cases}
\qquad t_i=z^{\mathrm{MEAN}}_i+0.1c_i.
\]

分歧弃权而不是把分歧当背景；两向都正才加、都负才减。0.1是本次先验固定预算，不是从GT效果得出的可靠校准值。完整输出不重求图、不clamp/minmax、不改阈值。场转FP32→bilinear64→1024、strict>.5→将二值work bilinear到原H/W、strict>.5，两次align_corners=False。

空参考返回空candidate（明确为无目标条件，此边缘输入不承诺相对任意base仍≤.1）；非空但缺背景角色弃权保留MEAN。零Q/K或zero margin修正为0。原生teacher状态/host都只读。

### 同预算强简单完整控制

每行均通过同`base+.1*tanh(g/.07)`/renderer；主一致性自身已在[−1,1]不再tanh：

| 行 | guide/response |
|---|---|
| forward | a |
| reverse | b |
| symmetric_logits | `G((S→+S←)/2)` |
| symmetric_margin | `tanh((a+b)/(.14))`直接response |
| direct_H20 | unit(rawH20)余弦的同FG/BG margin |
| direct_final | unit(actual native finalLN后既有固定Pi)余弦的同margin |
| mean.control | 精确原base，zero residual |

控制不计独立方法。未来机制比较应对这几行中的固定最强完整对照，而不能只胜MEAN就归因一致性有用。Head内unit是明示读出变换，不能称直接复现native attention概率。

## 必要性质与局限

非空有背景条件下 `|t−base|≤.1`（FP32舍入余量1e−6）；双线性插值是非负行和1，故第一阈值前的场扰动仍≤.1。只有阈值附近像素可改，不能删除base远高于.6的强错误或补base远低于.4的强漏检；原尺寸第二段可能受到邻近work二值变化影响。这个性质限制伤害幅度，不保证IoU或面积更好。

两向并非双重证明：同类与干扰可能在两向都被误认，或者真目标只一向表达得好。若Q=K、或实际两向margin完全一样，主法退化到单方向tanh响应，增加的一致性操作没有增量。小样本独立效应未测时不能以两向相关/分歧率充当语义证据。

## 物理可实现正负例

`scripts/run_qk_role_consensus_v3.py --self-check`构造一个真实执行的冻结LayerNorm(eps1e−5)+无bias combined QKV小块，D8、2头×4通道。5个正交零均值hidden状态送入同一个LN；用一份固定W_Q/W_K映射合法单位head向量，**不是直接任意填相似矩阵**。R FG的Q/K为+e1，R BG为−e1；3个query组成目标与两个干扰。理论margin分别：

| 组 | a | b | 对称margin | 一致性 |
|---|---:|---:|---:|---:|
| 真目标 | .5 | .5 | .5 | 正 |
| 干扰1 | 1.3 | −.3 | .5 | 0 |
| 干扰2 | −.3 | 1.3 | .5 | 0 |

真实FP32对称logit margin三组range仅2.6822e−7、symmetric-margin response逐值一样；rawH20/final与参考余弦角色margin近零。base统一.45，一致性完整token、96²工作与原尺寸mask只恢复目标；两个单方向、两种对称、H20/final控制都失败。它说明该耦合可作用于完整输出且不是对称margin的重命名，不说明真实DINO分布中存在有利class关系。

**同一个冻结teacher**保留负例：交换真实目标与一致干扰的hidden角色，算法新增12个false positive token、漏12个true token；query fixture标签只在输出之后作证书检查。另独立Q=K副本验证退化时主场与symmetric-margin完整场逐值一样。详见`toy_check.json/toy_fields.npz`。这不是自然RGB或真实DINO类别验证，不计为实测收益。

## 真实首case无GT feasibility

只用1 CPU，现存M5 raw native_pair第一case `0_0_72`；没有第三次编码、GPU、下载或query GT。Actual teacher block21严格加载已绑定Eva参数，pre-RoPE投影由真实Norm1/QKV/Q/K norm产生，prefix5丢弃，16×64头全部等权；并未使用saved post-RoPE Q/K当pre-RoPE。Reference annotation仅生成合法R mask/coverage。

`actual_feasibility_v1.json`：actual teacher块加载0.157876秒、Q/K投影1.187652秒、全部7修正行+MEAN完整field/mask3.867679秒，含读cache/hash/加载总wall10.050211秒；Linux peakRSS1182016KiB（约1.13GiB）。Meanbase来自已封存nine_public600_v2 `fields/000000.npz`，明确为existing_FP16_conditional_cache_MEAN_graph_float64 variant；direct_final控制读native_pair内的fresh FP32 native finalLN后固定Pi。

实际209/4096个query token两向sign分歧（无语义归因），367都正、3520都负；两向margin最大差0.04111593，主场最大修正0.09426015。主法work mask相对base改2698像素；这是活动量，不是纠错像素。全部输出hash已封存，raw native_pair前后SHA相同。**这个case此前已暴露，不能视作独立确认；本修订尚未读取其query GT或计分。**

当前消耗的teacher投影/残差是缓存feasibility成本。独立RGB部署必须计入真正R/Q raw H20编码、完整MEAN宿主、位置basis冷启动/producer；以后若与完整base一次编码共享，必须绑定同producer及实测调用数。不能据此次没有编码写“完整方法新encoder成本为0”，也不能将过去4线程47–53秒pair编码与本次1线程阶段耗时无标记地相加。

## 接口与唯一后续动作

模块：`src/ics/methods/qk_role_consensus_v3.py`；独立缓存入口：`scripts/run_qk_role_consensus_v3.py`。入口只打开rq/rk/qq/qk、cov/base、rawH20/final control features、原HW，绑定actual pre-RoPE producer/MEAN producer及input hash，拒绝无producer或包含未知teacher层位的packet。Toy强fixture改进不改预测模块/参数；真实feasibility绑定的method SHA与冻结contract一致。

**后续仅由root判断是否授权固定有限4例（主＋所有同预算控制）。** 不按本case分歧率、预测面积或已知旧GT调τ、幅度、头/层、方向正负号或另开变体。若完整效果不胜最强单向/对称/H20/final对照，撤回一致性独立价值；若只有teacher特征对照有用，保留该简单组件。即便有限4有利，也不自动启动24/600或声称原问题解决。

## 新授权固定smoke4已完成：完整输出未胜MEAN

Root随后明确授权同smoke4固定质量测量。已复用M5真实native_pair的其余3例，无新编码；首例feas预测原样复用，没有改动代码SHA、参数、block21/16头或.1预算。全4与全部control按统一12文件hash、case/source/weight/teacher shape/patch5..4100/MEAN producer、完整renderer核验封存，再以独立1CPU读取GT。

1024用同bound600_v2 evaluation manifest的packet.truth/native及storedMean/RCG/fine controls；原尺寸另用M5统一seal中合法annotation路径、class+1、实际JPEG H/W，并在解码GT前绑定SHA。原native/control比较行是缓存1024预测映射到原尺寸的共同最终renderer，不冒称另外重跑独立RGB-original native入口。

| 固定完整行 | work1024 class-summed mIoU | original class-summed mIoU |
|---|---:|---:|
| agreement | 33.822374 | 33.810613 |
| forward | 33.237763 | 33.248256 |
| reverse | 33.694192 | 33.689376 |
| symmetric logits | 33.386311 | 33.383195 |
| symmetric margins | 33.393899 | 33.389982 |
| direct rawH20 | 34.601188 | 34.602252 |
| direct final(Pi) | 32.512259 | 32.520284 |
| MEAN exact host | 35.684576 | 35.695213 |
| native cached comparison | 36.316536 | 36.360804 |
| RCG | 35.473585 | 35.484891 |
| fine16 | 34.911639 | 34.937233 |
| fine64 | 35.907872 | 35.931733 |

首4 work agreement−MEAN **−1.862202 pp**，描述性照片簇配对95%区间[−3.045823,−0.678582]，0胜/3负/1平；agreement−directH20 −.778815（1胜/2负/1平）。虽然agreement比forward +.584611、reverse+.128181、symmetric logits+.436063，但仍低于强base和rawH20。

相对MEAN的组件增删：work新增TP6646、新增FP60281、删除TP3、删除FP1525；原尺寸新增TP1192、新增FP11206、删除TP0、删除FP634。当前损失主要来自新增误检，不能把相对native的累计纠错当作QK组件本身的收益。完整I/U及全部baseline/照片簇2000 RN0区间见`score4_v1/report.json`，相对MEAN的原始像素账本见`component_edits_vs_mean.json`。

一致性response逐点绝对幅度不大于单向或同号对称margin，分歧时又归零；因此比单向/对称损伤少可能只体现更保守的修正，而不是新增类别识别。没有在此活动4上新调alpha或添加matched-movement变体来解释/救分。4例/4类/4照片簇且此前已暴露，只是有限质量检查，不能概化关闭整个QK方向。

### 当前缓存重放成本与真正producer成本分开

| case | actual pre-QK抽取秒 | 全部field/control+render秒 | 本次是否重新做RGB编码 |
|---|---:|---:|---|
| 0_0_72 | 1.187652（已有） | 3.867679（已有） | 否，直接复用封存预测 |
| 1_0_73 | 1.081858 | 2.969307 | 否，复用M5 native_pair |
| 2_0_74 | 1.086953 | 2.982615 | 否，复用M5 native_pair |
| 3_0_75 | 1.104158 | 3.002123 | 否，复用M5 native_pair |

单worker1线程，块参数meta加载/strict load .137428秒；含cache读取/hash/权重身份、首例复制、三例推断/保存的新增batch wall19.023164秒，Linux peakRSS1226668KiB（约1.17GiB），4GiB守卫未触发。独立评分1线程2.767763秒，source、raw/host预处理和输出费用都分段记录。

这些均为当前cached replay。继承的真实native R/Q producer此前分别耗52.76/47.80/48.32/47.16秒（4CPU），未包含本次1CPU时间，更未把两种线程口径机械相加成部署测量。独立RGB部署仍须计完整MEAN宿主和真实rawH20采集；本次没有重编码不等于该成本为零。

本次任务至此完成：`actual_smoke4_seal_v1.json`、`score4_v1/pre_GT_audit.json`、`report.json/report.md`及逐例I/U均回本地。远端分别`runs/qk_role_v3_smoke4_v1`与`runs/qk_role_v3_score4_v1`。**未扩24/600、未改共享PLAN/Git或M5文件；QK-v3独立增量仍0，固定版本当前无完整增益。**
