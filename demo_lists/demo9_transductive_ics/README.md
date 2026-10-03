最新（2026-10-03 晚，Claude，决策学习主线，首批实测）：匹配保持冻结（完整公共 FoRIS），只学习“在哪里下判断”。一个约 10 万参数的小读出，输入全部是与类别无关的成对证据（16 张关系图＋第 6/12/18 层与末层的参考前景/背景边际），初始恰好等于 FoRIS 自己的掩码；评哪一折，就只用另外三折的 1800 个训练 episode 拟合，训练与评测不共享任何图片（Codex 的图片隔离清单 TRAIN2400／DEV241／CONFIRM600，官方 COCO-20i，种子 0）。读出种类在 DEV241 上选定并写入文件后，才打开 CONFIRM600 的标签。结果：确认集放回完整流程、原分辨率、同 CRF：FoRIS 59.78 → 63.33，+3.54，95% 区间 [+1.96, +4.96]，四折 +3.7／+3.9／+2.5／+4.1；开发集 59.12 → 62.57，+3.44 [+1.56, +5.14]。patch 级确认集 +3.86 [+2.58, +5.25]，只看分数的同结构对照 +2.27 [+1.59, +3.20]。增益集中在 FoRIS 失败的 episode（FoRIS IoU<0.3 平均 +6.7，≥0.7 为 +0.1）；600 个里有 42 个掉 10 点以上，其中 35 个是误删了真目标。学习曲线从每模型 600 个 episode 起持平（+4.15／+4.23／+4.03），加入 16 个特征主成分反而降到 +1.62，线性逐 patch 融合（19 个参数）已有 +2.19。无训练的参考读出删除在 241 个 episode 上为负（第 12 层 −15.22，末层 −1.39）。这是开发/确认子集上的结果，不是标准 4×1000 主表；主表、追加训练数据、尺度对齐、只用宿主掩码的变体和“二次观察”正在服务器队列里。表格、预测卡与阶段计划见 [PLAN.md](PLAN.md) “Claude plan, 2026-10-03 evening”；证据 `results/decision_v1/{fit,infer_1_dev,infer_1_confirm}/report.json`；脚本 `scripts/decision_{cache,fit,infer,zoom,protocol,summary}.py`、`tics/decision_heads.py`。

最新（2026-10-03）：Claude extent_v1全241任务完成（官方COCO-20i seed0，40开发＋201后续，完整FoRIS原尺寸class-mIoU/同CRF，逐例基线40exact）。后续基线58.1446；contrast+2.0251，95%区间[-1.6105,3.0226]；boundary−0.2234、round_trip−14.7582、signature−3.6666、zoom_pair+0.6023，均未达预定存活门槛。zoom_query朴素对照+0.9643；两zoom臂轮数不同，不能将差归因参考裁剪。GT cut诊断+11.4799是CRF前模型尺寸选阈值后评分，不是最终输出的严格上界。聚合统计HGB留折测量器仅收回preCRF gap1.06%，不代表空间场无信息。证据results/extent_v1/run/analysis.json、完整241packets与decision_audit.json。此前未执行exact-level/RGB对照40例完成：native65.1614、scalar68.0091、DINO58.9328（−6.2286CI[-11.5218,-.2073]）、RGB36.5676，停止此均值亲和度构造；证据results/native_membership_v1/query_boundary_v1/experiment40/analysis.json。最新用户要求服务器保持开机，等Claude新代码就绪接续；不自动关机、不重跑这两矩阵。研究目标未完成。

## 最新实测：所给联合对应零训练构造未获益

官方COCO-20i、seed0、已有DEV十例（四折3/3/2/2，十个类别），完整公共FoRIS原图输出：native62.9760，严格联合翻转62.1600，同patch删除数低分删除63.0812。同最终删除像素预算的GT算术诊断64.2385，不是三点候选族或CRF方法上限。联合−native−.8161pp，照片连通组探索性95%区间[−1.5818,−.1067]；四折均负，1/10提高。本批每类恰一例，class-sum与episode mean相同，不能套作完整基准分数。

原图联合删FP3189、伤TP5758；朴素对照6285/2562。8793见证无算术/严格翻转违约，邻居FG覆盖不构成错误背景解释的因果证明。全FG检索后来撤销215/262翻转，但剩余动作更偏向误删真目标；同十例完整回放46.18秒：native62.9760、全FG撤销62.8727、随机缩减62.7561、同数量分数删除62.9412，全FG相对native和分数控制的区间均跨0。结果见results/extent_v1/relational_v1_full_fg_causal10/paired_analysis.json。R241也已完成580.303秒、native241/241 exact：query-core删除+.4560pp[+.2815,+.6305]且四折为正，block12/末层固定删除均负；所有样本均为旧DEV，不能推成DINO上限。当前GPU由用户关闭；十种不同机制仅在服务器CPU准备，完整整合尚未验收，没有新十方法分数；SAM3建议只核论文，不自动下载或开卡。保留小证据，原见证压缩释放34,679,325 bytes；共享输入/他人资产未动。本会话未运行T1，以下均为历史状态。

## 较早：参考排序，查询定界；未执行训练

用户已明确纠正先选响应再训练头的偏离，当前训练入口撤回并归档。采用所给“在哪里停”候选：同完整FoRIS分数场，用查询原生特征边界选择水平集；需先胜过完整公共FoRIS、同处理scalarcontrast和RGB边界。现存40复用DEV的patch常值scalar67.05对中点63.14是方向信号，非完整流程优势；10个条件性CPU核心案例通过，内部纹理边界误切反例已保留。无新真实质量结果、无有卡GPU运行。下方此前准备状态已被本节和PLAN首节覆盖。

## 最新用户纠偏：完整方法主线，计划只部分完成

当前（2026-10-03）仍无新真实质量收益。唯一响应条件算法正在无卡CPU准备；实际安装timm的小模块10例已通过，完整训练/预训练DINO实验尚未执行。现有照片与官方掩码真实准备已通过：每折60TRAIN/60DEV、原10评估任务；held类别及所有角色照片隔离，评估仍为复用开发40例，不能称独立确认或充分训练。未租有卡、未下载资产。后续旧条目保留各自历史时点；PLAN首节和STATUS为当前状态。

完整机制对照和Claude计划验收已写入[PLAN.md](PLAN.md)首节。七种候选做过有限开发试验；官方协议四折主表、完整TPA式强对照与独立泛化未完成。最新causal_v3已完成40个复用开发任务、四折各10：公共完整FoRIS65.1614，旧direct-predict+CRF64.2836；轴修正65.3195，相对完整入口仅+.1581pp，区间[-.4139,.7117]。实际199.0877秒实验后浏览器确认866已关机；旧队列不重跑。当前没有新方法的独立强基线优势。

本轮多代理数学审查撤回单参考响应pair-ratio editor：degree归一化仍会重复计算共用证据；单参考拟合好也可在query反转。六个CPU反例检查已通过，记录在results/native_membership_v1/response_relation_contract_cpu.json。唯一正在构造的观测接口是参考FG/BG判别轴上的两种有符号value抑制；9个CPU代数检查通过，未接真实encoder、未训练、未开GPU、无质量结果。它与即时attention差分必须区分；不能把线性改写或CPU通过称方法创新。整体目标未完成。

## 较早用户决定与深入分析

当前全部实例已关机（含无卡环境），平台刷新确认；停止自动开机和新GPU实验，本地继续分析。七候选的有界pilot已执行，不代表原十张研究卡的全部科学问题已回答。

低秩度量现在有具体的能力限制证据：理想单位特征＋正交字典下，已选模型分数修正最多.0231，硬掩码内部余量.5；十例约1049万像素只有1800个在理想可改变范围，任意理想纠错上界也仅+.1841pp，实际变化0。不是FP32完整证书，也不能推出特征无信息。完整推导、实际读出合同和代理不迁移问题已写HANDOFF；小证据见metric_reachability_audit.json及metric_reachability_iou_cap.json。最重要的纠正是先确认干预能触达要修复的错误，保留完整强宿主，再谈优化与扩训练。

## 最新实测与资源状态（2026-10-02）

七候选的有界GPU pilot已完成，没有建立可用方法增益。resume2全部20阶段896.5秒完成后自动关机，浏览器独立确认；现无卡分析，5分钟监控已启用。临时tensor52文件释放1,800,941,454bytes，小结果/源码/checkpoint保留在results/native_runtime_v1与completed_runtime_evidence.tgz；不重建失败缓存。

官方COCO-20i fold0/seed0标准first10开发接口：INSID3 52.4769/FoRIS CRF64.9976。局部PG42.7817对matchedraw42.6840 +.0977pp，探索区间[-.3668,.5765]；fullGL29.4929 −13.1911pp，[-22.9355,-.6806]。背景/网格/prefix/OT都没有实用增量，停止扩大固定构造。E3另外隔离cohort24train/8dev/10infer、10epoch：INSID3/protected/unprotected77.8527，fixed77.8751，FoRIS CRF79.1930；不能与标准first10绝对分比较。protected/unprotected严格开发IoU选epoch1，fixed选10；不是last参数变化代表部署有效。

E10五例原生亮度压力40.0384→54.4173只属意外诊断。已补固定.75完整FoRIS CRF同10任务强控制，baseline64.9976→64.6877，−.3099pp探索区间[-.8494,.2720]，停止扩展。新预测约25秒，guard31.5秒后自动关机且UI确认；分析状态名称合同错在CPU收据，0分析错误已核实，不重跑GPU。现无卡分析；没有新算法或突破主张。所有区间/样本/完整字段见各card分析，当前不是全四折方法分数。

以下是历史转导研究结果，协议及所有权不变，不是本轮重启队列。

# demo9：转导式上下文分割——把没有标注的同类图当作参考

### Earlier preparation snapshot (superseded by latest audit)

Seven candidate algorithms and ten **independent** experiments are specified in [HANDOFF.md](HANDOFF.md) and [PLAN.md](PLAN.md), with code and machine-readable cards in `results/prepared_independent_experiments.json`. Thirteen small CPU/source-operation suites passed; see `results/prepared_methods_cpu_receipt.json`. They validate equations, gradients, hooks, leakage guards and interfaces, not real DINO segmentation benefit. No GPU experiments, new downloads or default ten-job queue. Protected metric has a bounded learned fusion gain and full implicit/active-step gradients; views and prefix prompts have distinct lawful support conditioning; OT relaxes marginal mass. Archived graph propagation remains a cheap control. Historical numbers below retain their original protocol and do not score these candidates.


状态：2026-10-02 立项。信号为正：COCO-20i 标准 episode 上四折都涨，平均 +6.1 mIoU（55.5 → 61.7），叠加在 FoRIS 上平均 +4.2。
方法还很粗，只测了 COCO-20i；离“其余图给真掩码”的上限还差 6 点。

## 问题

上下文分割（INSID3、PR-MaGIC，CVPR 2026 两篇 oral；后续 FROST、REBASE、FoRIS）都是一张参考图对一张目标图地做。
demo4 量到的事实是：单张参考图的证据不够——INSID3 漏掉的前景 90% 是同一物体里参考图没出现过的部分；
参考图从 1 张加到 5 张涨 6–9 点。实际使用时，一个概念通常要分割一批图，这批图本身就是没有标注的同类图。

**问题 A：不增加标注，只用这批无标注图，能拿回多少“多参考图”的收益？**

## 设置

每个类别：1 张带标注的参考图，N 张含该类别的无标注图（它们同时是测试图）。测试图 j 可以使用其余 N−1 张无标注图。
指标沿用少样本分割的算法：每类交集之和 / 并集之和，再对类别平均。COCO-20i，4 折，每折 20 类，每类随机抽 1+16 张（种子 0）。
特征、聚类标签缓存后，所有规则在 GPU 上几十秒跑完（`demo4_incontext_seg/icx/fast.py`，与官方实现逐项核对一致）。

## 结果

### 主结果：与基准相同的标准 episode

参考图和目标图就是 COCO-20i 基准抽出的 episode（每折前 400 个）；方法额外能看到至多 15 张同类的无标注图（其他 episode 的目标图）。
只给该 episode 自己的目标图打分。`results/episodes_eval_f*.json`，脚本 `scripts/episodes_eval.py`。

| 方法（基础方法 INSID3） | 折 0 | 折 1 | 折 2 | 折 3 | 平均 | 相对单样本（80 个类的配对差，95% 区间） |
|---|---:|---:|---:|---:|---:|---|
| 单样本 INSID3 | 54.2 | 58.2 | 54.6 | 55.1 | 55.5 | — |
| 朴素自训练 | 56.1 | 62.7 | 59.0 | 57.6 | 58.8 | +3.3 [+1.1, +5.5] |
| 合并投票 + 往返过滤，4 轮 | 56.9 | 63.0 | 60.5 | 59.5 | 60.0 | +4.5 [+2.7, +6.3] |
| 合并投票 + 互相一致过滤，4 轮 | 59.8 | 64.9 | 60.7 | 61.3 | **61.7** | **+6.1 [+4.2, +8.2]** |
| 其余图给真掩码（上限） | 67.6 | 66.8 | 66.3 | 69.3 | 67.5 | +12.0 [+9.4, +14.6] |

- 四折都涨（+5.6、+6.7、+6.1、+6.2）；80 个类里 47 个涨超过 2 点，12 个跌超过 2 点。
- 做法说明：
  - 朴素自训练：把 INSID3 对无标注图的单样本预测直接当额外参考图，用它原有的多参考图规则。
  - 合并投票：INSID3 的“每张参考图一票、过半数”换成“最近邻最相似的 5 张参考图投票”。
  - 往返过滤：用（无标注图，预测掩码）反过来分割带标注的参考图，与参考图真掩码的 IoU 超过 0.5 才当参考图。
  - 互相一致过滤：每张无标注图当唯一参考去分割其他无标注图，与它们当前掩码的平均 IoU 排在前一半的才当参考图。不看带标注的参考图。
- 互相一致比往返好，一个合理的解释是：单样本失败往往是因为那张带标注的参考图不典型，往返过滤又拿它当裁判；集合内部的一致性衡量的是“在这个概念里是否典型”。
  这等价于众包标注里“没有标准答案时估计标注者可靠性”的问题，带标注的参考图相当于一道标准题。目前只用了最粗的做法（取前一半）。

### 换基础方法：FoRIS（黑盒）

FoRIS 是 INSID3 之后最强的免训练方法（本机复现 4 折 60.5，另一个会话测的）。这里只调用它自己的 `predict`，不改内部；每折前 60 个标准 episode，池子至多 7 张，只做一轮。
`results/foris_stream_f*.json`，脚本 `scripts/foris_stream.py`。

| 方法（基础方法 FoRIS） | 折 0 | 折 1 | 折 2 | 折 3 | 平均 |
|---|---:|---:|---:|---:|---:|
| 单样本 FoRIS | 60.4 | 62.5 | 55.8 | 57.9 | 59.1 |
| 朴素自训练（FoRIS 自己的多参考图规则） | 64.6 | 64.5 | 63.7 | 60.3 | 63.3 |
| 加往返过滤 | 66.3 | 64.3 | 58.4 | 58.3 | 61.8 |
| 其余图给真掩码（上限） | 71.5 | 67.7 | 70.7 | 64.2 | 68.5 |

- 朴素版本四折都涨（+4.2、+2.1、+7.9、+2.4，平均 +4.2），说明这一步能叠加在别的方法上。每折只有 60 个 episode，区间还没算。
- 往返过滤在 FoRIS 上不稳（折 2、3 反而不如不过滤）；互相一致过滤在 FoRIS 上还没测（每轮要 N² 次调用，FoRIS 每次约 0.5 秒）。

### 流式设置（每类固定一张参考图）

每类 1 张带标注参考图 + N 张无标注图，无标注图同时是测试图；种子 0。`results/stream_eval_v0.json`，脚本 `scripts/stream_eval.py`。

| 无标注图数 N | 单样本 INSID3 | 朴素自训练 | 合并投票 + 往返过滤，5 轮 | 其余图给真掩码（上限） |
|---:|---:|---:|---:|---:|
| 4 | 59.7 | 59.0 | 59.9 | 64.2 |
| 8 | 58.9 | 62.0 | 61.8 | 67.9 |
| 16 | 57.9 | 61.3 | 64.2 | 69.6 |

- N=4 没有收益，N=8 起有。第 0 折另抽了一组（每类 48 张，固定前 16 张测试，`results/stream_eval_f0_pools.json`）：池子 16 / 32 / 48 的收益是 +6.0 / +5.0 / +6.4，
  上限 67.3 / 67.6 / 68.9。也就是 8–16 张就够，再多基本不涨。
- 每类只有一张参考图时方差很大：同一折换一组参考图，单样本从 62.0 变到 51.3。所以主结果用上面的标准 episode。
- INSID3 原有的多参考图规则在参考图多时不可靠：16 张真标注参考图下，逐图过半数投票 67.7，合并最近邻 69.6；个别类别（滑板）参考图越多越差。

### 伪掩码差在哪（第 0 折，`scripts/stream_diag.py`）

第 1 轮掩码的错误分成两类，各占一半损失：

| 类型 | 例子 | 第 1 轮精确率 / 召回率 | 修哪种错误能到上限 |
|---|---|---|---|
| 召回低（多实例、大物体） | person、hotdog、chair | 0.94 / 0.54（person） | 补上漏掉的前景：56 → 78 |
| 精确率低（小物体） | skateboard、backpack、wineglass | 0.42 / 0.95（skateboard） | 去掉误报：36 → 72 |

看图确认：小物体类的误报是个别图整张分错（把栏杆、坡道、穿红衣服的人当成目标），不是每张图都带一点污染。

### 试过、没有再提高的信任规则（第 0 折都在 66 左右）

区域判断与带标注参考图的 patch 判断一致才信；跨图共识（其余图里过半认为是前景才信）；把伪掩码之外的区域当“未知”而不当背景（person 从 52 到 76，但多数类变差）；
纯标签传播（没有 INSID3 的种子和聚合，63–64，多轮后漂移）。

## 与已有工作的关系（需要正面回应）

把预测掩码加进参考集在经典少样本分割（PPNet、半监督 FSS）和医学分割（级联式上下文分割）里有先例。
这里的不同点必须落在：免训练的基础模型设置、往返验证、多参考图聚合规则的修正、收益随池子大小的规律，以及能叠加在任意单样本方法上。
这些还没有做完，现在只能说“方向成立”，不能说“方法成立”。

## 下一步

1. **可靠性估计做成正式的方法**：把“取一致性前一半”换成按可靠性加权（众包里的可靠性估计，带标注参考图作标准题），并在 FoRIS 上验证。
2. **伪掩码的两类错误**：召回低的类需要“未被选中的区域不当背景”，精确率低的类需要挡住整张分错的图。这是从 61.7 到 67.5 的空间。
3. **稳健性**：池子里混入不含目标概念的图；每折用全部 1000 个 episode；多个种子。
4. **更多数据集**：LVIS-92i、PASCAL-Part、PACO-Part、FSS-1000 等（INSID3 的 9 个基准）。标注文件在 Google Drive 上，服务器访问不了，需要用户提供；
   COCO-20i 的官方掩码也一样（目前用的是从官方标注重建的掩码，单样本比论文低约 1.3 点）。
5. **论文的对照**：必须和“把各方法自己的 5 样本设置”比标注成本，和经典的半监督少样本分割（PPNet 等）比思路。

### 本会话接续：固定预算的混负池验证（开发诊断已完成）

四折共400 queries/80 classes已完成：真正官方BF16单图57.394，共同FP32后端compact单图56.268，clean无筛选pooled59.829/AG60.525，mixed无筛选pooled58.725/AG59.704。mixed AG−pooled为+0.979点、class-paired CI[−0.074,+2.000]；mixed AG−真正官方BF16为+2.310、CI[−0.666,+5.424]。不能说已证实筛选的独立优势，也不能把共同FP32后端配置当成未修改官方。各折大特征已删除；结果及反例保留。这是旧重建掩码、另一抽样/精度配置的开发证据，不是新的官方掩码 E0/E1 表。

TPA近邻已核读并进入 `PLAN.md` E1b：同池、同底座、同标签预算，比较便宜扩库与跨图验证的独立收益/整流程代价。准备好的 paired cache 经10例逐像素核对全部exact，90MB烟测缓存已删；不能外推全400/所有种子都exact。

首个confidence-free FG/BG单原型对照30 episodes/fold0/seed0只读复用200-episode paired gallery、重建掩码：1shot60.872，naive56.787，support+pool bank patch48.977/cluster49.686，true64.991。cluster−1shot约−11.186点，episode-paired CI[−16.4,+0.7]。这是弱读出，停止扩大，不能据此否定或宣布超过原版TPA；没有原文的置信度/输出融合。结果 `results/prototype_bank_pilot_f0.json`，脚本 `scripts/prototype_bank_eval.py`。下一步强TPA适配控制必须面对二分类阈值退化与硬掩码接口，先冻结合同。

用户明确让本会话转向demo9。独立执行目录`/root/autodl-tmp/demo9_transductive_ics/results/open_pool_v1`，脚本`scripts/open_pool_{cache,eval,queue,analyze}.py`；不覆盖demo4旧脚本或结果。四折各取前400标准episodes内每类前5个，共400开发queries；同一support/query、同样实际池数最多15，清洁池与替换约一半不含目标图的池配对。负图由COCO split的目标存在性仅用于采样，推理不可见；全部query/donor GT槽位在ImageSet中置零，只给support注释。比较单图、朴素多参考自训练、无筛选pooled、已有agree/tophalf4轮、往返阈值.5；这是检验已有收益的实际条件，不是已建立的新方法或独立论文测试。

每折编码与100条评分已完成，大特征逐折自动删除，无下载。保留manifest、每episode I/U、最终小patch掩码、筛选记录和清理凭据。初始接口错误发现精度合同差异：官方`_extract_features`实际上BF16 autocast，原demo9快缓存则在编码后转FP32再归一化/去位置/解码。三对在相同FP32后端/B1导出配置下官方与紧凑reader mask IoU均1.0，reader与旧ClassSet逐位一致；原BF16官方掩码相对缓存约.91665/.97891/1.0，不冒称同一precision。实验另保存真正原始BF16官方单图对照，原错误证据保留，缓存未重抽；所有臂共用同一标注与评分合同。

PPNet(ECCV2020)已用无标注图丰富部件原型，PANet(ICCV2019)已有反向prototype alignment；“加伪参考/往返/共识”不能单独充当新颖性。需要实际辨明混负池中的错误和强同信息控制之后，再发展有效机制。

## 文件

### E4a 单种子假设的首次因果核查（2026-10-02）

假设来自用户：部件已被参考支持，但最终仍要求与单seed相似。源码核实为真；不等于已测出大的任务损失。
固定reference/mask/candidate/seed及seed-area boost、原.2阈值，仅删除intra相似度，official paired COCO fold0前30/seed0：1shot62.035→49.694，pooled P1参考57.595→54.905。新增像素TP比例分别21.6%/34.3%；仅加新增GT前景的宽松界分别66.913/59.869（+4.878/+2.273）；后者episode-paired CI[+.150,+3.746]。这是逐像素GT诊断，不是合法方法或可实现的整区域oracle。

强FoRIS完整pipeline前10个相同标准episode：56.768→56.989（+.221，episode CI[−.8,+1.2]），只补新增GT前景的界57.769（+1.001，CI[+.1,+1.7]）。native replay10/10 exact；只改变prior中的intra因子，其后归一化/boost/disagreement/reweighting全部保留；FP32 native feature forward当例RAM复用，无新磁盘cache/下载。未达到大瓶颈预测，不扩单因子消融。小样本不否定所有部件取证方案；该局部界不能解释全部漏分。

源码/数据/结果：`scripts/seed_bottleneck.py`、`scripts/foris_seed_bottleneck.py`；`results/seed_bottleneck_pilot_f0{,_summary}.json`、`results/foris_seed_bottleneck_pilot_f0.json`。FoRIS镜像文件SHA及重要实现差异记录在PLAN E4a；不是论文排行榜复现。其他执行者200例seed/ridge诊断只读核对，未重跑或修改。下一步须检验局部证据的独立可靠性，原native最近邻FG/BG票及FoRIS多原型/困难BG对比不能被改名当新机制。

交接说明在 `HANDOFF.md`，下一步实验在 `PLAN.md`。

- `tics/`：方法本体（缓存特征上的 INSID3、可靠性估计、迭代）。
- `scripts/cache_episodes.py`、`run_episodes.py`、`run_blackbox.py`、`stats.py`：建缓存、评测、黑盒基础方法、配对统计。
- `results/`：本页表格对应的 JSON。
- 本页的数字出自探索阶段的脚本，在 `../demo4_incontext_seg/`（`icx/fast.py`、`scripts/episodes_eval.py`、`foris_stream.py`、`stream_*.py`、`pseudo_shots*.py`）。
  那批缓存是逐张编码的，单样本一行与官方实现不完全一致（见 `HANDOFF.md` 的“踩过的坑”）；各行之间的差是配对的。新脚本已改为成对编码，绝对值以 `PLAN.md` 的 E1 重跑为准。
- 服务器：代码 `/root/autodl-tmp/demo9`；特征缓存 `/root/demo9_cache`，用完即删，目前为空。
