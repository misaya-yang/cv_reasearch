# demo9 下一步实验计划

## Latest user-authorized experiment: conditional candidate ranking (2026-10-02)

The user explicitly accepts frozen-encoder small-head training. Retired M1, old info/SAM/GIC queues and native f2 remain retired. Acquire first 10 native paired reference/query full-token samples, then append frozen candidates and donor evidence within a 2GB pilot budget; no new downloads. The acquisition is preparation, not a gain result.

Pilot40 completed; its two held classes offered only 0.132pp candidate-choice room (direct90.6467 vs GT-choice90.7784, 12 queries). It is a fit/interface check, not a useful efficacy test. Retain compact evidence and retire its 1.704GB token files.

Next fixed development acquisition: all20 fold0 classes,6 episodes/class requested; 12 training classes,4 development classes,4 test classes with all-role photo purge before extraction. A 6GB temporary limit is justified by broader supervision, not an unlimited cache; shared disk37GB free, reserve5GB. This remains development, not a new blind cohort. Prepare pair features, then first20 candidate sets; expand only if TRAIN+DEV candidate-choice diagnostic room is >=2pp (test labels excluded from this decision). Include matched original8scores, native naive and documented cached AG4 before further ranking evaluation. No static weak-control win is a contribution. Current historical M15 tables remain a different budget/protocol.

Strong controls, preregistered before v2 test results: a fixed Ridge and HistGradientBoosting scalar readout uses the same train labels and native K7 bank (original8score raw/rank/zscore/direct flag), with no hyperparameter sweep. Compare with scalar MLP/RQ/RQdonor, native naive, documented cached AG4, direct and GT-choice diagnostic. Bootstrap whole test-photo connected components; episode and class resampling are sensitivity reports, not independent-photo evidence. Counterfactual first fixed training pair is only a data-interface check; failing one pair does not refute the eligible-pair family.

Counterfactual data-only intervention is now authorized: fixed10 training-role pairs (preselected before candidate predictions) completed; 4/10 meet unchanged both-target oracle>=.5, disjoint best sets, cross-regret>=.1 across4 concept pairs. Only these frozen8 training tasks go to ALL supervised controls. First-pair failure remains recorded; no replacements or rounded threshold changes. Same shared RGB/tokens/candidate union/P1 union/provenance/direct0 across each paired task; only legal reference annotation, derived support scores and evaluator labels differ.

1. Assumption: concept-switch supervision helps the existing reader use the reference rather than a dataset-wide candidate preference.
2. Prediction: unproven; measure paired-task correct-switch rate, training fit and unchanged held24 native scores, not surrogate loss. The earlier RQ/RQdonor score72.184 vs direct73.714 on this development pilot is the no-CF control.
3. Match: retain only a net task-beneficial arm, then design a larger unseen-photo/class evaluation; 4test photo-components do not establish solid evidence.
4. Mismatch: stop this data-only intervention; no layer/epoch sweep. Diagnose condition-use and candidate-source errors from retained small records, retire large features when no further necessary comparison uses them.

1. Assumption: candidate-conditioned reference/query evidence, and possibly donor evidence, can improve held-class whole-mask selection beyond matched supervised scalar scores. Old info already read full query tokens; this is not a first-full-token claim.
2. Prediction: no justified point estimate. Test a preregistered useful-effect target of +2pp versus the matched scalar selector; donor increment must separately exceed paired uncertainty. Baseline, naive pooling and candidate GT-choice diagnostic accompany the same samples. This pilot cannot establish publication-level gain.
3. Match: retain the simpler successful arm, freeze its protocol, then acquire an independently isolated cohort. Counterfactual support-mask-only tasks must preserve RGB/pool/candidates and share supervision across controls.
4. Mismatch: inspect train fit, held-class selection regret and degradations; do not extend epochs/layers or rebuild the retired 19GB bank without evidence identifying a distinct failure. Missing donor tokens makes RQpool not evaluable.

Epoch selection uses development photos/classes only; never the test set. Every support/query/donor photo role is purged across splits. Full final 4096x1024 tokens are stored per native pair context, not deduplicated across different batches. Candidate labels retain native I/U and original-resolution I/U separately. First ten samples are smoke/data-interface evidence, not method results.

2026-10-02 写。代码都已写好并在服务器上用 30–100 个 episode 跑通过（冒烟测试，不是结论）。
每个实验先写四行（检验什么 / 预测 / 符合则 / 不符合则），再给命令、耗时和停止规则。按顺序做，一次只跑一个。

## 当前判断和下一步（2026-10-02 夜；接手的 agent 从这一节读起）

### 归因结果（2026-10-02 深夜，第 0 折 400 个 episode；先读这一段，它推翻了下面的 M1）

脚本 `scripts/probe_errors.py`、`probe_select.py`、`probe_reliab.py`，结果 `results/probe_errors_f0_400.json`、`probe_select_f0_400.json`、`probe_reliab_f0_400.json`。都用了真值，是诊断不是方法。

| 行 | mIoU |
|---|---:|
| 单样本 | 55.0 |
| 不筛选合并投票 / 现有方法 | 57.0 / 59.1 |
| 只用“容易”的池内图（单样本掩码 IoU ≥ 0.5，占 62%）的伪掩码 | 62.1 |
| 容易的用伪掩码、难的给真掩码 | 67.3 |
| 全部给真掩码，合并投票 | 67.7 |
| 16 个单参考假设（直接单样本 + 经每张池内图的两跳）里按真值挑最好的一个 | 70.7 |
| 只用一张池内图加它的真掩码，按真值挑最好的那张 | 75.1 |
| 图集级聚类后用真值给模式贴标签（M1 的上限） | 54.2 |

1. **M1 已否定**：按真值贴标签的上限 54.2，低于单样本。区域原型在图集层面聚不纯。可脱离性那一步没有测的必要了。
2. **我之前说“误报是滑板上的人”是错的**：误报里 71% 是没有标注的背景（看图是滑板下面的坡道、船下面的水），人只占 4%；74% 贴着目标。漏检里 76% 是已找到实例的部件。
3. **真正的结构：用哪张参考图，比怎么合并重要得多。** 一张选对的参考图（75.1）胜过 15 张真掩码合并（67.7）；即使只有伪掩码，16 个假设里也几乎总有一个很好的（70.7）。合并投票把这个优势平均掉了。
4. **缺口的直接原因是那 38% 的“难图”**：它们的单样本掩码平均 IoU 只有 0.20。查询图对参考图来说难的时候，池子里和它像的图也同样难、同样错。给这些难图真掩码值 +5；完美知道谁是难图也值 +5（62.1 对 57.0）。
5. **无标注信号分不清对错**：区分难易，互相一致性 AUC 0.81，全部特征的逻辑回归（留一类）0.84，往返 0.70；按互相一致性保留 62% 得 58.8。挑假设，最好的信号（把候选掩码当参考反推参考图）与真实 IoU 的秩相关 0.55，挑出来的只有 54.5–56.7。用容易的图去修难图，难图的 IoU 只从 0.20 到 0.37，对查询没有额外帮助（61.7）。按往返顺序逐张传播 56.9，也没用。

**结论**：解码、聚合、筛选规则、图集聚类、顺序传播都试过了，无标注的做法都停在 57–59。剩下的空间（到 62、67、71）都卡在同一件事上：没有标签时判断一张掩码对不对。这和 demo4（按真值选簇 82，规则追平基线）、demo8（学出来的验证器不如基线）是同一个障碍。

**下一步只有一个值得先做，而且不用 GPU**：`probe_select_f0_400.json` 里已经存了每个 episode 16 个候选的真实 IoU 和全部无标注得分，`probe_reliab_f0_400.json` 里存了每张池内图的特征和真实质量。在这两张表上离线训练打分器（留一类交叉验证），看“挑假设”和“分难易”各能到多少。挑假设到不了 62，方向按 G1 处理：+4 的收益、朴素的机制，不够 solid accept。


### 量到的

第 0 折，官方掩码，缓存与官方代码逐项一致，池子最多 15 张。结果文件 `results/probe_decoder_f0_400.json`、`probe_decoder_f0_200.json`、`probe_seed_f0_200.json`，脚本 `scripts/probe_decoder.py`、`probe_seed.py`。

| 行 | 400 个 episode | 前 200 个 |
|---|---:|---:|
| INSID3 单样本 | 55.0 | 59.4 |
| 朴素自训练（INSID3 多数票） | 57.0 | 59.0 |
| 不筛选的合并投票，一轮 | — | 61.4 |
| 现有方法（互相一致取前一半，4 轮） | 59.1 | 60.5 |
| 池子给真掩码 + 合并投票（上限，用了真值） | 67.7 | 68.6 |
| 池子给真掩码 + 闭式岭回归分类器（用了真值） | 67.4 | 69.5 |
| 伪掩码 + 岭回归分类器 | 56.4 | 61.7 |
| 去掉“与种子的相似度”一项：单样本 / 真掩码 / 伪掩码 | — | 52.0 / 66.6 / 58.9（原 59.4 / 68.6 / 61.4） |
| 标准题往返与池内真实值的秩相关：召回率 / 精确率 | 0.72 / 0.33 | 0.75 / 0.33 |

前 200 个和全部 400 个差得很多（收益 +1.1 对 +4.1）。**一折 200 个 episode 不能下结论**，以后每个结论至少四折各 400。
官方掩码对基线的影响没有固定方向：第 0 折 +1.09，第 1 折 −0.68（各 200 个）。

### 这些数排除了什么

1. **解码方式不是瓶颈。** 给了真掩码，最近邻投票和线性分类器都停在 67–69。去掉单种子那一项反而掉 2–7 点（查询掩码的精确率 0.83 → 0.75）。E4a 的 30 例结论相同。不再改聚合、不再换解码器。
2. **筛选不是机制。** 筛选相对不筛选的合并投票约 1 点（另一会话四折 400 样本：60.5 对 59.8），前 200 个上甚至是不筛选的更高。
3. **瓶颈是伪参考的质量。** 59.1 到 67.7 这 8.6 点全在这里。

### 为什么筛选只值 1 点

按可靠性筛选或加权（众包模型）成立的前提是各“标注者”的错误相互独立。这里所有伪参考出自同一个预测器、同一张参考图，错误有共同的原因：参考图没覆盖到的外观（漏检），和与目标一起出现的上下文（误报）。同一个概念下各张图犯的是同一类错，证据是“用伪参考反推参考图得到的召回率”与“池内伪掩码的真实召回率”秩相关 0.72。
互相一致性量的是自洽，而相关的错误恰恰自洽。所以筛图、加权、多轮迭代只能去掉个别离群的图，动不了系统偏差；自训练的不动点保留了预测器的偏差。

推论：剩下的 8.6 点只有两条路能拿，**估计并修正系统偏差**，或**引入不经过那张参考图的证据**。下面三个实验分别对应。

### 缺口具体在哪（从已有结果文件算出，`scripts/analyze_gap.py`，不用 GPU）

1. **收益几乎全部来自救回失败的 episode，同时在伤害本来就好的。** 第 0 折 400 个：单样本 IoU 低于 0.1 的 63 个 episode，单样本 0.02、现有方法 0.23、真掩码 0.41；单样本高于 0.7 的 187 个，0.84、0.81、0.82。也就是参考图多了以后，好的 episode 连给真掩码都不涨，用伪掩码还掉 3 点。
2. **缺口集中在少数类。** 四折 80 类（旧缓存）：上限空间 12.0 = 已拿到 6.1 + 缺口 5.8；缺口最大的 10 个类占了 49%：skateboard（单样本 16 / 现有 15 / 真掩码 75）、carrot、banana、microwave、truck、baseball glove、baseball bat、tennis racket、boat、dining table。22 个类是变差的，合计损失 0.8 点。
3. **没解决的是“分多了”，不是“分少了”。** 按池内伪掩码的精确率三等分，缺口是 +0.121、+0.034、+0.029；按召回率三等分是 +0.084、+0.025、+0.073，而且召回低的那一档现有方法已经拿到 +0.084。漏检这一侧现在的合并投票已经在修（person +17、sink +15、truck +10）；误报这一侧完全没动。
4. **误报是什么**：skateboard 的查询掩码精确率 0.14、召回 0.89，池子给真掩码后 0.78、0.87。分进来的是滑板上的人。缺口大的类多数是这一型：被人拿着、踩着、穿着的东西，或者总和某个背景一起出现的东西。
5. **标准题发现不了这种错**：skateboard 池内伪掩码真实精确率 0.46，用标准题往返估出来是 0.80。原因是参考图里常常没有那个人，往返回去自然看不出错。

### 真掩码到底提供了什么，伪掩码为什么给不了

池子给真掩码时，查询图里“滑板上的人”那些 patch 在每张参考图里的最近邻都是“人”，而且被标成背景，合并投票就把它否决了。**真掩码提供的是混淆物在原位置上的反例。**
伪掩码里人被标成了前景，15 张图都这么标，错误完全一致。所以筛图、加权、迭代都救不了；要的不是更可靠的正例，是反例。
只有一张标注时，反例只能从两处来：参考图自己的背景（参考图里刚好有那个混淆物时），或者**图集的结构**。

### M1（已否定，上限 54.2，保留作记录）：在图集上“先聚类、后贴标签”

想法来自半监督学习的聚类假设：无标注数据不提供标签，但提供密度结构；一张标注用来给“簇”贴标签，而不是给每个像素贴标签。

1. **图集级的外观模式**：把参考图、查询图、池内所有图的区域原型放在一起聚类（沿用 INSID3 自己的聚类方式和阈值），得到跨图的外观模式。这一步完全不用伪掩码。滑板和滑板上的人会落在不同的模式里，即使对某个“人”的 patch 来说，参考图里离它最近的是滑板。
2. **用参考图贴标签**：含参考图前景的模式是前景模式；含参考图背景的模式是背景模式；参考图里没出现过的模式是未知模式。
3. **未知模式用“可脱离性”判断**：统计这个模式的每次出现是否挨着（或同属一个物体节点）某个前景模式。
   - 经常脱离前景模式单独出现的（旁边站着的人、远处的水、别的柜子），是独立的东西，判为背景；
   - 几乎只挨着前景模式出现、且在多数含目标的图里都出现的（参考图里没拍到的腿），是目标的部件，判为前景。
4. **输出**是清洗后的池内掩码和每张图的否决区域，交给任意基础方法（INSID3、FoRIS）当参考，所以仍然是可叠加的。

它和已有做法的区别：朴素自训练和 TPA 式原型库只用高置信的正例；这里用的是图集的密度结构和**负证据**。混入不含目标的图以前只会添乱，在这里变成反例来源（一张图里没有任何前景模式，它的所有模式都是反例）。这也给了“为什么要一批图”一个比“多几张参考”更实在的理由。

按顺序检验四条假设，都在缓存特征上做，每条几分钟；前一条不过就停：

| | 假设 | 量什么 | 预测 | 不过则 |
|---|---|---|---|---|
| A1 | 图集级模式够纯 | 用真值给每个模式贴多数标签，得到的掩码 mIoU（上限）；模式纯度的分布 | ≥ 72（高于真掩码合并投票的 67.7），纯度呈两头分布 | < 66：这个粒度下模式不纯，换一次粒度；仍不行就弃 |
| A2 | 参考图能贴对标签 | 只用参考图贴标签（未知一律当背景）时，模式标签对真值的精确率和召回率；skateboard 的掩码精确率 | 精确率 ≥ 0.8；skateboard 从 0.14 升到 ≥ 0.6；召回会掉 | 精确率 < 0.7：参考图前景的 patch 太脏，改用参考图掩码内部的 patch（腐蚀一圈）再试一次 |
| A3 | 可脱离性能分开“混淆物”和“部件” | 在参考图没出现过的模式里，真前景模式和真背景模式的可脱离率，算 AUC | ≥ 0.75 | < 0.65：这个统计量没有信息，未知模式只能交给 H2 的打分器 |
| A4 | 合起来有净收益 | 清洗后的池内掩码当参考，四折各 400，对现有方法和不筛选合并投票的配对差 | +2 到 +3；缺口前 10 的类拿回三分之一以上；变差的类不超过现在的 22 个 | < +1.5：只留 A2 的否决部分，重新评估 |

风险先写明：部件级数据集（Pascal-Part、PACO）里目标是物体的一部分，物体的其余部分总挨着它；但它们在参考图里就是背景，第 2 步已经标成背景，不会走到第 3 步。目标很小（skateboard 平均 86 个 patch）时参考图前景的 patch 会混进背景，A2 专门量这一点。

### S1 顺手的一条：单样本已经好的不要动

第 1 点说明多参考会伤害本来就好的 episode（187 个里平均掉 3 点，四折合计 0.8 点）。先量上限：按真值“单样本 IoU ≥ 0.7 就保留单样本结果”能涨多少；再用无标注信号做门控（查询图自己的单样本掩码当参考去反推参考图的 IoU）。上限不到 +1 就不做。

### H1 偏差修正：估两个错误率，而不是一个可靠度

- **检验**：每个概念的漏检率可以从标准题估出来（秩相关 0.72）；漏检率高的概念改用“伪前景可信、掩码外当未知”的方式使用伪参考（`ImageSet.predict` 的 `known` 参数），其余概念保持现状。
- **预测**：旧数据里用真值补漏值 +4；全体改成“掩码外未知”时 person 52 → 76 而多数类变差。所以用真实召回率做门控的上限应有 +2 到 +3；换成标准题估计的召回率，拿到其中六七成，即 +1.5 到 +2。
- **符合**：方法的第一步定为“先估偏差再决定怎么用伪参考”。
- **不符合**：真实召回率门控都不到 +2，说明“掩码外未知”补不回漏检，弃掉这条，不调门限。
- **对照**：不门控的全体未知化；不筛选的合并投票。先做门控上限，再做标准题门控。误报率估不准（0.33），这条只管漏检一侧。

### H2 图集里到底有没有足够的证据（测量装置，最关键）

- **检验**：池内某个簇是否属于目标，除了“像不像参考图”，图集还提供五类证据：
  a 被其他图的伪前景匹配到的强度和次数；b 与标准题背景的匹配强度；c 在本图聚类树上与伪前景的距离（是否同一个物体节点）；
  d 这种外观在池内出现的频率，以及出现时是否总挨着伪前景；e 标准题估出的漏检率。
  用留出的类别训练一个小的有监督打分器（逻辑回归或梯度提升），输入 a–e，预测簇是否在真掩码内；用它修正池内伪掩码，再做合并投票，量查询 mIoU。它是测量装置，不是方法。
- **预测**：没有把握，这正是要量的。判据事先定好：拿回 8.6 点中的 4 点以上算证据足够。
- **符合**：从特征重要性里写规则；或者把这个小打分器本身作为方法的一部分（在基类上训练，对新类通用，不训练骨干）。
- **不符合（不到 2 点）**：图集除了“多几张伪参考”之外没有更多可用信息，方向封顶在 +4 左右，按 G1 重新评估：+4 的收益加朴素的机制够不上 solid accept。
- **注意**：COCO-20i 各折之间共享图像（`docs/reference/coco20i-image-overlap-audit-2026-10-02.json`），训练打分器要按图像去重。demo4 的单对图打分器只到 58–60，这里的新证据是 a、d、e 三项，报告时把它们单独消融。

### H3 上下文混淆能不能约掉（账本）

- **检验**：精确率低的概念（skateboard 把滑板上的人也分进来）误报的是共现的上下文，只有标准题的背景能否决它。量两件事：误报簇与标准题背景、前景各自的最大相似度；参考图里根本不含这个混淆物的 episode 占多少。
- **预测**：相当一部分 episode 的参考图里没有混淆物（标准题精确率的秩相关只有 0.33 就是这个迹象）。
- **符合**：这部分误差在只有一张标注时不可约，从目标里扣掉，论文里如实说明；精力全放在漏检一侧。
- **不符合**：标准题背景能否决大部分误报，那就加一条“匹配标准题背景强于匹配任何伪前景则否决”的规则，单独量。

### 顺序和判据

本会话执行接续（按最新版顺序）：四折主表复用已完成的 `probe_decoder_f0_400.json` 的1shot/naive/ours/true行，f1–f3各400建立official paired缓存，逐样本exact audit，完成主表后删除每折大缓存。预注册预测：跨折平均现有方法相对1shot约+4–6点、真掩码池相对1shot约+8–13点；若跨折/区间不支持，则不把f0+4.1推广，重新按G1判断。主表不给弱200例或历史单编码基线作替代。
编码期间实现M1 A1–A3的最小测量；先30例、同一固定gallery、raw/debiased特征空间和tau须明确冻结，mode真值只用于诊断。A1前记录每图区域oracle，避免把图集模式纯度与区域生成混为一谈。A2含FG和BG标签的冲突模式规则事先固定，不能事后挑解释。
H2依最新排序为M1 A3不足时的后备；训练/验证以类别和全部support/query/pool图片隔离，额外监督的收益只叫测量证据，简单相同监督控制同样训练。未读H2测试标签之前固定特征/容量/选择规则；拟合失败不能称信息不存在。
解释边界：目前两种读出和删因子失败不能证明所有解码都非瓶颈；相关错误可能降低共识判别力，但筛选不普遍要求错误独立。0.72往返召回相关不是“不同图错误相关”的直接估计。M1可脱离性、负证据与不可约性都是待测假设，不把8.6点当可到达信息或信息论上界。

1. 四折各 400 重建主表（E0/E1，官方掩码）。上表只有第 0 折。
2. **M1 的 A1 → A2 → A3**（同一份缓存，各几分钟，前一条不过就停）。A1–A3 都过再做 A4。
3. S1 的上限（几分钟）。
4. H2 的打分器作为后备测量：M1 的 A3 不过时，把“参考图前景占比、参考图背景占比、可脱离率、挨着前景模式出现的频率、模式大小”作为特征，量图集证据的上限。
5. H3 并入 M1 的 A2（参考图里有没有混淆物，A2 的统计里顺带得到）。H1 降级：漏检一侧现有合并投票已经在修，除非 A4 之后召回型的类仍有明显缺口，否则不做。
6. 判据：A4 相对“不筛选的合并投票”四折不到 +2，不扩数据集、不跑全量；M1 和 H2 都不过，按 G1 重估。

不再做的：改聚合或去掉种子项、换解码器、一致性筛选的新变体、用参考图校准分类器阈值（50.0，比不校准低 6 点）。

### 对论文的含义

- M1 通过：论文讲三件事。设定：一张标注加一批无标注图。解释：伪参考的错误相关，所以可靠性筛选无效，缺的是反例。方法：图集级聚类、用标注贴标签、用可脱离性找反例；混入无关图时反而更强。对照是朴素自训练、不筛选的合并投票、TPA 式原型库，底座是 INSID3 和 FoRIS。
- 只有 S1 或 H1 通过：总收益约 +5 到 +6，机制是门控，偏弱，要靠 FoRIS 和多数据集撑。
- 都不通过：停。
- 动手之前要查的前作（工作流 D）：cluster-then-label 类半监督方法、判别式聚类协同分割（Joulin 等 2010）、基于共现的无监督物体发现。要说清与它们的区别在于“一张掩码给模式贴标签 + 可脱离性当反例”。

## 2026-10-02 修订：核对相邻工作和官方掩码之后（其中的数字判断以上一节为准）

方向不变，变的是顺序和对照表。下面的 E0 是新增的；E1、E1b、E3 合并成一个问题来回答。

**相邻工作（原文已核对）**

| 工作 | 做什么 | 和本方法的关系 |
|---|---|---|
| TPA（arXiv 2608.08290，2026-08） | 开放词汇分割：从一批无标注的部署图里取高置信 patch，聚成类别原型库，与原模型的 logits 线性融合；免训练、单次、可叠加在任意模型上 | 任务不同（文本指定类别），但“无标注池 + 免训练 + 可叠加”的思路相同。**我们的朴素版本就是它搬到上下文分割上**，所以朴素版本不能当贡献，只能当对照 |
| TF-SSD（CVPR 2026） | 协同显著分割：一组相关图，没有参考掩码，SAM 候选 + DINO 跨图原型筛选 | 设定是“只有池子、没有参考图”。我们的设定在它和上下文分割之间 |
| PPNet（ECCV 2020）、不确定性半监督少样本分割（2022） | 元训练的少样本分割里用无标注图补充原型 | 需要训练、不是基础模型时代的方法；作为思路上的前作引用，并做一个同信息的对照 |

没有搜到“免训练的上下文分割 + 无标注同概念图集”的直接撞车工作（搜了两轮）。这个领域每月都有新稿（FROST 6 月、TPA 8 月、FoRIS 9 月），时间风险是真的。

**危险在哪**：不在“有人做过”，在我们自己的数字——机制相对“不筛选、直接多用伪参考”的增量还没立住。

- INSID3 上朴素 +3.3、机制 +6.1（旧缓存，其中多少来自合并投票和多轮、多少来自筛选没有拆开）。
- FoRIS 上朴素 63.3 反而高于往返筛选 61.8。
- 另一会话的 400 个开发样本诊断：可靠性筛选相对不筛选的合并投票，干净池约 +0.70，混合池约 +0.98，后者区间跨 0（见 E1b 的预测一行）。

也就是说，目前的收益主要来自“多用伪参考”，这正是 TPA 那一层的做法；真正算方法的那部分只值 1 点左右。

**调整后的顺序**

1. **E0 换成官方掩码**（新增，见下）。
2. **E1 + E1b + E3 回答同一个问题：机制值多少。** 同一份缓存上比：朴素自训练、便宜的原型库（E1b，TPA 式对照）、随机取一半、可靠性筛选；干净池和混合池（另一会话的 `open_pool_*`）都要，INSID3 和 FoRIS 都要。
   判据：机制相对最强的便宜对照不到 +2（区间跨 0 也算不到），就不扩数据集、不跑全量，把主要精力转到 E4——从伪掩码的两类错误里找机制，那里还有 6 点。
3. E2、E6 照旧。E7 只用已经在服务器上的小数据集：COCO-20i（官方掩码）、LVIS-92i、Pascal-Part、PACO-Part、SUIM。
   ISIC、iSAID 属于后期，G2 通过之后再下；PerMIS 不做（要 119 GB 的 TAO 测试集，每组也只有 3 帧）。

### E0 官方掩码

- **检验**：之前用的掩码是从实例标注重建的，把 crowd 区域也算成前景，官方掩码不算。逐（图，类）比较，两者 IoU 平均 0.954，6.3% 低于 0.9（`scripts/check_data.py --sets coco --coco-compare 0`）。这是否改变基线数字、能否解释本机 56.3 对论文 57.6。
- **预测**：官方掩码下 INSID3 单样本高 0.5–1.5 点；各方法之间的配对差基本不变。
- **符合**：之后所有缓存用官方掩码（`_paths.py` 的 `COCO_ANN` 已默认指向官方掩码）；旧表注明用的是重建掩码。
- **不符合**：与论文的差距另有原因（权重副本、episode 顺序），单独查。
- **命令**：`scripts/cache_episodes.py --fold f --n 200 --check 200`，分别用默认掩码和 `DEMO9_COCO_ANN=/root/demo4_cache/data/COCO2014/annotations`，看打印的 released mIoU。

### 同信息对照

TPA 式的原型库对照见下面的 E1b（另一会话写的，含适配限制，照它执行）。另外两个对照：PPNet 式（无标注图特征补充部件原型）在同一份缓存上实现，预计不高于原型库；
TF-SSD（只有池子、没有参考图）需要 SAM 权重，先不跑，论文里作为设定对比说明。

## 通用约定

服务器代码 `/root/autodl-tmp/demo9`，依赖 `/root/autodl-tmp/demo4`（编码器封装、episode 列表）和 `/root/demo4_cache`（环境、权重、COCO 掩码，只读）。
特征缓存放 `/root/demo9_cache`，**跑完即删**。

```bash
cd /root/autodl-tmp/demo9
export PYTHONPATH=/root/demo4_cache/env DEMO4_GPU_FRAC=0.3
P=/root/miniconda3/bin/python
$P scripts/cache_episodes.py --fold 0 --n 400        # 约 10 分钟，3.6 GB；末尾会打印“缓存版与官方实现的 mIoU”，必须相等
$P scripts/run_episodes.py --file /root/demo9_cache/episodes_f0_n400.pt --out results/e1_f0.json
$P scripts/stats.py results/e1_f*.json --base 1shot --vs agree:tophalf
rm /root/demo9_cache/episodes_f0_n400.pt
```

- 每个结果表都带三行对照：`1shot`（INSID3）、`naive`（朴素自训练）、`true`（池子给真掩码，上限）。
- 结论只用 `stats.py` 的配对区间（在每折内对 episode 重采样）；一折 100 个 episode 的区间约 ±5，不能下结论。
- 每跑 50 个 episode 会打印一行。前 200 个 episode 已经说明问题时就停，不必跑完。
- E1 前先运行 `scripts/audit_paired_interface.py`：用 10 个 episode 逐像素检查准备好的成对缓存与未修改官方输出，要求 exact mask，而不是平均 mIoU 相近或 mask IoU > .999。失败时只检查已有样本的编码、归一化、投影、压缩接口，不进入四折方法矩阵。烟测缓存当轮删除。
- 随机/前半/后半选择必须匹配实际保留数，分数并列时固定 image-ID 排序。当前 `choose` 用中位数包含并列，可能保留多于一半；未经这个控制不能把差异全部归因于排序。

## E1 可靠性规则到底有没有信息（先做）

- **检验**：伪参考图的“互相一致”分数是否反映它的质量；按可靠性加权是否不差于“取前一半”。同时用成对编码的新缓存重建主结果表（旧表的缓存是单张编码，单样本与官方实现不完全一致）。
- **预测**：池子 15 张时，`agree:tophalf` − `agree:randhalf` ≥ 2 点；`agree:bottomhalf` 低于 `randhalf`；`agree:tophalf:w`、`spectral:tophalf`、`gold+agree:tophalf` 与 `agree:tophalf` 相差 1 点以内；`agree:tophalf` 相对 `1shot` 为 +5 到 +7。
- **符合**：规则定为其中最简单、不带阈值的一个（优先带“标准题”的 `gold+agree`，理由见 E3），进入 E3。
- **不符合**：
  - 若前一半与随机一半没有差别：收益不来自可靠性估计，而来自合并投票和参考图数量。论文的贡献点改为聚合规则，可靠性一节删掉。
  - 若相对 `1shot` 的收益不到 +3 或区间跨 0：旧结果依赖单张编码的特征，先查 `1shot` 两种编码下的逐 episode 差异，再决定是否继续。
- **命令**：上面的通用命令，默认 `--variants` 即这组对照。4 折各 400 个 episode，每折缓存 10 分钟 + 评测约 30 分钟。先只跑第 0 折前 200 个（`--limit 200`）。

## E1b TPA 近邻：跨图验证是否优于便宜的原型库扩充（必做）

用户指定近邻：[TPA, 2026-08-08](https://arxiv.org/abs/2608.08290)，已核读正文 §3.2–3.4 和 Appendix A。
它从无标注池的高置信度自身预测构建冻结 DINO 类别原型，以余弦分数与底座 log-probability 融合。
因此“免训练使用无标注池”“不改底座”“可叠加多个方法”不再作为 demo9 的独立创新点。
文本类别驱动 OVSS 与单张标注参考图的设置差异是真实差异，但不自动构成方法贡献。

- **检验**：跨图验证及保留空间参考的聚合，是否携带单次原型库扩充未保留的有效证据，且值得额外开销。
- **预测**：原型库应能取回部分新增图像收益；先验不支持宣称筛选有大额独立增益。已有另一配置的 400-query 开发诊断中 clean AG 对无筛选 pooled 仅约 +0.70 点，mixed 约 +0.98，后者区间跨 0。预注册探索预测为 AG−bank 在 0–2 点之间；这不是通过门槛或论文结果。
- **符合**：冻结规则后，在全部四折、同池的强底座上证明相对最强便宜控制的配对增益，或在相同整流程预算下改善质量—成本曲线，才继续扩大机制；争取至少 +2 点独立收益，不能仅引用相对 1shot 的增益。
- **不符合**：若便宜原型库追平/超过，或额外开销不能兑现任务收益，撤回跨图筛选/参考聚合的独立价值主张；停止扩大该 N² 规则，回到伪参考错误账本寻找仍缺失的证据。区间跨 0 视为未建立优势，不能包装成正结果。

**对照合同（先开发冻结，再四折验证）**：相同底座、DINO 特征、support/query/donor IDs、池大小、1 张标注、伪掩码初值、评分方式及 pool seed。
每表包含 1shot、底座原生 naive 多参考、无筛选 pooled 同轮数、便宜 FG/BG 原型库、随机等数量参考、可靠性筛选、给池真掩码的 diagnostic 上限。
分开消融“patch 筛选”“image 筛选”“保留空间参考与单原型压缩”“单轮与反复更新”，并报告实际选中数，不能混成一个增益。
记录编码次数、bank 构建/互相验证/最终预测开销、峰值显存、缓存大小和整 episode 时间；warm decoder 时间不代表总成本。

**适配限制**：原版 TPA 要求底座 dense probabilities；INSID3 接口给硬掩码，不能把 0/1 当校准后验。
原版阈值 `2/C` 在二分类为 1，会令锚点集合为空，不能据此判 TPA 失败。
先实现无置信度门控的 FG/BG bank 作为明确命名的 *ICS prototype-bank control*；若使用连续前景/背景分数选择锚点或融合，须记录其来源、尺度和固定策略，并在已有开发证据上冻结，不能用评测 GT 调参数。
原文的 argmax+confidence、minimum anchors=5、sum-to-unit-prototype、cosine/log-probability 融合列为原版合同；binary threshold、视觉 support 代替文本、复用 DINOv3、host score 适配分别列出偏离。
适配版只能叫 TPA-style 控制，不能声称复现了原文 OVSS 数字；仅同信息下实测比较，不跨协议排名。若 FoRIS 没有合法后验接口，同样不伪造校准概率。

**执行顺序**：先完成 E1 的 exact-interface 小核查；通过后在同一缓存上跑 E1 与冻结 cheap-bank 控制，先 50–200 个 episode 看方向，再由结果决定四折；不重复下载模型/数据，不另建方向目录。

首个实现是 `scripts/prototype_bank_eval.py`：一次汇总同一 1shot 伪掩码，FG/BG 各一个单位原型，分别比较 support-only、pool-only、support+pool；patch 余弦分类与固定 INSID3 cluster 内余弦均值分类均报告，不挑最好行。
每图每类少于 5 个 anchors 不入 bank；缺任一类回退原 1shot；不使用 query 本身建自己的 bank。推理对象的 query/donor GT 置零，诊断 true 和评分只在合法预测冻结之后执行。
它复用已有 debiased 特征，不是原版外部 DINOv2 / confidence / log-probability fusion。
这一步只测“单次压成 FG/BG centroid”能取回多少信息；若它弱，不足以否定完整 TPA-style 适配，也不再堆阈值补救。

首个 30-episode 运行卡：检验 pool bank 相对 support bank 的信息收益；探索预测为 pool bank 位于 1shot 与 naive 之间（未验证）；若接近 naive，冻结该便宜对照进入 E1b；若低于 1shot，停止扩大这个读出，先核查输出与原型污染，TPA 原版结论保持未验证。
命令（只读复用正在建立的 paired cache，不复制大特征、不删除其他执行者的 cache）：
```bash
cd /root/autodl-tmp/demo9_transductive_ics
PYTHONPATH=/root/demo4_cache/env DEMO4_GPU_FRAC=.3 /root/miniconda3/bin/python scripts/prototype_bank_eval.py \
  --file /root/demo9_cache/reb/episodes_f0_n200.pt --limit 30 --M 15 --out results/prototype_bank_pilot_f0.json
```
这是单折小 pilot；实际 pool count 随 200-episode gallery 的可用同类图数记录，不把它冒称固定 15。独立筛选优势仍需全部四折和冻结强适配对照。

**首测决定（已完成，不扩）**：30 episodes / fold0 / pool seed0 / 重建掩码 / paired cache，1shot 60.872、naive 56.787、support+pool bank patch 48.977 / cluster 49.686；pool 相对 support-only 有增益，但仍低于完整 1shot。cluster−1shot 为 −11.186 点，episode-paired 95% CI [−16.4,+0.7]，小样本且照片可复用，不作为最终泛化判断。
停止扩大这个 confidence-free centroid readout；**不能拿这个弱对照宣称超过 TPA**。完整原文式 confidence/host-output fusion 在当前 ICS 接口下尚未建立公平适配，仍是 E1b 待解决项。结果 `results/prototype_bank_pilot_f0.json`；五项 CPU 数值检查通过；未生成/复制大特征、未删他人的 shared cache。该预试不改变其他执行者正在进行的 E0/E1。

## E3 池子里混入不含目标概念的图

- **检验**：可靠性过滤能否挡住干扰图。现实里一批图不会每张都含目标。
- **预测**：`--M 15 --D 5` 时收益保住三分之二以上，被当作参考图的干扰图不到 15%；`--D 15`（一半是干扰图）时保住一半以上。
- **符合**：把“允许多少比例的干扰图”写成方法的适用条件。
- **不符合**：冒烟测试里（30 个 episode、池子 4+4）`agree:tophalf` 掉到单样本以下，`roundtrip` 没掉。若大规模下仍如此，说明“集合内部一致”在干扰图占多数时失效，规则必须锚定带标注的参考图（`gold+agree` 或 `roundtrip`），并回到 E1 重新比较。
- **命令**：`$P scripts/run_episodes.py --file ... --D 5 --variants agree:tophalf,gold+agree:tophalf,roundtrip:thr:0.5,agree:tophalf:w --out results/e3_D5_f0.json`，再跑 `--D 15`。输出里有“干扰图被当作参考图的比例”。

## E2 换基础方法（FoRIS，黑盒）

- **检验**：转导步骤是否与基础方法无关；可靠性过滤在 FoRIS 上是否有用。
- **预测**：`naive` 相对 `1shot` +3 到 +5（已有 4 折各 60 个 episode：+4.2）；`agree` 不差于 `naive`。
- **符合**：论文可以写“可叠加在任意单样本方法上”，主表给 INSID3 和 FoRIS 两列。
- **不符合**：若 `agree`、`roundtrip` 都不如 `naive`，说明 FoRIS 自己的多参考图规则已经抗噪，过滤只对投票式聚合有用；论文里如实写成“对 FoRIS 用最简单的版本”。
- **命令**：`DEMO4_GPU_FRAC=0.45 $P scripts/run_blackbox.py --fold 0 --limit 100 --M 7 --trust roundtrip,agree --out results/e2_foris_f0.json`。每个 episode 约 20–40 秒，一折 100 个约 1 小时；这是最慢的实验，先跑第 0 折前 30 个看方向。FoRIS 源码用的是另一个会话的只读副本 `/root/autodl-tmp/demo8_local_verification/foris_source`，不要改。

## E6 相当于多少张标注

- **检验**：1 张标注 + 15 张无标注图，相当于几张标注。
- **预测**：落在“多 2 张标注”和“多 4 张标注”之间。
- **符合 / 不符合**：这是论文摘要里那句话的依据，实测是多少就写多少。
- **命令**：E1 的命令加 `--diag`，表里会多出 `label+1`、`label+2`、`label+4`（池子里的图给真掩码，用 INSID3 原有的多参考图规则）。

## E4 伪掩码的两类错误（主要的开放问题）

### E4a 用户提出的单种子瓶颈：先做固定证据干预

源码证实 INSID3 区域分数是 cross × intra-to-one-seed × candidate-area；FoRIS 原型 prior 也含单种子相似度，但后面还做归一化、score boost 和 disagreement correction，不能据此推断其完整结果必然受害。
检验：同特征、参考、伪掩码、candidate、seed及其 area boost、原 .2 阈值，仅移除 intra-to-seed 一项。分别量1shot、native naive multi-ref、unfiltered pooled，不重跑互相筛图矩阵。
预测：若它是足以支撑新主线的重要瓶颈，其仅补回被该项压制的 GT 前景像素之宽松上限应 ≥5 点；直接移除约束的新增 TP 和 FP 同报，不预称有无标签收益。
符合：检查新增区域是否已有合法局部跨图支持，进入部件分别取证/背景反证；上限不是模型可识别的信息。
不符合：若连宽松上限都不足5点，或新证据未被这一项压制，不扩大这个构造；不调门槛/改oracle口径救主张。原版 TPA 强控制仍保留。
成本/命令：先10个随机CPU接口案例核对重建分数，随后只读现存official paired f0n400缓存、前30个标准episodes，`scripts/seed_bottleneck.py --file /root/demo9_cache/probe/episodes_f0_n400.pt --limit 30 --out results/seed_bottleneck_pilot_f0.json`，显存上限.3，无新cache/下载。
oracle行是输出空间逐像素只加真前景的宽松界，既不是合法方法，也不是可实现的整区域选择；原native输出重建要求每例exact，失败先修接口。FoRIS需另做完整pipeline干预才能转移结论。

首30例 INSID3 干预完成：1shot 62.035→去intra49.694（GT-pixel界66.913，+4.878）；同P1池 pooled57.595→54.905（GT-pixel界59.869，+2.273，episode CI[+.150,+3.746]）。新增像素中TP比例为1shot21.6%、pooled34.3%；pooled被放开30个区域中9个GT前景占比≥.8。可证存在受压制真前景，但该固定阈值/候选下不是已证实的大主线，更不是可用选择信息。
不扩大 INSID3 去约束构造，不调整.2阈值。单独检查强FoRIS的整个下游，避免把两套结构视为等价：前10个相同标准episode，原生FP32 encoder及整个FoRIS，仅prior源码中 `intra_sim` 换成1，后续minmax/boost/disagreement/reweighting不动；每例native replay exact，features只在当例RAM复用。
预测：若该机制在强系统上仍是主要损失，GT-only补回宽松界≥5点；若不足5点/完全无输出变化，则不扩大这一prior消融。无GT增益不作预先承诺；只去factor为朴素同信息控制，GT行仅诊断。命令 `scripts/foris_seed_bottleneck.py --limit 10 --out results/foris_seed_bottleneck_pilot_f0.json`，.45显存上限，无新cache/下载；先10例是否值得30例，不排全量矩阵。

强FoRIS10例已完成，native replay10/10 exact：native56.768，去intra56.989，差+.221点、episode-paired CI[−.8,+1.2]；该干预新增像素只加GT前景的界57.769（+1.001，CI[+.1,+1.7]）。未达到主瓶颈预测，不扩大这项FoRIS prior单因子消融；这是小样本开发结论，不否定所有部件组成方案。当前镜像 `models/foris.py` SHA256 `982cf4aa91ac90370151c353efb37240d7d4d5c35a0dfe9cc9509cd554c85fa9`；代码默认seed boost .25，不能等同论文未加系数的示意公式。
另执行者 `probe_seed_f0_200.json` 已完成相邻INSID3删因子/coverage/local实验，不重跑其矩阵。其local把全池FG极大值与仅gold图BG极大值比较，并给所有query clusters追加标签，不限于原candidate：与“保持候选、只消除单seed惩罚”是不同干预。它的低分不能单独否定局部取证；max搜索预算不对称与candidate变化需要分开解释，不能靠随意阈值修补。
还有一个源码约束：同一参考中 max-FG-sim > max-BG-sim 等价于全图最近邻落在FG（并列除外），已经是原backward投票的二值信息。新局部机制必须证明可靠性/证据来源/组合作用的额外信息；不能仅把这张票改名“背景反证”。FoRIS §3.1 本已有多FG原型和困难BG对比，也必须作为机制强控制。

从 61.7 到上限 67.5 的 6 点在这里。已知（第 0 折流式设置，`../demo4_incontext_seg/scripts/stream_diag.py`）：

| 类型 | 例子 | 第 1 轮精确率 / 召回率 | 修掉哪种错误能到上限 |
|---|---|---|---|
| 召回低 | person、hotdog、chair | 0.94 / 0.54 | 补上漏掉的前景 |
| 精确率低 | skateboard、backpack | 0.42 / 0.95 | 去掉误报（个别图整张分错） |

第一步是在标准 episode 上重新量一次两类错误各值多少：E1 的命令加 `--diag`，看 `pseudo-noFP`、`pseudo-noFN` 相对 `pseudo-all` 和 `true` 的位置。

已经试过、不要重复的规则（第 0 折流式设置下都在 66 左右，与不加这些规则的 66.4 持平）：区域判断与带标注参考图的 patch 判断一致才信；跨图共识阈值（跨折不稳）；
伪掩码之外的区域一律当“未知”（person 52 → 76，多数类变差）；交叉预测投票直接当输出（63.5）；纯标签传播（漂移）。

之后每次只测一条规则，要求：四折里“涨超过 2 点的类数 / 跌超过 2 点的类数”不变差，且两类错误中至少一类的差距缩小。
没有新的可测机制之前，不在这里堆规则。

## E5 定稿规模

规则定下来后：每折 1000 个 episode（缓存 8.6 GB / 折，一折一折做），池子抽样 3 个种子（`--pool-seed`），`stats.py` 出最终表。约 4 小时。

## E7 其他数据集（等用户提供文件）

INSID3 的 9 个基准里目前只有 COCO-20i，且掩码是从官方标注重建的。需要用户从 Google Drive 下载后传到服务器（链接见 INSID3 仓库 `docs/data.md`）：
COCO-20i 官方掩码、LVIS-92i、PASCAL-Part、PACO-Part、FSS-1000，以及跨域的医学 / 遥感 / 水下数据。
拿到后的工作：给 `cache_episodes.py` 加数据集开关（复用 INSID3 仓库 `datasets/` 里的 episode 抽取），其余脚本不用改。
部件数据集的“同概念图”指含同一部件类别的图。

## 进度门槛

| 门槛 | 内容 | 不通过怎么办 |
|---|---|---|
| G1 | E1、E3：成对编码缓存下收益 ≥ +4 且区间不跨 0；有一条对干扰图稳健的规则；E1b 对最强便宜同信息控制建立独立收益/整流程成本价值 | 收益 < +3：停，回到 demo4 的账本重新选问题；bank 追平则撤回筛选/聚合贡献，不扩大 N² 构造 |
| G2 | E2、E6：FoRIS 上四折为正；标注成本等价关系 | 只在 INSID3 上成立：降级为 INSID3 的改进，重新评估是否够投稿 |
| G3 | E7：至少 4 个数据集上为正 | 只有 COCO 成立：不投主会 |
| G4 | E5 + 论文初稿 | — |
