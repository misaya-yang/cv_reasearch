# demo9 下一步实验计划

2026-10-02 写。代码都已写好并在服务器上用 30–100 个 episode 跑通过（冒烟测试，不是结论）。
每个实验先写四行（检验什么 / 预测 / 符合则 / 不符合则），再给命令、耗时和停止规则。按顺序做，一次只跑一个。

## 2026-10-02 修订：核对相邻工作和官方掩码之后

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
