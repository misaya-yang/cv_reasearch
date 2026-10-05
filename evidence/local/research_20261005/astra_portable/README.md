# 固定候选：完整源码与可复核证据

本包从原缓存重建 RCG、完整掩码假设及 source/scalar 选择，再生成保留的自适应预算删除候选和必要控制。推断不依赖任何旧运行目录，不访问 query GT、native 或 pre。所有指定 episode 的预测全部封存后，才进入评分。CPU 单线程，无模型下载、RGB、训练或自动参数搜索。

这是已暴露 DEV220 上的探索性候选证据，不是完成的论文结论。保留的 `RCG_count_matched_delete` 原本是实验中的同预算控制，后因结果被选中继续核查；不能改称事先声明的新方法主臂。

## 当前结果与解释边界

全部为1024工作栅格，同类先累计 I/U，再作类平均。区间用2000次 RandomState(0) 照片连通组重采样。

| 固定输出 | 旧120 | 新100（现已暴露） | 合并220 |
|---|---:|---:|---:|
| native 缓存最终 CRF | 62.0176 | 59.7641 | 58.8299 |
| RCG | 63.7230 | 60.9558 | 60.8714 |
| MEAN_CONTROL | 63.5001 | 60.8164 | 60.5595 |
| 保留候选 | 64.0449 | 61.2986 | 61.5811 |
| 跨折固定比例、同域同排序 | 64.1650 | 60.5334 | 61.0986 |
| 同 K、全 RCG 域最低分删除 | 63.6628 | 60.1379 | 60.4006 |

合并候选相对 native 为 +2.7512，95%区间[1.1021,3.8103]；四折分别 +1.7697/+2.4103/+4.0771/+2.7663。旧120/新100分别 +2.0273/+1.5345。目标是整体稳定增益，不要求每一折或每批都超过2点，区间跨0也不是自动淘汰条件。

相对 RCG 的增量 +0.7097[-0.7049,1.3341]，相对 MEAN +1.0217[-0.5310,1.5035]，相对固定比例 +0.4825[-0.5216,1.0986]。这些增量仍有不确定性和集中性：删去任一类/例，native 增益最差仍 +2.0978；但删去 class58 后，相对 RCG 只剩 +0.1608、相对固定比例 +0.0148。该类仅含 `2_59_58`。这些删除是诊断压力测试，绝不成为部署筛例规则。

同 K 全局删最低 RCG 分的控制比候选低1.1805点，区间[-1.6619,-0.0501]（控制减候选）。这支持当前输出中假设限制域提供定位价值，但仍集中：去掉 class58 后，候选对全局控制只剩 +0.2758；去掉最有利的5类约为 −0.0026。全局控制的 K 仍来自原限制域，因此此对照只隔离删除位置，不证明可丢掉限制域来计算预算。

没有证明新的 BG 像素排序。候选最终按已有 RCG 分数排序；BG只给每例删除数量。正确参考信息、候选相关性或 GT oracle 都不能自动证明可读出的语义身份规律。GT、RGB、预训练表示中的信息与此算法实际使用的关系要分开。

## 原始输入布局

每个 `--root` 指向解压后含以下子目录的根：

```text
ROOT/
  cache/evidence_v1/feat/<fold>_<e>_<c>.pt
  results/extent_v1/run/packets/<fold>_<e>_<c>.npz
```

PT：`q,r` 都为4096×1024，安全读取 `torch.load(weights_only=True,map_location='cpu')`；`debiased` 仅可选元数据，不再去偏。NPZ通过 `allow_pickle=False` 读取。

- 所有基础组件需要 `cov,score`，均为64×64。cov是参考 mask 的面积覆盖，score是既有 FoRIS 连续输出。
- **复现已报告的候选预算默认还需要 `fg_max,bg_max`**，各4096值。这是原 packet 中的 Part1 source cosine 最大值，角色分界 cov≥0.5 / <0.5。它们不是 query GT，也不是最终 FoRIS score。
- 若只有 q/r/cov/score，显式指定 `--bg-evidence recompute`，从存储后的 FP16 q/r 转float32计算 q@r，不额外单位化。该方式可运行，但并非原 packet 的数值精确替代，见后面的真实漂移检查。默认不静默回退。
- `truth`、可选 `native`、可选 `pre` 仅评分读取，uint8 np.packbits 的1024²二值图。预测阶段不需要这些字段。
- JSON/CSV manifest需要 `fold,e,c,support,query`；支持原完整 manifest。`manifest_available220.json`附当前220例及旧/新 batch。相同key和照片身份去重，冲突或缺例报错。
- 可重复传 `--root`；不会复制大特征，不使用 manifest 中旧机器的绝对特征路径。同名文件在多个根出现时必须字节一致。

## 运行

依赖仅 Python、NumPy、SciPy、scikit-learn、CPU PyTorch。实测版本在 `environment_linux_cpu.lock.txt`，不是自动安装脚本。

```bash
python rerun.py all --root ROOT --manifest manifest_available220.json \
  --out NEW_RUN220 --expected 220
```

多个分包直接追加 `--root PART02 --root PART03 ...`。当前只用已有220，缺的21不需要补交。若以后已有完整241缓存，可用其 manifest，默认 expected=241；不允许少例静默降级。

严格分开两个阶段：

```bash
python rerun.py predict --root ROOT --manifest MANIFEST --out NEW_RUN --expected 220
python rerun.py score --root ROOT --manifest MANIFEST --out NEW_RUN --expected 220
```

`all`也严格先完成全部预测。保持验证盲态时只用 predict。输出目录必须是新目录；已评分报告拒绝覆盖。评分默认基线 native_cached_CRF；无native时明确传 `--baseline RCG`，不伪称有native对照。pre可缺省，有时验证score复原pre的逐像素差。

只有最小输入时：

```bash
python rerun.py predict --root ROOT --manifest MANIFEST --out APPROX_RUN \
  --expected 220 --bg-evidence recompute
```

`--audit-feature-evidence`在packet模式额外计算一次完整 q@r，记录两种 maxima 的最大误差、K差和删除mask差；不会改用审计结果。完整220重跑的耗时和机器漂移应实测，不能由小smoke线性外推。

## 原组件与候选的固定定义

RCG：alpha=.5、cross-k10、query互20近邻图、lambda16、fidelity下界.1、CG rtol1e−7/atol1e−9/maxiter300。MEAN_CONTROL：alpha=.25，同图同λ；两者并非只改变guide。source/scalar从原eligible seed枚举完整hypothesis，同cached INSID3的average-linkage cosine .4及cross×intra×occupancy>.2。全部源码在包内，不假称官方双表示/RGB CRF复现。

R=RCG，S=source_contrast_mean，T=scalar_contrast，D=R&(S|T)，E=R&~D。

1. 原packet bg_max−fg_max先float64相减，再float32双线性放大到1024，align_corners=False。
2. K为E内该差严格大于0的像素数。没有query标签参与。
3. `RCG_count_matched_delete`：仅在E内按放大后的RCG连续分从低到高删除恰好K个。
4. `global_RCG_same_budget`：K不变，改为在整个R内删最低分K个。定位控制，未改预算来源。
5. `crossfit_fraction_RCG_rank`：对每个测试折，从其他三折计算sum(K)/sum(|E|)，排除所有与测试折共照片连通组的训练例；再删floor(fraction*|E|+.5)个E内最低RCG分。此参数只用无标签预算，不用GT。比例在**传入的完整cohort**上计算，所以5例smoke的比例不同于220报告比例；改变cohort后不能冒称固定比例输出完全相同。
6. 附原 `explicit_BG_delete` 与 `FG_count_matched_delete`（同K，E内最低fg_max）作为最初控制。空E自然K=0；空hypothesis bank原default/source/scalar均为空，按真实规则得到E=R，不添加特殊救援分支。

所有排序ties用稳定排序、原图row-major索引。各臂只删R，不增加前景。仍保留原6个OR/AND和三路5臂，供完整对照；不按新GT选最佳输出。参考角色全FG/全BG仍沿原锁定实现报错，不制造标签相关fallback。

## 封存与统计

推断先逐例生成组件/预算，再用这些无GT预算计算跨折比例并补齐固定比例控制，最后写完整freeze；在此之前不评分。

输出 `frozen/<key>.npz`（packed masks、RCG/MEAN字段、实际采用maxima）、`freeze.json`（全部输入/输出/源码SHA、逐例审计）、`crossfit_parameters.json`。评分后输出逐例整数 I/U、四象限及相对RCG增删TP/FP的 `episodes.jsonl`，和报告。

同类先ΣI/ΣU再macro；2000次RandomState(0)照片连通组有放回抽样，保持抽样重数后重新聚合，各次未出现的类省略。按数值fold/e/c顺序建组。全220为76类/218照片组，不是两批指标的加权平均。含batch时同时分批报告，始终给per-fold和episode升降。

## 已实际验证的重跑范围

本次新增5例从原PT/packet、clean cwd完整重跑：`0_11_0,1_4_57,2_59_58,3_56_43,3_57_67`。包含class58集中病例、缺目标病例、空bank病例。预测19.76秒，含额外一次完整q@r精度审计；CPU/BLAS/Torch线程均1。5例全部先封存，再评分。它们只是执行/数值一致性检查，不是新准确率结论。

- 已有参考可对比的RCG、source/scalar/default、D、原BGgate、FG同K、保留候选、全局同K，5例逐像素差均0；有参考的RCG连续字段最大差0。
- 跨折比例helper由原220无标签预算重算，所有分母/分子/训练身份/比例精确一致；将这些原220比例作用于新重算的5例，固定比例mask也逐像素差0。5例自身的cohort比例没有伪装成220比例。
- packet与FP16缓存重算maxima最大误差6.7592e−5。五例K分别差+2/+3/+2/0/+1，保留候选相应差2/3/2/0/1像素。原BG直接gate可能差更多（最多7像素），说明即使最大浮点误差很小，也不能声称二值决策全同。
- 另做q/r/cov/score最小输入predict-only检查，packet没有truth/native/pre/fg_max/bg_max。结果在 `evidence/minimal_input_check.json`，明确为recompute近似路径。
- 尚未用这个最终整理入口重新完整跑220或241；既有220数字来自原完整研究运行，附整数I/U可独立重算。不能把5例parity扩大为任意机器/全cohort逐位保证。

检查小重跑：

```bash
python rerun.py all --root ROOT --manifest smoke_manifest5.json --out SMOKE5 \
  --expected 5 --smoke --audit-feature-evidence
python check_candidate_smoke.py --run SMOKE5 --out smoke_check.json
```

只重算已交付证据，不读特征/GT：

```bash
python analyze_evidence.py --out evidence_recomputed.json
```

## 完整源码与证据

入口 `rerun.py` 必须与 `components.py,rcg_readout.py,hypothesis_source_contrast.py,adaptive_deletion.py,statistics_randomstate.py` 同目录。其余验证/分析源码 `check_candidate_smoke.py,analyze_evidence.py` 也完整提供。没有任何运行时对旧结果目录的依赖。

`evidence/scored220_compact.jsonl`含全部核心对照逐例整数I/U与照片身份；`evidence/candidate_reassessment.json`、`global_same_budget_report.json`、`fixed_fraction_report.json`、`original_composition_report.json`保留精确原报告。`localization_sensitivity.json`及重算脚本给集中性。`original_BG_signal_report.json`保留未证明新BG排序的证据。原报告中的绝对路径仅是历史来源文字，不被预测入口使用。

`delivery_manifest.json`给源码与证据SHA。原研究预测没有改写。不要只转入口而遗漏同目录Python依赖。
