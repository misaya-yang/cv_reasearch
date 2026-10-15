# 下一步：新照片留出收尾与论文对照表（2026-10-13 C）

接 `NEXT_EXPERIMENT_PLAN_20261013B.md`。B 的决策表第一行已经触发：E1 中 UFraw+、UF+、MFU 三臂在 S 半十域不低于完整 FoRIS，E2 按预定规则选中 MFU，T 半十域点估计全正（均值 +3.16 pp，七域区间不含 0；COCO +1.13 [−0.06, +2.14]、SUIM +1.02 [−0.33, +2.41]、Road +0.97 [−4.61, +4.64]）。数字已由 `evidence/research/x1013_E2_T_iu.jsonl.gz` 独立复算一致。B 规定的下一步是“对未曝光数据做确认”。

x1013 的 13,613 例都是已曝光开发缓存，T 半只是其中未参与选择的一半，不是新照片。真正的确认已经在 `x1012s_tasks/PHOTO_HOLDOUT_PLAN.md` 中冻结：3,211 个任务，参考与查询原照片都与登记的开发曝光零交集，主方法正是 MFU。它在 2,209/3,211 处停止（`predict.log` 最后一条 2,200，05:55），没有错误记录，当前无进程。

## 其他已关闭的路线（本轮）

- 预测面积门控的组合（`ADAPTIVE_FOLLOWUP.md` S1–S3，`FIXED_AREA_COMPOSITIONS.md` R1–R3）：没有一臂同时拉平 COCO、ISIC 并保住 Lung。与 W09、W10 一致：按面积选尺度不可行，两种尺度的场直接融合（MFU）可行。
- 像素修正与 CRF 的拓扑主张（`MECHANISM_DIAGNOSTIC_RESULTS_20261013.md`）：Fundus 的 IoU 增益来自覆盖率，clDice 无可靠提升；Road 上两者都降低中心线质量。论文不写拓扑。
- 相对几何、左右身份（`MECHANISM_EXPERIMENT_COMPLETION_20261013.md`）：标签不合格，停止。
- T6 角色对称：S-self 不满足入选条件。
- 等面积排序“十域全优”的主张：T 半 iSAID −0.02，按 B 的规则撤回，改为“无一域显著更低，LVIS、PASCAL、PACO、Lung、Fundus 显著更高”。

## H1 完成新照片留出（必须）

1. 续跑（已完成的任务按 `records/` 跳过，约 1,000 例，按此前 5.4 秒/例约 1.5 小时）：

   ```bash
   PY=/Users/misaya.yanghejazfs.com.au/miniconda3/envs/aidemo/bin/python
   $PY -B scripts/x1012s/photo_holdout.py predict
   ```

2. 确认 `sealed.json` 生成、`n=3211` 后评分一次：

   ```bash
   $PY -B scripts/x1012s/photo_holdout.py score
   ```

3. 评分前不看任何 query 标注，评分后不改公式、不换臂、不补样本。

**评分前写定的读法。** 主比较不变：`main − U`（已在 PHOTO_HOLDOUT_PLAN 中冻结）。另外把 `main − FoRIS` 的逐域与覆盖域等权均值作为论文主主张的确认，读法如下（区间取 `scores/report.json` 中同一 2,000 次照片 bootstrap）：

| 结果 | 结论 |
|---|---|
| 覆盖域全部点估计 ≥0，且等权均值区间下界 >0 | 主主张在新照片上成立：“十个基准均不低于 FoRIS，均值显著更高”，并列出各域区间 |
| 有域点估计 <0，但该域区间含 0 | 改写为“无一域显著低于 FoRIS，k 域显著更高”，如实列出该域 |
| 有域区间上界 <0 | 新照片上十域主张不成立；该域另作诊断（只用 S 半与已曝光数据），留出集不再用于任何选择 |

Fundus 只有 16 例，Road 162、SUIM 180，这三域区间会宽，结论只按区间写，不合并其他数据补量。

## H2 T 半的消融表（可选，只用于论文，不用于选择）

E1 已为全部 13,613 例存了各臂掩码。用 E2 的同一评分函数在 T 半给出 `candidate、C1、M、M+、UFraw、UFraw+、UF+、Ugm+、MFU` 相对完整 FoRIS 的逐域差与面积三档表。

- 目的：在未参与选择的一半上说明两条线各自的缺口（候选缺大目标、MEAN 缺细结构）以及融合后三档都为正。
- 预先写定：无论 T 上哪一臂更好，主方法保持 MFU；若某单线臂在 T 上也十域不低，只记录，不提升为主方法（提升需要新的确认集）。

## 不做的事

- 不再提出新臂，不再做面积门控、拓扑、几何或像素修正的变体。
- 不重新跑 DINOv3 于已曝光数据。
- 留出评分后，不用它的逐域结果调整任何定义。

## 交回

- `../cv_data/a/x1013_photo_holdout/scores/report.json` 与一页短报告 `docs/research/x1012s_tasks/PHOTO_HOLDOUT_RESULT.md`：覆盖域逐域五臂绝对分、`main − U`、`main − FoRIS`、`main − MEAN`、`main − G_mfu` 及区间，参考复用数。
- 紧凑逐例证据复制到 `evidence/research/photo_holdout_iu.json.gz`。
- H2 若执行：表追加到 `x1013_tasks/E2_T_REPORT.md` 末尾，注明“选择冻结后的描述性消融”。

## H3 官方协议主表（H1 之后执行）

**动机。** 到目前为止，MFU 的全部证据都来自我们自己抽样的任务池：x1013 的 13,613 例和新照片留出。FoRIS 论文表 1 用的是官方发布的 seed-0 加载器，指标为每个 fold 内按类池化 I/U、类均值，再对 fold 求均值。论文主表必须在完全相同的官方任务上，给出 MFU 与完整 FoRIS 的配对结果，用发布代码的 CLI-1024 坐标。方法已冻结；官方任务与开发池有重叠，这是基准测量，不是泛化检验，泛化由 H1 回答。

**做法。** `scripts/x1012/w12_official.py` 原样复用新照片留出的五臂封装（main=MFU、U、FoRIS、MEAN、G_mfu），不改公式和参数。清单：

- COCO、LVIS、PASCAL-Part、PACO-Part、SUIM：`official_s0_manifest_20261008`，共 49,318 例；
- ISIC：`extended_benchmarks_seed0_20261010_v3`，600 例；
- iSAID：用冻结的扩展准备脚本，按官方 seed-0 重新生成三个 fold 全长，共 10,063 例，输出到 `isaid_seed0_full_20261013`；
- Chest：600 例；Fundus：200 例；
- DeepGlobe road 没有官方协议，沿用本地协议，另行标注。

运行顺序从小数据集到大数据集：Fundus、ISIC、Lung、COCO、SUIM、PASCAL、PACO、iSAID、LVIS。每个数据集全部预测封存后单独评分，评分不影响其余数据集的预测。

**读法（评分前写定）。**

- 每个数据集给出五臂的官方 mIoU、`main − FoRIS` 及按 query 照片 1,000 次 bootstrap 的区间。同时列出 FoRIS 论文发表值，用来核对我们重现的 FoRIS 是否对得上。
- 重现的 FoRIS 与发表值相差超过 1.5 pp：先查协议差异，再写主表。
- MFU 在某个官方数据集上低于 FoRIS：如实写进主表，不改方法。
