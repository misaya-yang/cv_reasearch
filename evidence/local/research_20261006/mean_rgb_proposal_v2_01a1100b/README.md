# RGB局部proposal读出v2：600例事后重放已完成

**完整实测class-summed mIoU为63.007428。** 相对v1全cut提高+0.103477 [0.080924,0.177966]；相对原MEAN +0.075882 [−0.006268,0.135967]。修订改善了v1的读出副作用，尚未建立优于MEAN的稳定优势；完成这一固定修订后不扩展变体搜索。

## 固定定义与完整控制

`D = Hcut1024 XOR Hsame128unary1024`；`Hfinal1024 = where(D,Hcut1024,HoriginalMEAN1024)`。D外逐像素保持原MEAN，D内采纳原RGBcut；没有新的权重、温度、lambda、阈值或图选择。最终mask是局部替换后的binary1024，**不声称仍是原128 Potts目标的全局最优解**。cut=unary（zero graph）时D为空，精确返回当前MEAN。

模块`src/ics/methods/mean_rgb_proposal.py`提供direct predict与纯boolean compose；predict复用原v1完整组件，输出1024binary field及1024/原尺寸mask。原mean_rgb_potts、M4、shared backend、PLAN和旧outputs未改动。独立revision increment=0，不计原创机制。

| 完整行 | class mIoU | v2减该行，配对95%区间 |
|---|---:|---|
| original MEAN | 62.931546 | +0.075882 [−0.006268,0.135967] |
| full RGB cut v1 | 62.903951 | +0.103477 [0.080924,0.177966] |
| same128 unary | 62.769442 | +0.237987 [0.173127,0.352218] |
| RCG | 63.088942 | −0.081513 [−0.428154,0.355455] |
| fine16 | 63.464527 | −0.457099 [−0.895428,−0.075406] |
| fine64 | 63.681184 | −0.673756 [−1.479707,0.751583] |

这些新数值由完整新mask实际计数，不是把旧+.134509与−.162104相加。相对fine16仍显著较低；不宣称超过最强完整读出。

## 暴露、源绑定和封存

同source public600：600 occurrences、80类、570个原照片连通组。class-summed计数，配对区间沿用项目2000次RandomState(0)原照片连通组bootstrap。该方案在看过源600 aggregate质量之后明确指定，因此是 **posthoc exploratory replay、非独立确认**。

源目录为`/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/next_candidates_public600_v1`。先核source seal中的config/inference/evaluation manifest哈希，再核全600 receipt、PRED、field与评测packet字节哈希；只hash packet字节，不解GT。派生阶段仅解三个预测键：mean.control、mean_rgb_potts、mean_rgb_unary.control。全部新600 mask/field/receipt封存之后，才用单独score命令打开query GT。

源fullcut、sameunary、originalMean三控制完整保留，且评分精确复现源行值。新seal有600条receipt摘要。新远端目录：`/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/mean_rgb_proposal_local_v2/replay600`；代码在同独立namespace的`code/`。没有修改旧source目录。

本地`server_replay600/`保存新config/seal/source_validation/两个manifest与完整score、counts、bootstrap。`summary.json`记录源与新seal SHA256、源码SHA256及精确数值。完整600预测/field/逐例receipt保留在上述服务器新目录；服务器执行与评分已经完成。本次Git记录包含config、inference manifest、seal、source validation及完整聚合报告；1.28MB evaluation manifest、逐例metrics、counts与bootstrap draws保留本地。

## 完整检查和成本

`check_readout.py`与`check.json`：8个boolean输入组合核验D外MEAN identity及zero-graph identity；实际RGB128固定组件的完整正/负合成例分别添加2280/删除2280像素，1024field renderer XOR=0，原尺寸75×109返回完整输出。600派生compose均核验D外MEAN精确保留。相对原MEAN共改840,993像素，这是改动量，不是正确性收益。

派生使用2CPU；源字节核验0.313735秒，derive16.949927秒，峰值RSS62,136,320bytes（约59MiB，低于4GiB）。没有encoder、MEAN或RGB组件重算。

**保留原producer成本**：旧整个cached source pipeline墙钟478.560846秒；逐例MEAN+RGB串行成本和1170.961380秒；源全部选定行逐例串行成本和1905.829201秒（含absorption/gaussian等其他行，不等于本控制单独成本）。派生16.95秒不能替代旧478.56秒生产成本。report中的generic inference_wall_seconds仅为本次派生计时，另外显式保存inherited_full_pipeline_seconds等继承成本。缓存最初DINO producer成本未重新测量。

## 唯一执行入口

`scripts/replay_mean_rgb_proposal.py derive --source <sealed-public600-source> --out <fresh-output>`过滤GT、核验并封存；之后单独`score --out <same-output>`。入口只有source/out绑定，不提供参数搜索选项。结果已完成，无需重复600例或另开variant。
