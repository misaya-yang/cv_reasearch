# Pro M5固定smoke4：完整封存与统一评分（2026-10-06）

**四例已全部完成，四份代码/输入身份统一封存后独立1CPU评分；M5原尺寸class-summed mIoU=15.8059，同renderer MEAN=35.6952，差−19.8893 pp。** 工作1024按原packet.truth评分：M5=15.7953，MEAN=35.6846，差−19.8892 pp。全部四例的69个ROI在四层β=1误差均为0，数值问题已解决；当前负结果来自完整算法的输出行为，不能再归因于该query尾块bug。

当前权威证据：[统一seal与逐例评分JSON](actual_smoke4_v1.json)。首例报告与历史失败保留，不将第一例IoU0作为总体价值判断；本报告使用固定完整四例`0_0_72,1_0_73,2_0_74,3_0_75`，与M1/M4同清单。所有source hashes、individual receipts、packet identities、原GT映射及原gate已写入统一seal；方法SHA统一为`b9da3fd3b42c08a27c3634d86251b36e75269ff5414105d8408566dabe4c1653`。

## 完整比较

| 固定行 | work1024 mIoU | 原尺寸mIoU | 原尺寸Δ vs MEAN |
|---|---:|---:|---:|
|MEAN/zero|35.6846|35.6952|0|
|M5-extrapolate|15.7953|15.8059|−19.8893|
|endpoint|24.8716|24.8509|−10.8443|
|hard-channel|16.1141|16.1170|−19.5782|
|native-region|31.0231|31.0149|−4.6803|
|response-objectness|0|0|−35.6952|
|mass-objectness|0|0|−35.6952|

外推在该四例上未胜endpoint、hard或native-region，也未胜宿主MEAN。原尺寸四例合计：add-TP=3,633、add-FP=187,806、delete-TP=53、delete-FP=41,249。新增误检远多于新增真目标，解释了IoU损失的主要方向；不以首例的误检删除量包装收益。两个objectness诊断四例均为空输出，亦没有分割增益。

这是已暴露public/dev4的有限机制屏；四例都非独立确认，也不是全部600例的质量结果。它支持“当前冻结M5 v0在这份清单明显失败，额外外推必要性没有得到支持”，不支持“所有局部消息路径都不可能有效”。不自动开启β/温度/层数/限幅扫描或新的机制。

## 计算与资源

case0复用原完整seal，不重编码/重推。其余三例按用户新授权3worker×4CPU并行，process RSS上限8GiB、group上限16GiB；实际group峰值9,124,597,760 bytes≈8.50GiB。三worker全部exit0，group完成244.30s。没有GPU、下载、安装、Git或24/600扩展。

| case | paired capture | 全部branch推断 | 当前入口时间 | 入口范围 |
|---|---:|---:|---:|---|
|0_0_72|52.76s（既有）|132.57s|144.73s|cached恢复，含加载；不含本次encoder|
|1_0_73|47.80s|163.27s|213.79s|fresh capture至完整输出，另列模型冷加载|
|2_0_74|48.32s|180.76s|231.80s|fresh capture至完整输出，另列模型冷加载|
|3_0_75|47.16s|169.68s|219.58s|fresh capture至完整输出，另列模型冷加载|

时间包括主法、β1审核与hard等研究控制，不是单独主分支部署成本；不把Pro算术0.833图像等效写成真实秒数。三例fresh入口不含外部MEAN宿主重建，明确使用已经存在的cachehost输入；不是独立RGB到分割的全部费用。统一评分0.2477s，独立1CPU。

全部69×4层严格审核为0，仍保持原atol/rtol=5e−5；4N query/N reference覆盖、所有固定控制及两级renderer都执行，没有ROI高分过滤、候选缩减或β1 shortcut。四例均从实际row读取原`debiased=True`，basis/权重/source与首例相同，未重开gate。

## GT及协议边界

work1024的GT严格来自各自`packet.truth`，不以新近邻GT替换历史工作空间指标。case0此前单例检查使用原annotation近邻1024 GT；在本四例统一评分中改按用户指定packet.truth重新评分，case0没有重推。原尺寸GT从统一seal中的固定annotation路径、class foreground value及image H/W映射读取，全部四例存在且shape匹配；annotation hashes在评分记录中列出。

新三个case的推断没有读取query GT；所有四例统一seal/individual预测哈希验证之后才打开packet.truth与原GT。case0已有单例评分的暴露历史明确保留，这批数据不是新holdout。

宿主仍为`existing_FP16_conditional_cache_MEAN_graph_float64`，新M5分支为fresh FP32 native/fixed Pi。新的reference完整mask按nearest1024→area64产生coverage，不默称旧producer逐值相同；每例与旧coverage最大差在各binding中记录。这些身份限制没有消失，也没有把成绩搬到Pro纯FP32宿主或历史其他renderer表中。

## 可核对远端目录

```text
/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/pro_message_audit_v1/
  code_b9da3fd3b42c/                         # 四例共同方法代码
  outputs_v3_cached/                        # case0原seal/fields/masks
  smoke4_fixed_v1/case1/episode_01_1_0_73/
  smoke4_fixed_v1/case2/episode_02_2_0_74/
  smoke4_fixed_v1/case3/episode_03_3_0_75/
  smoke4_fixed_v1/merged_sealed.json          # 统一四例source/预测封存
  smoke4_fixed_v1/score_smoke4.json           # exact packet.truth + 原GT
  smoke4_fixed_v1/group_state.json           # group峰值及全部exit0
```

任务已完成：固定smoke4全输出、全审核、统一封存和独立评分。当前没有M5活动进程，也未消费/测试新的M1联合capture API；该扩展仍不属于本次质量结论。后续由根代理据本负结果和完整成本收束当前v0边界，本报告不产生自动研究待办。
