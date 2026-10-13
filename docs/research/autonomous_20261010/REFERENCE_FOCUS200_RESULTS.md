# 实际参考放大200例：匹配新信息有局部收益，完整方法未胜出

2026-10-10，`codex_m4`。固定候选 `actual.equal.crf` 未超过同批完整 FoRIS、MEAN 或已有快区域；此版不采用，不按数据集或事后最佳臂选择读出。全200预测、六连续场、direct／原CRF、CLI1024／原图均已封存后评分。

|CLI1024，同批规定口径|Deep道路100，累计I/U|PACO部件100，87观察fold/class|
|---|---:|---:|
|完整FoRIS|29.377542|45.629834|
|MEAN|26.465369|46.824561|
|既有快区域|37.344028|40.914271|
|实际focus，固定equal＋CRF|25.619896|39.056805|
|插值focus，匹配equal＋CRF|26.979236|35.977374|
|实际focus，whole query＋CRF|23.079959|41.163913|
|插值focus，whole query＋CRF|23.865591|39.782990|
|实际focus，native4 query＋CRF|26.884548|36.003119|
|插值focus，native4 query＋CRF|28.339876|31.911821|

实际focus对匹配插值的equal终点差为Deep−1.359340、PACO+3.079431点。PACO丢失全部目标的例数从插值21降至实际11，但FoRIS只有4；对FoRIS候选删掉4,359,261 TP同时删5,800,192 FP，另加563,857 TP／696,737 FP。Deep对插值额外增加175,895 TP和1,479,906 FP，删除32,581 TP和306,268 FP；对FoRIS新增6,460,888 FP。因此几何纯度提高、恢复部分漏检及匹配PACO正差都不能替代完整胜出。

PACO原图固定actual.equal为39.008701，插值36.060131，FoRIS45.563915；完整原图表及PACO固定303槽位另存。Deep输入本来为1024，两个frame相同。本次都属于已有曝光开发样本；PACO多数fold/class只有一张查询照片，原评分的类内照片bootstrap条件区间不能解释成独立跨照片泛化。Deep公开原始道路抽样协议缺失，本100为明确自定义探索，不能与论文13.3混减。

## 同一监督、不同实际观测

只用合法reference mask选择一个16对齐512focus，规则及源码在评分前冻结。保留whole reference上下文；参考像素不擦除。实际focus用真实RGB重新经DINOv3-L编码；匹配控制从whole-R raw64裁32并插值64，使用完全相同细mask、physical-area权重、参考token采样ID与角色损失。whole/native4查询实际输入与旧200相同，七个raw输入逐例绑定。共同APD分支只使用whole R/Q与reference mask。

前置mask检查使用Torch nearest canonicalization。最初PIL-nearest审计对PACO产生18个不同box，producer在编码前拦截并完成纠正；原审计保留为历史，正式输入只绑定修正后的 `d809c7c…47f1`。实际法不是occupancy估计，零点也未被证明是可迁移后验阈值；这仍是普通参考二角色分类的信息实验，不能据此宣称原创贡献。

## 实际验证与成本

两真实episode（Deep0、PACO100）由独立代理从冻结代码、七实际raw输入和lawfulmask重放：六signedFP32场逐位相同，48个完整mask零不同像素，无queryGT／旧预测读取，无原封存目录写入。41个live与frozen外部源文件及native CRF二进制身份核验通过。全200独立复核也已完成：4,800候选I/U、1,200基线I/U、9,600编辑、6,000mask几何、400GT身份、200物理质量／quadrature、2,263SHA、1,400raw元数据绑定和400逐例两帧账本全部一致；24项配对区间差≤1e−10。raw实际tensor重读由producer／runner及两例独立重放证明；全200账本复核没有另冒称重读全部payload。

独立排序证据进一步限制解释：PACO actual.equal的whole／FoRIS-FG／BG宏AUC .903303／.744748／.872473，均略低于derived .905327／.749004／.879841；source physical risk也由.136170恶化到.142125。PACO正差主要是恢复范围变化，不能归给更好的目标身份。Deep source risk .330153→.324028、FoRIS-FG AUC .734403→.740592略改善，但新增FP远大于新增TP。native/global的域差异仍在：Deep FG local .744492胜global .681433，PACO global .761059胜local .707250。没有匹配的whole-only新臂，不能把这组对照扩大解释成加入focus相对无focus的单因素收益。完整证据见 [reference_focus_evidence.md](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/reference_focus_evidence.md)。

192个新focus O24、4cache hit，196唯一输入、96次MPS B2；实际编码334.109秒，producer总365.476秒。四CPU进程、各两线程，六臂缓存推断366.676秒；单例七raw读取／核验均.577736秒，六头共.331731秒，固定actual.equal CRF均.949647秒。此吞吐包含全部匹配控制及产物保存，原六输入编码已缓存；尚未测量七视图冷输入端到端时延。

产物：[report.json](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/reference_focus_head200_20261010/report.json)、[seal](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/reference_focus_head200_20261010/sealed.json)、[config](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/reference_focus_head200_20261010/config.json)、[逐例I/U](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/reference_focus_head200_20261010/scored_episodes.jsonl)、[编码收据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/reference_focus200_20261010/COMPLETE.json)、[独立真实重放](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/scene_model_validation/reference_focus_actual_replay.json)。方法／实验规格：[REFERENCE_FOCUS_EXPERIMENT.md](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/REFERENCE_FOCUS_EXPERIMENT.md)。

自主goal继续active；十数据集超越、可靠泛化与原创贡献均未达成。
