# 当前本机执行

当前用户明确要求换数据集，MEAN单方法200例：LVIS十折各20，200个不同类别和查询照片，排除已知旧查询；运行在`cv_data/a/lvis_mean200_20261008/`，参数与PACO保持相同，完成后只为LVIS单独评分。
LVIS200已完成并评分46.263（CLI1024 class mIoU），源码、参数与模型身份和PACO一致；进程76319已正常退出，未安排其他批次。

当前用户明确追加第三组PACO-Part MEAN200例，每折50，查询照片与前400例无重叠；运行在`cv_data/a/paco_mean200_batch3_20261008/`。完成后按已有逐例I/U合并600例，算法参数保持相同，只落实本组追加200。
第三组已完成并评分50.763；三组合并600例为44.942，合并时编码0次、不重开query mask，并由独立逐类求和核对。进程73791已正常退出，未安排第四组。

第二批已由用户明确要求：PACO-Part MEAN单方法200例，四折各50，查询照片与首批无重叠；运行在`cv_data/a/paco_mean200_new_20261008/`，复用首批O24缓存。只安排这一批200；关于五批的统计问题不自动扩展运行规模。
第二批已完成、统一评分43.961；两批按逐例I/U合并400例为43.710，合并编码器调用0且不重开query mask。运行70481已正常退出，未安排第三批。

当前用户明确要求PACO-Part只测MEAN，200例（四折各50），运行目录`cv_data/a/paco_mean200_20261008/`；完成后评分，不自动扩大。使用已冻结seed1前缀，不按分数选样；保存所需O24原始FP32特征供复用，不运行RCG/fine对照。
该200例已完成、封存及评分：CLI1024 mIoU44.543，进程66862已正常退出。预测、原始O24缓存、源码及分数均保留，未安排后续大批量。

用户最新要求覆盖旧运行安排：先约1000例以内初筛，未经小规模涨分证据不直接运行6000或全量。
开发控制器24973和worker24974已退出；954条保留，COCO952、PASCAL/PACO各1条。同级USER_STOPPED.json保护重启，所有下方“6000活动”描述仅作历史。
Pro提供参考方法和计划，运行规模由用户任务决定；不将Pro的2000/集作为必须执行的数量。当前直接复用COCO952已有预测做探索评分，不新增编码，不称五集或正式验证。

用户最新要求优先：官方全量已停止，不重启、不新增。30717/30718已退出，533条COCO记录及全部产物保留。
`cv_data/a/official_foris_20261008/USER_STOPPED.json`是停止标记；控制器主动拒绝自动重启。下方官方活动描述仅为历史。
6000例dev子集PID24974现在也已停止，不自动恢复。未来全量先满足成熟开发领先及已实现/验证的可复用原始DINO缓存，约400GB容量应通过范围和去重设计处理。

原始FP32缓存的五集首例完整读回检查通过，含R/Q的O24、Q/K16/24及四个移位Q的O24；FoRIS/RCG/fine/MEAN连续场、原图与CLI掩码逐位一致，CPU编码器调用0。
实际报告：`cv_data/a/raw_feature_cache_complete5_20261008_v2/report.json`。C1可用`--raw-cache-profile <profile.json> --device cpu`读取，缺特征即失败；实际候选仍受表示前提门控。
这仅完成五例实现检查，完整清单所需缓存与候选领先尚未完成，不满足新增全量的启动条件。

600例探索已完成并评分：`cv_data/a/explore600_20261008_v1/run`；结果见其上一级的`SUMMARY.md`和`scores.csv`。
每集120例，seed0，复用已就绪资产；两个进程均退出0，不重复启动或改变这批已封存的统计口径。
后续按Pro最新完整修订版推进seed1开发2000/集、seed2验证2000/集和seed0官方全长；数据不足只等待对应任务。

唯一目标是同条件最终 mIoU 的净涨分。INSID3/FoRIS 的弱点是候选来源；共享上下文等候选能否涨分未知。局部错误修正或机制检查通过，不单独构成成功；是否采用依据完整分数及配对收益。
以 `docs/research/PLAN.md` 为当前任务定义，状态、协议、资产和候选登记保存在本目录。只在 `codex_m4` 上工作。

阶段0正在进行：官方源码与服务器核心源码已核同；COCO/PASCAL-Part/LVIS/PACO-Part 的固定首例完整基线已封存。
这四例是实现检查，不是完整官方复现或确认结论。五集原图池及SUIM整池均已下载并全清单SHA通过，完整官方复现仍未完成。

运行使用 `aidemo` 的 Python，所有大文件留在 `cv_data` 或忽略的 `outputs/m4`。例如：

```bash
python scripts/frozen_dino_plan.py prepare \
  --datasets coco,lvis,pascal_part,paco_part,suim --smoke \
  --out outputs/m4/official_five_smoke_protocol_v1
python scripts/frozen_dino_plan.py evaluate \
  --manifest outputs/m4/official_five_smoke_protocol_v1/manifest.json \
  --datasets coco,lvis,pascal_part,paco_part,suim --split smoke \
  --out outputs/m4/official_five_smoke_B0_v1
```

`--smoke` 固定每集首折的第一例；不因结果或文件缺失改抽样。正式 `prepare` 不加 `--smoke`，
并要求原图池逐文件SHA receipt通过，按官方长度保留全部合法重复。旧`--split official`评分目前只公布完整FoRIS，
B0预测封存但暂不计分；该旧入口行为不作为新seed1/2验证方案，不继续做官方减开发的确认审计。

`evaluate` 推断阶段只加载参考RGB、参考二值mask、查询RGB；原图主读数与1024 CLI读数均保存。
全部预测封存后，评分阶段才读取查询mask。也可用 `--no-score` 推断后，再单独调用 `score --out <run>`。
`--resume` 核对输入与实际源哈希，恢复已提交的逐例结果；已封存运行沿用归档实现，不重新编码或评分。
进程排他锁防止同一运行重复启动；进程是否存活应查实际PID/工具句柄，不能仅凭状态文件判断。

当前入口已支持完整基线、新seed1/2清单和查询照片分组统计；旧开发包原类别映射已完成，候选入口尚待实现。
`prepare --split dev`固定seed1，`prepare --split val`固定seed2，每集2000例；COCO/PASCAL/PACO每折500、LVIS每折200、SUIM无折2000。
五集两个集合均已固定，开发与验证各10000例；新集合主区间为10,000次查询照片分组，旧600结果不回写。
分支位置基、真实提取检查和前提测量产物在`cv_data/a/representation_premise_20261008/`。
25例推断已完成，未读query标签；一个完整父流程与带提取运行的预测/字段逐数组相同。
活动6000例推断在`.../dev6000`，实际PID24974、工具会话25177；先查实际活性及`activity.json`/`infer_dev6000.log`，不重复启动。
另一个活动任务为`cv_data/a/official_foris_20261008/`，官方FoRIS控制器会话25937，当前COCO PID30718。
该控制器逐数据集封存与双口径评分；49,318例官方清单已准备完成。进度读同目录`activity.json`和当前数据集日志，不能仅凭旧PID判断后续数据集停止。
两个任务共享MPS/CPU；当前耗时有并行影响，不作独立效率结论。官方FoRIS是允许提前读取的基线，候选官方尚未开分。
使用`--representation-basis .../branch_basis --no-score`；推断代码及src源文件保持不变，避免未完成运行恢复时混入不同实现。
新单基线入口为`run_insid3_baseline.py --baseline insid3|foris`；多运行配对与双帧统计使用`score_sealed_baselines.py`，不能合并不同输入或不完整运行。
前提margin采用实际FoRIS stage2二值FG、源20%困难BG、等权均值，并在配置明确记录RCG没有该BG规则；不能写成未经核实的B0面积权重。
完整开发前提为COCO/PASCAL/PACO各2000例；封存全部字段后，运行`score_representation_premise.py`。选择半扫描，复核半只读选择，不能根据复核换表示。
旧官方减开发的确认方案已被Pro完整修订版取代；不继续扩展旧重叠审计。现有`confirm`入口仍禁用，不能冒充新验证入口。
Pro最新回复已完整取得（8305字符），19:50 UTC再次读取无更新；来源为`outputs/m4/pro_context/latest.json`。

检查命令：

```bash
python scripts/check_segmentation_metrics.py
python scripts/check_m4_protocol.py
```

统计主汇总是类内累计I/U、类均值、折均值。600探索保持100,000次(fold,class)内配对episode重采样；
新候选主区间改为10,000次(fold,class)内查询照片成组配对重采样，episode区间仅作附表。
两者都不能写成未见照片总体保证。gross四类增删与净变化分开。
一次只改一个环节；现有SAFR开发组合已关闭，没有新候选被确认，也没有达到最终五集目标。
