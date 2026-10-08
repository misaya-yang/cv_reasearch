# 当前本机执行

600例探索已完成并评分：`cv_data/a/explore600_20261008_v1/run`；结果见其上一级的`SUMMARY.md`和`scores.csv`。
每集120例，seed0，复用已就绪资产；两个进程均退出0，不重复启动或改变这批已封存的统计口径。
后续按Pro最新完整修订版推进seed1开发2000/集、seed2验证2000/集和seed0官方全长；数据不足只等待对应任务。

目标是同条件最终 mIoU 涨分。INSID3/FoRIS 的弱点是候选来源；共享上下文等候选没有预先成立的收益。
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
COCO两个集合均已固定；缺少输入的数据集不补抽或计入部分结果，其余就绪数据集继续。新集合主区间为10,000次查询照片分组，旧600结果不回写。
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
