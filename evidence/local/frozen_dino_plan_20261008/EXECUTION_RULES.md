# 当前本机执行

用户最新指令优先：先完成已经启动的600例探索，不继续局部反复审查，也不等待官方原图池全齐。
活动运行：`cv_data/a/explore600_20261008_v1/run`；日志/选样/缺失项位于其上一级目录。每集120例，seed0，复用已就绪资产。
当前进程PID1885、工具会话30537；恢复前先确认活性，不重复启动。以下官方全量工作留待探索汇总后决定。

目标是同条件最终 mIoU 涨分。INSID3/FoRIS 的弱点是候选来源；共享上下文等候选没有预先成立的收益。
以 `docs/research/PLAN.md` 为当前任务定义，状态、协议、资产和候选登记保存在本目录。只在 `codex_m4` 上工作。

阶段0正在进行：官方源码与服务器核心源码已核同；COCO/PASCAL-Part/LVIS/PACO-Part 的固定首例完整基线已封存。
这四例是实现检查，不是完整官方复现或确认结论。五集原图池及SUIM整池/首例仍待下载校验。

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
并要求原图池逐文件SHA receipt通过，按官方长度保留全部合法重复。`--split official` 的评分只公布完整FoRIS；
同时生成并封存的B0预测暂不计分，留给去掉开发episode后的单次B0确认，避免提前读取B0确认成绩。

`evaluate` 推断阶段只加载参考RGB、参考二值mask、查询RGB；原图主读数与1024 CLI读数均保存。
全部预测封存后，评分阶段才读取查询mask。也可用 `--no-score` 推断后，再单独调用 `score --out <run>`。
`--resume` 核对输入与实际源哈希，恢复已提交的逐例结果；已封存运行沿用归档实现，不重新编码或评分。
进程排他锁防止同一运行重复启动；进程是否存活应查实际PID/工具句柄，不能仅凭状态文件判断。

当前入口已支持完整基线与统计核心；五集开发清单的原类别映射、官方/开发重叠审计及候选入口尚未完成。
确认执行目前明确禁用，待清单与暴露账本绑定后再开放。Pro最新长回复的末尾尚未取得，不补写缺失规则。

检查命令：

```bash
python scripts/check_segmentation_metrics.py
python scripts/check_m4_protocol.py
```

统计主汇总是类内累计I/U、类均值、折均值。候选比较用100,000次(fold,class)内配对episode重采样；
它描述给定照片池的抽样不确定性，不能写成未见照片总体保证。gross四类增删与净变化分开。
一次只改一个环节；现有SAFR开发组合已关闭，没有新候选被确认，也没有达到最终五集目标。
