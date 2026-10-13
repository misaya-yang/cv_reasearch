# 在另一台机器接续研究

本分支携带研究记录、代码与小型数值证据。数据集和模型需要时正常下载、准备，大文件不进Git。

## 先了解研究并复算已有结果

先读[研究导读](RESEARCH_GUIDE.md)，再读[STATUS](STATUS.md)、[RESULTS](RESULTS.md)、[当前PLAN](PLAN.md)。导读连接了原始假设、干预与控制、实际结果、下一步为何改变；保留的历史研究记录是证据，不是待启动任务。

无需本机归档、cv_data、GPU、torch或网络，可用Python标准库复算已保存C1与B1核心表：

```bash
python -B scripts/read_research_results.py
```

它只读 `evidence/research/*.jsonl.gz` 的整数I/U，不读取图像、GT或历史绝对路径。输出沿用observed (fold,class)槽均值/CLI1024，不把开发统计变成官方主表，也不重算置信区间。

原报告的便携内容在 `evidence/research/primary_reports.json`；关键理论机制数值与固定方法规格分别在 `mechanism_reports.json`、`method_specs.json`。源SHA用于身份，Mac绝对路径仅是历史位置；这些JSON的内嵌内容无需那个位置存在即可阅读。

## 实验数据、模型与依赖的准备

约定目录：

```text
paper_project/
  cv_reasearch/                 # Git：代码、研究记录、精简证据
  cv_data/                     # 按需下载、准备的数据与模型，不进Git
    demo4_cache/               # 基础数据/模型路径
    datasets/ics/              # 扩展数据、标注与协议数据
    native_assets/             # 原位置基等固定资产
    third_party/               # 对应版本FoRIS/INSID3/CRF源码
    runtime/                   # 平台相关实现；Mac .so不能直接当Linux二进制
    a/                         # manifests、raw缓存、封存字段与实验输出
```

|工作|需要的外部资产|
|---|---|
|读论文问题、理论、结果及源码|无；本仓库内足够|
|复算携带的C1/B1表|无；只用紧凑I/U|
|查看全部案例Mask、重算更细机制|对应实验的prediction/records/字段，按报告定位|
|接续R1/R2/R3|P0 explore/fields、B4固定场与预测、R1/R2/R3冻结配置/预测/收据、输入RGB/合法Mask及其依赖|
|重放完整FoRIS/MEAN或新缓存实验|对应raw O24与entry/profile、原图/Mask、数据几何、basis及FoRIS/CRF运行时|
|从RGB开始编码|上述资产加DINO权重和匹配的编码环境|

按所做实验下载、准备数据与模型，也可复用已有资产。缓存可从匹配的原图、权重和配置重新生成。复用已有缓存时，输入/crop/权重/层/精度需与其manifest相符。

默认资产根为仓库同级的 `../cv_data`：DINOv3权重在 `demo4_cache/models/dinov3-vitl16-timm/`，基础COCO数据在 `demo4_cache/data/COCO2014/`，扩展数据在 `datasets/ics/`，固定位置基在 `native_assets/positional_basis.pt`，FoRIS/INSID3/CRF源码在 `third_party/`。具体依赖按上表的目标实验选取。

## 路径与运行环境

`lab.py` 的仓库根和 `cv_data` 根按相对位置定位，但已有manifest的 `payload_path`、Mask路径和 `source_sha256` 键仍可能包含生成机器的绝对路径。使用这类已有缓存时，按本机实际位置适配路径。

复用已有封存结果时，保留原始manifest与seal；路径变化可用显式映射或派生manifest处理，不能修改封存文件后仍称原SHA。新运行写新的 `cv_data/a/<run>/`，不覆盖当前会话输出。CPU/MPS插值的边界差异已有实测记录，重放须以原封存输出和对应实现为准。

环境版本见[本机环境快照](../../evidence/research/environment_snapshot.json)，它是已观察环境，不是Linux可直接安装的锁文件。轻量结果复算无这些包依赖；完整M4与CUDA入口及平台差异见[scripts说明](../../scripts/README.md)。

## 分支是否已经带走这轮工作

工作目录的文件与已提交分支是两回事。这份研究整理通过单独提交同步到 `codex_m4`；仍在变化的R4/R5等后续实验由原会话另行提交，不混入这个已验证研究快照。以实际提交SHA及远端分支为准，不能凭同名本地分支推断所有工作都已同步。

交接时检查 `git status --short`：重要文档/脚本若仍为 `??`，它们尚未进入提交；工作树中的删除也未必已经提交。当前实验会话的新结果和代码应另行针对确定文件提交。不要用 `git reset --hard` 或全树覆盖解决并发改动。

`.gitignore`不等于磁盘整理。模型、特征、原图和GB级生成物继续外置；高价值理论、失败分析、方法源码和小证据应由Git携带。完整旧运行包只在需要特定复现时按清理收据取用。
