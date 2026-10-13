# 实际清理与并发误移修复

2026-10-12（America/New_York）。本次清理已实际移出旧目录，随后发现误移了另一会话当前工作；已停止继续删减并修复。精确机器记录见[cleanup.json](../../evidence/cleanup.json)。

## 2026-10-12后续纠偏：保留研究价值

此前按当前依赖闭包裁到约2.6MB过度精简。现已补回高价值成功/失败的理论正文、完整实验设计与研究源码，不以“当前是否import”决定保留。

- 选定179份科研Python：当前34、完成研究21、高价值负结果/理论诊断100、必要依赖24；145份此前缺失文件已补回，所有当前文件保持原版本。
- 42份核心研究记录另加6份配套报告已补回：可辨识性、source-span、Mask信息路径、Strong拒识、context/tangent/层级、局部观察、完整融合负结果等。
- 增加3.67MiB便携证据：47项主要来源、20项机制/纠错来源、14项方法规格，重要结论无需本机归档路径即可阅读。
- 现工作树约10MB（不含.git），没有带回大特征、完整Mask或GB级生成物。旧的大小和代码数保留在下文作为操作历史，不代表当前值。
- 在没有cv_data的临时目录中，已用Python标准库复算C1的2136例及B1共同K3的4585查询；C1对随仓库原报告的最大误差约1.42e-14。

[研究导读](RESEARCH_GUIDE.md)是跨机器理解入口，[接续说明](CONTINUING_ELSEWHERE.md)区分可随分支阅读/复算的材料与实际运行的外部资产。[验证收据](../../evidence/research/handoff_validation.json)及[逐项保留理由](../../evidence/research/retained_work.json)均在仓库。该轮检查时修改尚未提交；最终发布以本分支实际Git提交和远端SHA为准。并发实验后续修改单独保留，不由整理提交接管。

## 已完成的第一轮清理（历史快照）

- 清理前工作树（不含.git）：71,733个普通文件、33.915 GB。
- 整个旧outputs已迁至仓库外；旧evidence批量结果、已关闭方法、旧计划/启动包/重复复盘迁出。
- 2,490个原Git已跟踪路径当前标记为删除；整理提交前曾处于未提交状态；保留此操作快照，未改写Git历史。
- 永久删除233个可再生缓存或字节相同重复文件，共4.837 MB。
- 保留的紧凑证据包为6文件、1.658 MB：34家族结论与COCO4000/S600/N6200/C1/B1必要逐例I/U。
- 29份原必要科学源码字节未变；另外保留另一会话当前R1/R2/R3及其动态像素描述子。所有生成数据继续放仓库外。

这是减少活跃工作树和Git待跟踪内容，**不是宣称释放了全部磁盘空间**。可追溯原始结果仍在仓库外；.git历史未压缩或重写。

## 并发误移及修复

错误：使用旧依赖清单整目录迁出，没有识别当前另一会话新增的R1/R2脚本、计划和R1报告，也移走了R2动态加载的 `diagnose.py`，实际造成其FileNotFoundError。

修复：另一会话已恢复/修改的R1/R2/R3保持原状；仅恢复缺失的当前计划、主报告、x1012任务报告和像素描述子。用排他创建避免覆盖并发修改。R1/R2当前配置所列10项源码SHA全部一致；R1/R2/R3、A6/B1共用模块和动态像素描述子导入通过，没有运行模型或新科学实验。

当前执行依据是[单参考R1/R2/R3计划](SINGLE_REFERENCE_NEXT_PLAN_20261012.md)，[PLAN](PLAN.md)已同步。本次E1/E2建议不能覆盖另一会话正在执行的工作。后续清理须核对当前会话与快照后的新增/修改，不能把未出现在旧保留表中的文件默认当成过时。

## 仓库外位置与按需恢复

- 原始历史：[归档目录](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/repository_retired/20261013T014935Z)。
- outputs生成产物：[外部资产目录](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/repository_assets/20261013T014935Z/outputs)。
- 逐文件原始快照：[before_file_manifest.json](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/repository_retired/20261013T014935Z/_cleanup/before_file_manifest.json)。
- 每项核心证据的原路径、现路径和SHA：[results.json](../../evidence/research/results.json)的 `source_bindings`。

归档保留原仓库相对目录结构。只恢复具体需要的文件，先检查当前是否已有新版本；不要整体覆盖当前工作树或恢复旧队列。当前目录没有链接回整套历史的大目录软链接。原文件依赖绝对路径时，按对应收据处理，不把旧报告中的“运行中”当今天状态。

## 防止再次混入大产物

`outputs/`原来整体ignored，evidence原来混有tracked/untracked/ignored；本次三类都实际处理，未用整目录ignore替代清理。当前小型JSON/IU证据可被Git跟踪。

`python -B scripts/check_repository.py`检查整个工作树，包含ignored文件，拒绝特征/模型/图像二进制、outputs回流或GB规模回流。AGENTS已明确生成数据写 `../cv_data/a/`，当前研究代码和报告不能因旧清单被移走。
