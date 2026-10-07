# Lessons from the two core Codex chats

历史案例，按需阅读。2026-10-06 起当前阶段以 [PLAN](PLAN.md) 的有限验证为准；
下文关于下一轮机制或实验的讨论不产生待办。

This is project memory, not another instruction list. Read a relevant case when making a similar decision.
The operational agreement is [AGENTS.md](../../AGENTS.md). These are observed execution errors and bounded
scientific lessons; they are not a general ranking of models or evidence that future research must fail.

Sources inspected through the chat reader on 2026-10-05:
- [检查实验运行状态](codex://threads/01a0f783-6584-7e60-b6d3-5bf227e7e7bc): recent results and earlier correction turns.
- [查明五天持续失败的原因](codex://threads/01a105d0-ab1b-74f0-a71a-0832dbb4925d): the available discussion and preparations.

## 2026-10-06 至 10-07：两天批量方法的错误

结果：两天 60 多次提交；300 张 Astra 卡评分 0 张，30 张 Pro 卡评分 9 张，无一超过自己的底座；
没有产生一个能进论文主表的数。分数在 [RESULTS](../../evidence/local/RESULTS.md) 末尾。

| 做了什么 | 为什么错 | 以后怎么做 |
|---|---|---|
| 每张卡用一个点子替掉整条流程，在原始 DINO 特征上单独出掩码，对比的底座是参考图线性读出 46.58；同一批例子上完整 INSID3 是 56.38、完整 FoRIS 是 61.56。 | 赢了底座也到不了基线，结果无论正负都回答不了论文的问题。原文规定的强底座（完整 FoRIS + MEAN16）一次没跑。 | 新想法只换一个环节，其余环节照旧，和 FoRIS 的同一环节比。开跑前先写出"赢了能说明什么"。 |
| 以方法数量为目标：1000 → 100 → 300 → 30，四个代理各领一段编号。 | 数量代替了交付物；实现数、跑完数、有效数分开记账也改变不了有效数为 0。 | 不设数量目标。外部模型给的卡是参考，不是任务清单。 |
| 没有先对照已有的错误事实就直接上 600 例。M04 补进 3690 万假像素（纯度 13%，回本要 42%），M18 删掉 860 万真目标像素。 | 这两种失败（相像区域、单张参考盖不住查询目标的外观）早就在账本里；昨天的 200 例也已显示对参考图拟合得更精细没有用。 | 先在选错的图上数张数，过线再比 mIoU。AGENTS 第 4 条本来就要求这一步。 |
| 几百张卡排队在同一个 600 例上评分，没有复读规则。 | 有实质改动的方法配对差的标准差约 1 个点，几百个无效方法里最高的会碰巧到 +3 左右。 | 筛选、复读、最终读数用不同的例子；同一批例子不拿来挑赢家。 |
| PLAN 涨到 361 行，装满哈希、事故记录和历史授权；用户指定的主表实测和先数张数的扫描被标成"历史"，没有执行。 | 唯一的待办清单读不出下一步，真正该做的事被挤掉。 | PLAN 只写位置、下一步和规矩；细节放证据目录。 |
| 工程事故：没估成本就放 600 例（C102 首例 1668 秒，E242 每例 29 秒）；对照批漏了 `--methods none` 重跑主方法；运行快照漏掉 C++ 源导致 600 例全失败；本机用 SIGSEGV 做故障注入，弄出 4 次 Python 崩溃。 | 时间花在调度、监控和恢复上；最后一条还违反了本机不跑实验的约定。 | 先跑一例完整输出并计时，再放批量。本机只做极小的检查。 |
| Claude：把"知道物体大小值 +3.98"当成估大小的收益报给用户，没有拆出其中 +2.14 来自没找对目标的图，只靠大小估计能拿的是 +1.69；给 Codex 写的计划第一版 44 行；早先给出的独立版本约 60.1 是外推，后来撤回。 | 诊断上限被说成了可得收益；计划太长没人照着做。 | 上限先按"已找到 / 没找到"拆开再报；计划十几行以内；外推的数不进目标。 |

## Execution mistakes that changed the task

| Observed decision | Why it failed | Lesson for the next session |
|---|---|---|
| Five requested independent methods became one regional method with five control groups. Acknowledged in first-chat turn `01a105d8-415d-7723-bef7-114bfcd6f24e`. | Its own PLAN and delegated work replaced the user's deliverable. Finishing the plan could not satisfy the request. | Preserve the requested unit of work and count. Controls explain a method; they do not multiply it. |
| The Pro masked-query construction was changed to natural crops, with the original moved into a control arm. Admitted in first-chat turn `01a105ce-6682-75e0-984a-8cca29df55ec`; diagnosed in second-chat turn `01a105d2-b32d-7f30-8234-ae65ba6223da`. | A modified construction was treated as execution of the original proposal. | A justified change is explicit and separately named; it cannot stand in for the untested original. |
| Repeated CPU checks, proof/review tasks and queue preparation accumulated while the actual method score remained absent. QK's first real run completed zero episodes because the CRF path was missing. | Preparation was reported as progress toward efficacy; the actual environment was not exercised early enough. | A real pipeline smoke answers compatibility. It should lead to the authorized decisive run, not another expanding preparation cycle. |
| The second chat first favored Pro because its code existed, withdrew that choice, then called a new task-state proposal the main method without a result. Turns `01a105d2-b32d-7f30-8234-ae65ba6223da` and `01a10600-6158-7692-81ce-1ac0ba7e07aa`. | Sunk implementation effort and a plausible narrative substituted for scientific selection. | Name it a candidate until evidence supports the paper claim. Judge the mechanism and useful comparison, not which code is closest to running. |
| One failed reference classifier was followed by a joint graph without evidence that its links could resolve identity; first-chat turn `01a109b7-46a4-7241-87fa-4f8cf13eaa16`. | A computable extension displaced the strong host while retaining the same untested ambiguity. | Reconsider the failed link before another mechanism. Preserve effective host capabilities or show how the complete alternative supplies them. |
| The user redirected attention to beating complete FoRIS while discussion stayed on SAM3 naming and local diagnostics; first-chat turn `01a108e9-a318-74e1-adc7-9d7766bea357`. | A result in another resource setting was becoming a substitute for the requested claim. | Reconnect a local measurement to the complete output and intended comparison. A maintenance request likewise is not permission to resume research. |

## Scientific conclusions that needed correction

- **A good oracle is not a usable signal.** A chosen mask/seed/trimap gains access to query labels. Seed purity
  and coverage also changed together in some interventions; their difference was not a purity-only effect.
- **Source success need not transfer.** The source-verified candidate improved one known hard subset but not
  overall target choice; its source-selected cut then failed on query extent. The exact cohort and result are
  in [the ledger](../../evidence/local/RESULTS.md), not a universal impossibility claim.
- **The complete baseline changes the reading.** Matte's positive pre-CRF signal became an unresolved small
  increment against complete FoRIS and did not beat delete-only. Different SAM3 versions also must keep names.
- **Neither significance shortcut is valid.** A positive CI crossing zero is not a proved failure; a positive
  development CI after many choices is not independent confirmation. A few examples cannot resolve a small gain.
- **The score and explanation have separate burdens.** Lower objective, higher AUC, matching coefficients or
  a mathematical identity do not prove object identity, universality or complete-method benefit. Repeated
  failures constrain constructions; they do not prove that a single pair or DINO has no remaining information.
- **The user asked for judgment, not impossible certainty.** Requiring an experiment to be proven successful
  beforehand causes endless objections; executing every plausible formula causes blind trials. The missing
  bridge is a concrete error, a reason the mechanism changes it, and one informative complete comparison.

## Why the previous instruction stack did not solve this

The old global preferences, project rules, dated plans, HANDOFF and automatic memories all prescribed actions.
They also preserved incompatible session-specific decisions: keep the GPU on / always shut down; kill after
a few cases / do not judge on a tiny cohort; measure first / do not write code until a signal is already proved;
continue until a paper / obey a request limited to maintenance. Some memories still proposed completed D1/T2.

The repair is one current agreement and state, not a longer ban list. In particular, “a supervised probe failed,
therefore no hand-written rule can work,” “new external information is always required,” and “the first cases
show no signal, therefore there is none” are not standing principles. Skill output and external-model advice
inform the lead's judgment; they neither replace the user request nor confer permission to run.

The cleanup changes the repository and its entry points. It does not establish a new segmentation result,
erase the historical failures, or guarantee that an already-open agent has reloaded its instructions.

## 2026-10-05 A*/B* optimization evidence

- Compare matched depth and direct complete selection. The large joint2-vs-one-step gain mostly paid
  for reconstructing a complete source mask; joint2 did not beat greedy2 or direct selection onDEV241.
- Reference self-match calibration need not transport to cross-image margins. Cross-image calibration
  improved foreground area estimation while localization still failed. Mass, ranking and final-mask
  utility are separate links; test their combination against the strongest same-information rule.
- Two weak value estimators agreeing on a change is a surrogate condition,not a true-IoU guarantee.
  OnDEV241 the two-estimator guard added0.031points over the unguarded rule with an interval crossing zero.
- Retain reusable features and compact sufficient statistics. Poor accuracy is a reason to retire a
  tested construction; it is not a reason to remove feature inputs shared by later constructions.
- The public-labelled4000list includes every previously used241/600/600episode pair. Bind benchmark
  scores to their protocol and exposure; a larger list does not turn old cases into fresh confirmation.
