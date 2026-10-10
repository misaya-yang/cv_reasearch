# 研究重心纠正：共同的FoRIS判别核心

2026-10-10，用户指出：“研究了那么久研究了什么？没有发现我们核心都是基于foris的吗？”

现有主要成功确实共享FoRIS。之前按图、融合、读出分线能验证局部事实，却不足以解释共同核心为何有效、改变什么才是新增能力。新ridge+anchor也是FoRIS校正器，不能作为独立分割核心。这里不把用户的话解释成“必须摆脱FoRIS”或“必须保留每个FoRIS模块”；研究对象转为各方法共同依赖的有效判别链路，暂不继续追加外围修补公式。

## 共同依赖需要明确

- RCG/MEAN继承FoRIS连续score，再改rank、graph和读出。
- 原/快区域法在每个真实局部输入上调用FoRIS前端；整图FoRIS又定义query伪FG/BG来筛组件。不同观察改变了同一个前端的输入和竞争范围。
- fine/residual继承FoRIS/RCG/MEAN已有语义场，再改变细读出。
- 旧whole角色及新ridge校正继承完整FoRIS mask的符号和source score幅度。
- 独立R1-G移走该链路后，直接角色读出已严重失败；后加几何没有补回这些能力。

实际默认源码链：单位O24 → 按参考/查询语义分支APD → 参考FG聚类多原型与最相似20%背景的正交负方向 → 最近参考标签票 → query特征+RGB+位置聚类、单seed cluster prior → semantic disagreement与cluster重权 → 图内minmax/二值化/CRF。不是单纯“DINO相似度+CRF”。源在`cv_data/third_party/foris_official/models/foris.py`。

已有LVIS1400逐阶段证据（CLI）：FG27.7216、BG35.0643、vote38.1386、prior40.1218、penalty42.8745、pre44.1012、完整45.9481；是顺序管线结果，不是独立可相加因果效应。默认stage1正标量gate在后续选raw-target及再次单位化的路径中理想算术可抵消，不能将所有源码模块都称有效身份证据。

## 追加分工，仍通过MD交换

03线：在自己的目录新增`CORE_DEPENDENCY.md`。追踪上述共同依赖和实际有效计算路径，把各成功的“继承FoRIS能力”与“自身增量”对到准确代码/已测控制；结合既有atomic1400/其它已执行stage证据，回答目前确知的核心瓶颈和不能从外围成绩推断的内容。不要重复跑图、不新增后处理候选、不改既有报告或全局文件。

05线：在自己的目录新增`CORE_REFERENCE.md`。对照FoRIS真正的参考FG多模态、hardBG正交化、mask采样、对应与query seed机制，核查新ridge/R1-G实际保留/丢掉什么。找可以从现有缓存验证的最小“核心内”问题；不直接发明替换公式，不将参考自拟合当迁移证明。重点区分位置/尺度、参考任务定义、同场景组织和最终校准。必要来源读原代码与旧stage账本，暂不另起数值大队列。

主代理：读取并复用`src/ics/methods/stage_bank.py`、`scripts/lvis_atomic_study.py`及已保存atomic字段，准备同输入的核心响应追踪。先回答核心各原始证据在困难区域是否有效、哪一步增添/抹除目标；有必要的新派生字段只从已有raw replay获取。不会把又一个后处理上涨直接命名为新核心。

资源：两条追加静态/小账本线各1CPU，无DINO/MPS/新raw。旧五条研究及新增PACO/COCO/常量控制结果都保留，不改写失败。读取此文件及ROOT_RESEARCH是唯一交互入口，调度只发一次文件路径唤醒。
