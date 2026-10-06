# 参考锚点吸收残差：一项独立CPU候选

**可直接接入同一真实600例，但性能尚未测。** 模块[reference_absorption.py](../../../../src/ics/methods/reference_absorption.py:95)的接口是 `predict(q,r,cov,base)`；输入与现有缓存候选一致。`field`为完整64×64输出场，`info`记录active/abstain、锚点/图规模、残差、增删、时间。最多32参考锚点+64查询模式，共96个图节点，无4096²查询矩阵，无新增encoder。

新增判据是沿**整个未标注查询特征图**到达参考FG/BG吸收端点的路径概率。它不规定查询前景必须满足参考模式比例/轮廓/covariance；不学习Pro M2四状态关系势，不做hull点距或triplet匹配。原MEAN/Huber根据query输入场保真平滑；本辅助图没有base fidelity，而将参考FG/BG设为固定边界。Dirichlet/吸收随机游走是已有数学工具；这里仅是一项新的项目候选，不认证原创性。[完整卡片](method.json)。

最终场为 `base + h(mode) - d(mode)`：h是多步吸收值，d是同锚点的直接、类别平衡核得分。不作额外clamp或minmax，无两类参考边界连通支持的token保持原base。初步的“直接用h替换base”形式保留为控制。采用残差形式的动机来自已测的[九项public600报告](../../../../evidence/local/research_20261006/server_prepared_01a1100b/nine_public600_v2/score/report.json)：MEAN62.9315，而covariance12.3432、prior37.8833、quadratic51.6070；这些是其他算法的观测，不是本候选的分数或成功预测。

根代理接入应保存全部返回比较场，并同MEAN采用相同bilinear1024/strict>0.5：

| 返回键 | 定义 |
|---|---|
| `field` | base+h−d主法 |
| `nearest_control` | 同查询模式上的最近FG/BG锚点余弦margin完整分类 |
| `kernel_control` | 直接核d完整分类 |
| `one_hop_control` | base+仅参考邻边FG比例−d，无query→query传播 |
| `one_step_control` | base+一次加权邻域传播(d初始化、参考固定)−d |
| `component_control` | base+组件FG锚点比例−d，不使用传输路径长度/电导 |
| `full_harmonic_control` | 纯h完整分类，检验替换base风险 |

控制共享原图的active范围及查询压缩，范围外都保留base。对比one_step及component尤其重要：主法若不胜它们，不支持多步路径的必要性。所有参数固定于Config，不扫常数或把控制计作方法。

[check.json](check.json) 的证据分层：

- 六节点路径与解析解及独立吸收Markov迭代一致，误差2.22e-16；单类组件弃权、缺参考类保持base。
- 第一条曲线见证原样保留：主法焦点0.7902/0.2098，但one_step也0.7078/0.2922，不能据此主张多步必要。
- 额外中性枝干见证运行的是完整默认聚类/图/求解/读出前场流程：固定焦点q、R/cov与base，只交换远处两分支的连接角色，主法焦点0.7105316/0.2894684，五个廉价控制皆0.5。这证明查询远处路径能通过完整算法改变输出，不证明自然DINO类别路径可迁移。
- 负例：将同一个64格焦点组判为背景时，前景连接上下文会新增64个假前景格；桥接错误也可误删真实目标。未将该负例改成另一方法或隐藏。

[runner_check.json](runner_check.json) 已通过4096×1024随机查询、满16+16参考锚点、64查询模式、两次重复出现、双CPU worker和GT sentinel：八行packed1024输出、拒绝覆盖均通过。该32锚点路径候选时延约0.297秒，含压缩/图/solve/所有控制；完整读取/渲染/写入约0.437秒。两者都是本机小合成时间，非真实600运行时间或单独部署成本。

独立[run_reference_absorption.py](../../../../scripts/run_reference_absorption.py:17)读取NPZ的q/r/cov/base，保存全部场/掩码、source及输入/预测SHA和完成标记；没有query GT推断输入。核对命令：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 evidence/local/research_20261006/absorption_preparation_01a1100b/check_absorption.py
PYTHONDONTWRITEBYTECODE=1 python3 evidence/local/research_20261006/absorption_preparation_01a1100b/check_runner.py
```

真实DINO路径可能接错类别，64模式可能吞掉小目标，修正也可损伤本来正确的MEAN。建议同一固定600只跑一次完整比较，报告相对MEAN及one_step的配对增量、active/abstain及四类增删。未超简单控制或完整质量受损就关闭本固定候选，不扩展参数搜索。此worker未服务器操作、改共享入口、PLAN/STATUS或Git。
