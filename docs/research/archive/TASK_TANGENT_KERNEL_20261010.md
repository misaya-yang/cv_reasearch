# 冻结模型的任务函数先验：末块经验梯度核

2026-10-10。此前已测O24统计重组、全局相关先验和跨view绑定均未给出稳定跨域完整收益。新假设改为：**预训练网络参数化所定义的任务函数相似性，能否比末层特征余弦更可靠地迁移reference mask**。不修改DINO权重，不跨数据集训练，不使用FoRIS场或伪mask。

这不是把NTK、线性化迁移或kernel ridge声称为新概念。[Fast Adaptation with Linearized Neural Networks](https://proceedings.mlr.press/v130/maddox21a/maddox21a.pdf)已有有限梯度核迁移。当前仅检验一个具体的mask条件函数先验及其必要对照；最终贡献仍需实际机制、完整跨域收益和最近先例核查。

## 具体函数与可核验公式

当前真实模型为timm EvaBlock，末块24：`u=norm2(x_after_attention)`，`v=W1 u+b1`，`h=GELU(v)`，`z=x_after_attention+gamma2*(W2 h+b2)`，模型输出`o=LayerNorm(z)`。末层MLP为1024→4096→1024，GELU exact，两个dropout为0，最终LayerNorm eps=1e−5。所有这些已从实际本机模型确认。

从合法reference mask的连续面积coverage，计算单位O24的FG/BG加权均值差，归一化为固定任务cotangent a。定义标量任务探针`s_i=a^T unit(o_i)`；**a在求导时固定**，不对cotangent估计过程求导。

令`c_i = gamma2 * J_LayerNorm(z_i)^T J_unit(o_i)^T a`，`eta_i=(W2^T c_i)*GELU'(v_i)`。对末块fc1/fc2的权重及bias求梯度，经验核精确分解为

`K(i,j) = <c_i,c_j>(<h_i,h_j>+1) + <eta_i,eta_j>(<u_i,u_j>+1)`。

它等于真实有限MLP参数梯度的内积，不依赖无限宽假设。逐token百万维Jacobian无需显式保存：四个小型Gram相乘求和即可；核由梯度内积保证PSD。再以自核对角归一化，以共同kernel尺度比较。模型参数始终只读。

在reference的固定128/role面积quadrature上做weighted centered kernel ridge，lambda=.01、intercept不惩罚，目标为平衡二元role风险（不是校准概率）。五臂在候选推断／评分前固定：同次前向O24线性核、额外的O24固定位置投影强控制、MLP隐藏激活线性核、任务cotangent梯度核、全reference均值cotangent的同成本梯度核。位置控制用于避免只赢过未去位置的弱O24；梯度核本身仍是上式原始unit-output函数的精确微分。全reference轴只在cotangent中不用mask，监督label仍相同。

只有任务核胜过新隐藏激活、mask-free梯度核和两种O24，才支持任务条件化本身有效；仅因新增中间表示而涨分不能归给新函数先验。完整signed64场bilinear到1024、零阈值直接输出，不先加CRF；original采用固定binary读出。先检查真实数学与完整队列，不扫描块、kernel系数或temperature。

## 资产和执行范围

固定已有COCO/PACO/SUIM各20对、A/B两个真实任务；R/Q RGB、crop和1024变换不变。按真实输入key去重。现有O24缓存没有末MLP的input/preactivation/hidden或最终norm前输出，因此这些量必须通过真实冻结前向补齐，不能从O24伪造。

先对清单最前两个输入做无queryGT smoke，核对只读hooks、最终LayerNorm输出身份、旧O24差异、实际内存和Jacobian公式。通过后按同一已冻结清单继续所需输入，已完成key不重复编码。FP32的五种实际中间量全部保留用于后续复用，预计不超过22GiB；本机当前磁盘足够，MPS上限20GB，单个B1 producer。

旧O24与新B1 O24逐输入比较，不能假定不同batch相同。如果有差异，所有新kernel臂使用同次新O24，外部完整baseline需核对或从同O24重放后再给严格配对结论。

入口`prepare_task_tangent_features.py`，新分支缓存`cv_data/a/task_tangent_features60_20261010/`；不写回旧共享raw。算子`task_tangent_kernel.py`，独立autograd核对`check_task_tangent_kernel.py`。没有新的照片、数据集、下载或模型权重更新。query标签只在完整字段封存之后评分；任何oracle仅为另列诊断。

## 已通过的真实前提

实际60对去重为119个输入，全部FP32中间量已完成，每个输入新O24都与旧input-bound缓存逐位一致，末block-output经原LayerNorm也与实际输出逐位一致。两个smoke输入后没有重复编码；模型参数更新始终0。已有原生FoRIS结果可作严格同O24外部对照。

三真实token对末MLP共8,393,728参数的显式autograd，与解析Gram相对误差1.09e−14；MPS归一化核误差1.19e−7；原O24 kernel实现对直接线性ridge字段误差9.54e−7。没有query标签参与这些检查。

候选入口`run_task_tangent_kernel.py`，结果`cv_data/a/task_tangent_kernel60_20261010/`。全部120真实任务、五臂600完整预测先封存后评分；统计以两任务／照片平均IoU，并以连接的R/Q照片组bootstrap，不能当官方class mIoU或独立确认。全图和同一外部候选内／外AUC用于事后解释，不进入推断。
