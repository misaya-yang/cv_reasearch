# 同权重下检验覆盖回归与任务角色回归

本轮继续使用原Deep100/PACO100、codex_m4、aidemo及已存whole R/Q+四角真实O24。上一轮reference-LOO校准P0已经完整失败；不再重跑该阈值路线、R1-G、旧6000或官方全量。仓库另聊天正在研究旧APD-ridge参考IoU标定与FoRIS anchor，这里独立隔离一个尚未实测的监督目标语义，写入`cv_data/a/cache_mechanism_study_20261010/balanced_role_ridge/`，不修改其研究文件或冻结字段。

## 可被实测否定的差异

旧head在每参考128/role面积分位采样并去重，使用相同selected覆盖c和C=sum(c)、B=sum(1-c)，权重为a=.5c/C+.5(1-c)/B，回归目标为2c−1。它预测的是覆盖函数；不等价于平衡两个角色的二元平方损失。

保持ids/a/CPU FP64 dual ridge/.01及未正则截距不变，只改目标为

`y_star=(c/C−(1−c)/B)/(c/C+(1−c)/B)`。

这是`sum .5c/C(s−1)^2+.5(1−c)/B(s+1)^2`的精确折叠目标，常数项不影响系数。在纯binary patch上两目标完全相同；在mixed patch上不同。新的零阈值是该平衡角色回归的决策假设，不当作已校准概率、query像素最优阈值或创新性证明。单个patch的coverage也不是DINO语义纯度。

该方向与正则Fisher有已知代数联系；旧全参考Fisher的query迁移负结果必须保留为约束。当前区别是同既有采样/权重/正则化和相同实际APD/native观察下的目标对照，不以换名字重复宣称新机制。若信号或完整分数失败，即拒绝本对照，不扩大参数。

## 唯一固定默认候选与控制

沿原study已冻结profile、输入身份、whole semantic APD开关与basis。reference norm、raw whole64→128再norm、真实四角norm放置及APD顺序保持。old/new共享同6份raw和同观察；缺缓存失败，不重新编码或写raw。原old APD-ridge输出对原保存字段复现，记录实际浮点/掩码差异，不后验放宽容差。

默认完整输出固定为`(.5 score_global+.5 score_local4)>0`，原帧先bilinear连续score再strict>0。old/new各自global/local4为两个观察控制，共6臂。无FoRIS mask anchor、query伪标签、局部CRF、BP或自适应alpha，不按数据集名称选择输出。仍保留同批完整FoRIS、MEAN、已有快九窗作为强对照。全程一个CPU2线程数值进程，不占MPS。

源码、参数、输入及配置先冻结；全部200的六臂连续场和完整mask封存之后才读query GT评分。原图按Deep道路I/U、PACO原87已观察fold/class逐类累计I/U再平均，与原pilot完全同口径；另外沿已固定两个困难ROI计算old/new AUC/AP。所有样本已曝光，这是开发对照，不是独立确认。

结果必须交付：old/new复现收据，六臂配对完整mIoU，TP/FP/FN与编辑收支，困难区排序实际差异，raw验证/fit/field/输出的本轮CPU/IO时间。新增DINO应为0，但原始特征读取与新匹配不能假称字段重放；缓存时间不当冷端到端。

当前仅准备，没有结果、采用或下一批扩跑。Strong净涨分及可用成本目标未完成。
