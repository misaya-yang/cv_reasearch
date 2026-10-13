# 参考任务条件的冻结末块上下文相容性实验

2026-10-12。当前聊天的人类指令“完成后你继续推进实验，利用好GPU资源”授权继续实验。核心目标仍是原创、简单、跨数据集的完整单参考算法；本实验只检验一个此前没有完整测量的机制，不预称算法成功或原创贡献。

历史Q/K检索失败说明预训练关系投影本身未稳定改善读出；历史scalar gate消去说明条件必须进入真实运算。此处利用已保存的真实X23、冻结末块的V、输出投影、MLP及最后LayerNorm，检验任务条件的上下文相容性是否提供超过直接检索的判别。它不使用FoRIS预测、query伪标签、GT阈值或参考拟合来生成结果。

对每个query patch，分别以参考连续Mask覆盖c和1-c给全部4096参考patch的注意力权重加log质量并归一化。跨图使用实际末块pre-RoPE Q/K及1/sqrt(64)缩放，避免假定R/Q位置对齐；query自身保留原生RoPE。用原native attention context与该角色V context各一半形成新context，原5个prefix维持native。之后运行原冻结输出投影、残差、MLP和最后归一化，得到单位特征zF、zB及原z0。

唯一主场为mean((zB-z0)^2)-mean((zF-z0)^2)：将更少改变native query表示的角色解释为更相容。这个假设可能错误，尤其是背景更覆盖query上下文时；其符号不是已校准概率。固定混合1/2，不扫描层、权重、温度或阈值。

必要控制：同Q/K/V和角色质量但只在attention context空间计算距离的linear_context；同末块过程但参考Mask按固定置换打乱的null_context；同实际Q/K的角色logmean密度qk_density；相同软角色、O24余弦温度.07的o24_density。主符号零阈值和共同minmax/.5读出同时事先固定，minmax只检验范围效应，不事后选赢家。

样本为既有COCO/PACO/SUIM各20对，每对两个真实Mask任务，共120任务、119个唯一RGB输入。复用已封存X23和O24，没有新照片、encoder全前向或旧队列重启。M4 Max 36GiB、32核GPU、MPS FP32；GPU完成QKV、跨图attention、末块重放与读出。CPU仅做文件身份与评分。远端SSH连接当前关闭，不作为计算依赖。

先核对零替换与原末块一致、模型权重冻结、R/Q及参考Mask身份；推断拒绝query标签。所有五场／两读出、1200完整1024Mask封存后评分：完整含背景IoU、四编辑、任务切换AUC、全图及FoRIS前景内排序、完整GT阈值上界仅诊断。统一以每照片A/B任务均值汇总，R/Q照片关联组区间，不冒称官方class mIoU、独立确认或十域优势。

若没有超过简单匹配控制的任务判别增量，就关闭这个固定构造；若有局部信号仍须检查完整输出与跨域反例，不能仅以Mask响应或低上下文代价继续扩大。该实验改变推断时attention context，冻结所有参数；不能称DINO原生自带跨图注意力。

运行入口：scripts/run_task_context_compatibility.py；产物：../cv_data/a/task_context_compatibility120_20261012/。状态与完整结论在执行后补入本文件。

## 已完成结果与科学判断

全部120任务／1200个完整1024 Mask已先封存后评分，独立复核通过。下表是每照片A/B任务平均IoU，COCO/PACO/SUIM各20对，RQ照片关联组为20/18/20；不是官方class mIoU或独立泛化确认。

| 固定完整读出 | COCO | PACO-Part | SUIM |
|---|---:|---:|---:|
| 上下文相容性／零阈值主场 | 28.759719 | 14.999668 | 31.998390 |
| 同场／预设minmax控制 | 25.406675 | 27.448468 | 37.158318 |
| 线性context／零阈值 | 12.507447 | 10.625838 | 25.687255 |
| 打乱Mask／零阈值 | 3.957103 | 9.509926 | 13.505766 |
| Q/K软角色密度／零阈值 | 14.492041 | 29.063211 | 31.226580 |
| O24软角色密度／零阈值 | 26.636109 | 31.002366 | 51.172170 |
| 同输入完整FoRIS | 55.854415 | 47.682365 | 60.238585 |

正确Mask对打乱Mask有明确任务响应；主场任务切换AUC为.854570/.834982/.759526，O24为.910960/.923892/.843350。主场响应差AUC为.967476/.932676/.799098，打乱对照.508822/.488943/.565349。条件确实进入了实际查询推断，但强条件响应没有自动改善每个任务的目标归属。

完整投影／残差／MLP的主场在零阈值下高于线性context，但全图排序相对线性没有共同改善，PACO显著更差。不能只用零阈值完整增量声称非线性映射提供了更好的任务识别。主场对简单O24零阈值为COCO+2.124（区间跨0）、PACO−16.003、SUIM−19.174；后两域区间均负。FoRIS前景内排序相对O24：COCO+.013697区间跨0，PACO−.071850、SUIM−.071712且区间均负。

完整像素GT最佳level-set阈值上界仅诊断：主场52.782315/47.351686/56.488594，O24同类上界60.127803/57.940555/62.144061。缺口包含该标量场的排序能力，不能仅由改变零阈值解决。该固定相容性构造关闭，不扫混合系数、层或读出变体；未建立原创算法或多域完整优势。

这个否定的价值在于区分“Mask没进入有效运算”与“运算响应了Mask却没有更好识别任务”。后者在本实验实际发生。对新方法的约束是：任务条件需要帮助排除查询中的竞争解释，而不仅改变特征方向或使前景／背景上下文产生不同响应。结论限于本构造及其相容性准则，不排除其它任务条件化或其它利用这些内部量的读出。

## 身份 成本与复现

60个native query O24均与旧缓存逐位相同；三域零替换重放逐位相同，原参数digest前后不变。MPS FP32实际完成543次末块部分前向，含540次实验所需前向与3次零替换核对；全encoder新前向为0。全部case时间合计53.097秒（含缓存读取／字段与mask写入），main续跑52.433秒，首smoke4.670秒与main4.196秒初始化另计。最高MPS driver采样5.108GiB，不声称真实峰值／满载利用率或冷推理总成本。

独立CPU复核以NumPy索引映射GT、packed-bit popcount重算完整I/U/FP/FN与四编辑，并用sklearn加权ROC验证排序；共15560项核对，AUC最大误差2.22e−16。另以NumPy FP64独立重建真实QKV/RoPE/soft-role attention、原投影／残差／GELU／LayerNorm，验证三域两任务各9位置、五场共270个点；主场最大误差1.27e−10。独立验证10.819秒，0MPS、0新完整mask、0encoder。

- [完整评分与所有对照](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/task_context_compatibility120_20261012/REPORT.md)
- [机器完整评分与逐例得失](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/task_context_compatibility120_20261012/report.json)
- [全部输出seal与实际资源](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/task_context_compatibility120_20261012/sealed.json)
- [独立复核](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/task_context_compatibility120_20261012/verification.json)
- [完成收据与决策](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/task_context_compatibility120_20261012/completion.json)

所有进程已成功退出，没有活动producer／scorer；旧缓存与默认方法未改。核心研究目标仍未完成。

## 补充诊断：对线性弱控制的增量来自哪里

本项在主实验曝光后，只读取已封存字段；不是预先独立确认或新的算法。分别交换完整1024场的空间排序与分数分布，保持零阈值，平均rank处理ties。

| 每照片两任务完整IoU | COCO | PACO-Part | SUIM |
|---|---:|---:|---:|
| 原线性context | 12.507447 | 10.625838 | 25.687255 |
| 完整非线性context | 28.759719 | 14.999668 | 31.998390 |
| 原线性排序＋非线性分布 | 28.687765 | 15.756379 | 32.404965 |
| 非线性排序＋原线性分布 | 12.932194 | 10.064588 | 25.363518 |

原线性排序加非线性分布已经复现几乎全部完整增量，PACO/SUIM点数还更高；只换成非线性排序而保留原分布，仅+.424747/−.561250/−.323737，COCO/SUIM区间跨0、PACO区间为负。最多3像素的面积差来自tie平均rank，全部旧context／linear终点IoU复现。IoU和投影存在非线性，这些差不能相加作精确因果分摊。

因此，主场赢线性零阈值控制的正结果，不能被解释成新增的空间任务识别。此结论与主场输O24排序及完整阈值上界一致，也解释了为何“条件化与非线性生效”尚未完成核心任务。

[全部逐任务与RQ组区间](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/task_context_compatibility120_20261012/readout_diagnosis.json)。CPU诊断28.184秒，0模型／encoder／MPS，240个临时诊断二值化，没有保存新算法预测或改旧seal。
