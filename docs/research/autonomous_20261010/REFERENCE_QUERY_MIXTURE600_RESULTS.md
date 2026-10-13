# 参考到查询的角色分布600例：纯label-shift校准条件不成立

2026-10-10，固定已曝光Deep100／PACO100／COCO200／LVIS200。这个诊断没有产生新分割方法、阈值或mIoU。600组无query标注场先封存后评分，实际两个whole O24输入逐项读回；source APD分支与原scene600全部记录一致。当前结果否定把reference的角色分布直接当作query角色分布，仅用未知前景比例解释两图差异。

在相同参考pixel-area监督下，令参考特征条件均值为μ_RF、μ_RB，δ=μ_RF−μ_RB。坐标

```text
s(q) = (q−μ_RB)·δ / ||δ||²
π_coordinate = mean_Q s(q)
```

参考自身的前景／背景条件坐标均值恰为1／0。如果query两角色条件均值不变，π_coordinate等于query真实前景比例。此处检验该假设的必要条件，不宣称这是具有保证的无监督prior estimator。Reference和query标签都按16×16物理面积混合质量定义，不能将混合patch硬化后冒充同一个分布。

|同批macro描述量，source APD|Deep100|PACO100|COCO200|LVIS200|
|---|---:|---:|---:|---:|
|真实query前景面积均值，%|4.299|15.346|12.865|7.095|
|clip后的prior-coordinate平均绝对误差，百分点|17.528|12.563|15.471|12.484|
|query前景条件坐标均值，理想1|.81386|.54713|.66629|.59817|
|query背景条件坐标均值，理想0|.14246|.12693|.19654|.13270|
|query角色均值gap，参考自身为1|.67139|.42020|.46975|.46547|
|query全均值到参考条件均值线段的距离／源角色间距|1.12964|.89856|.73690|.71349|
|prototype方向的pixel-mass全图AUC|.90643|.91074|.95437|.95920|
|未clip坐标越出[0,1]的例数|23|12|4|17|

不做APD的相同rawunit控制，prior-coordinate平均绝对误差为13.641／10.679／13.054／10.058百分点，AUC为.89136／.90475／.93201／.94370。APD提高了这一方向的平均排序，却同时恶化该prior估计；排序与绝对读出不可混同。表中AUC为64网格对真实像素质量的诊断，不是最终mask成绩，也不能与focus128场AUC直接相减为因果收益。

逐例保存的未clip误差满足

```text
π_coordinate−π_GT = π_GT(E_QF s−1) + (1−π_GT)E_QB s
```

两项分别是前景分布偏移和背景分布偏移的贡献；600×两表征闭合误差均通过1e−12检查。Deep的两项均值为−.01039／+.13121，COCO−.04565／+.16812，LVIS−.02969／+.11905；大量背景的小坐标偏移足以产生显著前景面积偏差。PACO为−.07368／+.10143，较大的前景比例令两项部分抵消。clip后的误差另报，不把它代入这个恒等式。

这解释了为何一张reference的可分性、正确角色面积和高query AUC仍不足以校准query零点。它没有证明任意target-density、共用offset／contrast-gain模型或像素外观模型都失败；那些模型各自需要实际检验，不能凭此直接附加复杂模块。

## 必要条件与数值边界

独立静态／合成审阅指出：坐标正确、全均值在参考线段上，都不能证明完整label shift成立。交换query两角色可产生零线段残差而估错prior；前景和背景偏移也可互相抵消；垂直于δ且相互抵消的条件均值变化完全不能被该标量看见。因而本结果是必要条件失败的证据，不能把必要条件通过当成充分性或后验概率保证。通用mixture比例估计本来依赖分布条件；参见 [Ramaswamy et al., ICML2016](https://proceedings.mlr.press/v48/ramaswamy16.html) 和 [PU label-shift研究](https://arxiv.org/abs/2502.21194)。

当前600参考最小角色间距为.1816167，两角色GT面积均存在，全部顺利完成。脚本的通用near-zero-gap合成压力例会触发过严绝对闭合断言，且无query角色输入目前不适用；这不影响本panel，也不能把这个诊断脚本当作已健壮的生产推断API。数学上退化的参考零gap作为明确null处理。代码与所有场保持本次冻结，不为结果改实现。

## 产物与验证

入口：[diagnose_reference_query_mixture.py](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/scripts/diagnose_reference_query_mixture.py)。实际运行从冻结脚本执行，绝对父root来自冻结配置；父scene600源／清单／配置／seal、原profile／basis和实际raw requests全部绑定。无encoder／CRF／新mask、无原产物写入，生成耗时132.98秒，单进程两CPU线程。读取的query标签和baseline路径在生成阶段被阻断，全部600场先封存。

完整数值：[results.json](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/reference_query_mixture600_20261010/results.json)、[逐例条件矩与分解](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/reference_query_mixture600_20261010/scored_episodes.jsonl)、[seal](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/reference_query_mixture600_20261010/sealed.json)、[config](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/reference_query_mixture600_20261010/config.json)。独立数学／source绑定审阅：[review.json](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/autonomous_20261010/scene_model_validation/reference_query_mixture_review.json)。完整600评分已独立复算：600 query/reference mask身份、1,200表征AUC／角色矩／误差账本、八组汇总、1,200旧raw元数据及1,829 SHA均一致，最大差2.22e−16；预先指定Deep0／PACO100的四实际raw向量及四表征moment／centroid重放逐位相同，不冒称另重读全部600 raw。见 [reference_query_mixture_evidence.md](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/autonomous_20261010/reference_query_mixture_evidence.md)。
