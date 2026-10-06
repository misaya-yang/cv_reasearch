# ProM2 CPU计算优化：同合同四例核验

平均单例六臂推断：14.888804s → 2.938184s，实测5.067×加速。四例×六臂field逐元素完全相等，最大绝对误差0；1024与原尺寸renderer mask XOR全0。2CPU/2threads，峰值RSS 809,816,064bytes。

首例profile：16.798s中20次`np.add.at`耗13.723s（81.7%），role_dictionary14.200s。将聚类中心累加改为按token顺序排序的one-hot CSR乘FP64特征，特征只转换一次；每簇仍按照原token顺序以FP64累加。没有改变32role/20iter、种子、top2、互20边、空间边、四状态表、block删组、任何六臂或求解器参数。

| episode | 原实现秒 | 优化秒 | 六臂field/两级mask |
|---|---:|---:|---|
| 0_0_72 | 14.843142 | 2.884934 | 完全相等 / XOR=0 |
| 1_0_73 | 14.859450 | 2.998489 | 完全相等 / XOR=0 |
| 2_0_74 | 14.888545 | 2.996417 | 完全相等 / XOR=0 |
| 3_0_75 | 14.964081 | 2.872894 | 完全相等 / XOR=0 |

原实现SHA256：`46903c790c8b29ca34fde3b1af5bdba3a33b6d19d1d7dfd8d6567643513e0baf`。优化SHA256：`27c867bd814a986e178696b07a6fe1733aa7f479b47ded83b747db5b81be7c74`。

实现只改`src/ics/methods/pro_reference_relations.py`；复核脚本`scripts/profile_pro_reference_relations.py`；完整receipt与每例profile文本位于本目录。脚本只加载q/r/cov/score，不读取query GT、不启动编码器；传入的class/fold字段不进入predict。

远端独立namespace：`/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/pro_relations_profile_v1`；优化模块为`pro_reference_relations_optimized.py`，四例证据为`parity4/`。运行原模块只读自`code_318a5564bafd`，原immutable snapshot和既有600队列未改动。

这四例只检验同合同计算和输出一致性，不是新独立分割收益。计时有cProfile、两核亲和性、线程上限与当时服务器负载；没有把该均值外推成600队列实际完成时间。优化后首例主剩余成本：六臂infer_roles1.046s、两图edge_types.958s、role_dictionary.685s，无需继续修改方法。

示例复核参数（新out必须不存在）：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/root/demo4_cache/env /root/miniconda3/bin/python
  runs/pro_relations_profile_v1/profile_pro_reference_relations.py
  --baseline-file code_318a5564bafd/src/ics/methods/pro_reference_relations.py
  --candidate-file runs/pro_relations_profile_v1/pro_reference_relations_optimized.py
  --source-repo code_318a5564bafd --manifest bound600_v2/smoke4.json
  --out runs/pro_relations_profile_v1/<fresh_output> --limit 4
```
路径基准均为上述cvpr_prepared9根目录；命令展示为参数清单。
