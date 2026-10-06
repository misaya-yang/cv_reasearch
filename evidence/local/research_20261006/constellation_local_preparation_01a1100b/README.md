# Constellation局部融合v2准备结果

这是明确的新版本 `reference_constellation_local_v2`，不是将修订成绩追归原v1。只改变最后的融合范围，参考聚类/地标、响应、峰、姿态、留出验证、NMS、全部Config和canonical token几何均复用原版。[实现](../../../../src/ics/methods/reference_constellation_local.py:14) 调用原v1一次，然后复用它的prior、bag和global场；原v1文件与结果保持原样。

定义 `D = {i: prior_i > 0}`，即接受并经NMS保留的姿态所转移的参考前景正覆盖并集，含正的双线性分数。在D内使用原v1完全相同的融合结果，在D外复制原base，包括原base超出[0,1]的数值。不加框、膨胀、姿态阈值或其他参数；也未恢复原图aspect。[公式及限制](method.json)。这是64格连续场的局部保证；双线性1024读出可将影响扩散到邻近插值像素，不能声称nearest-expanded D外每像素都不变。

现有反例取[四地标单位特征夹具](../constellation_preparation_01a1100b/check_geometry.py:55)，再加入 `base[2:6,45:49]=.9` 的不受姿态支持区域。两个目标的“真实身份”是合成反例假定，实际真实episode为0。

| 合成对象/性质 | 原全局v1 | 局部v2 |
|---|---:|---:|
| 匹配目标保留 | 4/4格 | 4/4格 |
| 未匹配base目标保留 | 0/16格 | 16/16格 |
| 64格场D外与base最大差 | 全图衰减 | 0 |
| 姿态、prior、bag | 原值 | 与独立原v1调用逐值一致 |

[local_check.json](local_check.json) 另验证D内结果与原v1一致、D外不裁剪、缺少姿态与geometry_weight=0时保留base。

**保留的失败条件：** 如果那16格实际是错误目标，v2也保留16/16。局部版不再自动删除没有姿态支持的错误目标；错误接受姿态仍可在D内添加假前景，分数coverage也可能在D内削减正确边界。姿态、长宽比、形变及多实例参考模板的问题原样保留。真实收益、成本、原创性都未验证。

独立入口[run_reference_constellation_local.py](../../../../scripts/run_reference_constellation_local.py:20)只读取 `q,r,cov,base`，输出同双线性1024读出的六行：local_v2、global_v1、原base、clipped_base、bag、prior，并保存local/global/base/bag连续场及局部region。source、v1依赖、renderer和每个prediction/receipt均记SHA；完成标记在全部预测写出后生成。

[runner_check.json](runner_check.json) 已用4096×1024特征、两次重复出现、两个CPU worker验证：GT sentinel未打开、六行packed1024输出、覆盖外连续场不变、插值影响范围外mask与base一致、保留重复、拒绝覆盖已有输出。命令：

```sh
PYTHONDONTWRITEBYTECODE=1 python3 evidence/local/research_20261006/constellation_local_preparation_01a1100b/check_local.py
PYTHONDONTWRITEBYTECODE=1 python3 evidence/local/research_20261006/constellation_local_preparation_01a1100b/check_runner.py
```

根代理接入时可调用 `predict(q,r,cov,base)`：`field`是v2，`global_field`是原v1，`bag_field`、`prior`、`local_region`供控制/计数。独立runner的NPZ预测行名称分别为 `constellation_local`、`constellation_global_v1.control`、`mean.control`、`clipped_mean.control`、`constellation_bag.control`、`constellation_prior.control`。本worker未修改共用入口、PLAN、STATUS，未SSH或运行真实600例。
