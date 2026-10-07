# Pro M5：首例完成、严格审核通过、独立评分（2026-10-06）

**已完成首例`0_0_72`的完整七行预测、封存和独立1CPU评分。全部69个ROI（9R+60Q）在block21–24的β=1最大绝对误差均为0，原atol/rtol=5e−5保持不变。** 复用真实native_pair.pt，没有第三次encoder调用。数值/入口已成立；该例所有行和同读出MEAN的IoU均为0，没有恢复指定目标，不支持分割收益或原创性。

当前结果见[机器证据](actual_result_v3.json)和[方法卡](method.json)。旧`actual_audit_v1.md/json`保留两次失败轨迹，当前状态以本报告为准。算法源码SHA=`b9da3fd3b42c08a27c3634d86251b36e75269ff5414105d8408566dabe4c1653`，独占本项文件；没有改共享入口、M1、PLAN/STATUS或Pro报告。

## 组件定位与同合同修复

第6个参考ROI恰有130个token，query64分块为64+64+2。逐层用真实native输入隔离，实际paired block重演对保存状态均0；norm1、完整qkv的V、输出proj、norm2、MLP和完整finish对真实中间量也全部0。只在未padding的ROI-query SDPA出现误差：

| block | 未补齐query尾块的SDPA maxabs | query-only补到64/256行 |
|---|---:|---:|
|21|1.049041748e−5|0|
|22|7.688999176e−6|0|
|23|1.513957977e−5|0|
|24|2.026557922e−5|0|

pointwise的原130行、ceil64到192行、min256三种处理都与native中间量一致，因此不是这次130行GEMM尾块或V内容差异。独立2CPU只读review也排除了saved/contiguous/native-V stride差异，定位同样的SDPA query尾块。

正式修复在attention query尾块不足64行时补dummy zero queries，计算后立刻裁掉dummy输出。dummy从不进入K/V、ROI集合、质量统计或区域pool；不改变任何真实Q/K、β、差分/限幅、温度、阈值、数据或网络算术精度。attention每个query行独立，因此实数数学与Pro原路径相同；这里只保持CPU原生SDPA的收缩kernel档位。β1仍实际计算LN/V、SDPA、投影、gamma、MLP及残差，**没有返回native缓存绕过审核，也未放宽容差或换FP64网络计算。**

参考ROI大小=`[1707,239,181,717,188,130,410,144,380]`。修复后一次完整运行通过全部69×4层审核；原query全四分区覆盖4N，reference覆盖N，完整endpoint/hard/native-region/zero及两个objectness诊断均输出。

## 成本与完整输出

- 4CPU线程，8GiB RSS守卫；实测峰值3,184,361,472 bytes≈2.966GiB。
- 完整cache到所有字段/两级掩码的推断132.5697s；含本地模型加载和cache绑定的cached入口144.7265s。
- 两次既有paired capture为47.18s和52.76s；本次没有重新编码。不能将分阶段相加称独立端到端实测，也不能据此承诺60秒或队列分钟数。
- 1024工作空间及640×640原尺寸bool掩码均已封存；zero字段及两种完整掩码与同renderer MEAN逐点相同。
- 原native cache SHA在预测后复核未变。GT只由独立1线程评分进程在验证seal/输出hash后首次读取；评分0.0826s。
- 当前无M5活动进程，没有自行扩4/24/600或启动其他资源。

## 首例真实质量

GT不是空：工作空间74,546前景像素，原尺寸28,966。MEAN原尺寸预测75,508像素全部落在GT外；所有新行仍未恢复GT。

| 行 | 原尺寸IoU | 原尺寸预测像素 | add-TP | add-FP | delete-TP | delete-FP |
|---|---:|---:|---:|---:|---:|---:|
|MEAN/zero|0|75,508|0|0|0|0|
|extrapolate|0|40,720|0|1,124|0|35,912|
|endpoint|0|78,513|0|3,066|0|61|
|hard-channel|0|130,933|0|70,790|0|15,365|
|native-region|0|48,212|0|36|0|27,332|
|response-objectness|0|0|0|0|0|75,508|
|mass-objectness|0|0|0|0|0|75,508|

extrapolate净减少34,788个误检像素，但未恢复漏目标，因此IoU不变。空输出objectness也能删掉全部误检却仍IoU=0，不能把删除量本身称为参考类别证据或完整方法收益。工作空间全部IoU也为0，原始I/U和四类编辑详见JSON。

此case来自已暴露public/dev row，不是独立确认。宿主明确为`existing_FP16_conditional_cache_MEAN_graph_float64`；M5分支为fresh FP32 native、固定原episode Pi gate=True。新的reference coverage来自完整mask nearest1024→area64，与旧packet最大差0.08203125已记录；不将当前结果称为纯FP32宿主全图链复现或历史分数复现。

## 可复核远端位置

```text
/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/pro_message_audit_v1/
  code_b9da3fd3b42c/                 # 本次不可变代码副本
  outputs_v3_cached/sealed.json      # 69个ROI全部native审核0
  outputs_v3_cached/fields.npz
  outputs_v3_cached/masks.npz        # *.work及*.original完整bool
  outputs_v3_cached/inference_diagnostics.json
  outputs_v3_cached/score_one.json   # 独立1线程，GTafterseal
  component_diagnosis.json          # 原130token ROI逐组件证据
  outputs_v2/episode_00_0_0_72/native_pair.pt  # 唯一复用真实cache
```

后续是否运行其他现有例，由根代理依据本轮用户范围与成本统一调度。本文不生成方法变体/β扫描/新队列。共享M1/M5的`shared_m1_cache`接口已在本地源码具备实际args/kwargs和未投影final LN schema；联合共享验证尚未运行，不能将本次cached审核当作其证明。
