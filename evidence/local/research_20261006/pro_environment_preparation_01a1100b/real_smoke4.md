# Pro M4：四例真实完整结果（2026-10-06）

结论：**M4-paired原尺寸class-summed mIoU为22.1424；cached完整native的Pro renderer variant为36.3608，差−14.2184。** matched class-LDA25.7565、ref-canvas31.2604也超过paired。这四例不支持当前完整M4-v1的自然查询收益，不能以成功拟合、源均值±1或精确cut替代这个负结果。[real_smoke4_46466/report.json:2-26]

固定清单为已暴露public0四次出现`0_0_72,1_0_73,2_0_74,3_0_75`，来自根代理指定bound600_v2/smoke4。没有按GT选例。四例是执行/效果smoke，不是独立确认，未从4例估计置信区间。

| 完整行 | 同历史GT的1024 mIoU | 真实原annotation的原尺寸mIoU |
|---|---:|---:|
| M4-paired | 22.1591 | 22.1424 |
| M4-mean | 20.9937 | 21.0029 |
| M4-class-LDA | 25.7705 | 25.7565 |
| M4-CE direction | 21.3093 | 21.3125 |
| M4-CE-self | 21.9853 | 21.9974 |
| M4-ref-canvas | 31.2531 | 31.2604 |
| cached native + Pro原尺寸renderer variant | 36.3165 | 36.3608 |
| cached MEAN + 同renderer variant | 35.6846 | 35.6952 |
| cached RCG + 同renderer variant | 35.4736 | 35.4849 |
| cached fine16 + 同renderer variant | 34.9116 | 34.9372 |
| cached fine64 + 同renderer variant | 35.9079 | 35.9317 |

数值来源：[report.json](real_smoke4_46466/report.json):4-26。native/历史controls的原尺寸mask缺少现成直接输出，明确从已封存binary1024依Pro规定bilinear到**实际原RGB H/W**再>0.5重构；是缓存renderer variant，不宣称新执行的native原尺寸生产路径或同当前FP32 producer数值复现。M4内部六行使用同一实际FP32冻结timm-DINOv3-L和同一Pro renderer，matched机制比较不受这个外部baseline边界影响。

## 完整改动与失败环节

paired相对该native original variant：增TP3,837、增FP82,091、删TP69、删FP51,472。[report.json:35-40] 有有用的false-removal和漏目标恢复，但错误包含增量损害完整输出。首例native/paired原尺寸IoU均0；paired删除51,127个FP而没有恢复任何TP，不能据删除数量宣称已解决其身份失败。另三例paired对native都降低IoU。[episode_metrics.json](real_smoke4_46466/episode_metrics.json)的逐例I/U与四类增删为完整依据。

原尺寸逐例paired/native IoU：`0_0_72=0/0`、`1_0_73=.10198/.34059`、`2_0_74=.38511/.60448`、`3_0_75=.39861/.50937`。这些是观察；环境方向消除了自然类别信号、合成shortcut或自然范围校准失败仍是待区分解释，不能从4例直接归因。当前v1没有胜过part-scatter控制或R画布，因此不能声称分离环境与部件散布已经兑现收益。

## 实际执行与数学检查

- 四例均主动执行**17次真实1024编码**：原Q一次、edited Q八次、edited R-canvas八次；六行输出全在真实未修改原Q上推断，无几何/统计回退；全部使用同32个canonical点、同四固定条件和成对正负映射。
- 主法与class-LDA每例复用完全相同数值λR。25—29步的FP64 CE均达梯度∞范数目标1e−8，四例最终梯度约8.74e−10至6.24e−9；没有用LDA冒称CE。真实cut能量/maxflow最大gap约1.1e−11。逐例receipt保存在[receipts](real_smoke4_46466/receipts/0_0_72.json)目录。
- Pro767没有显式写出CE自身偏置的惩罚和self-readout额外尺度；实现约定为共同拟合未惩罚b，CE方向使用主法midpoint/separation，CE-self使用raw `w·q+b`，同Potts/renderer。这个细节已在method.json及receipt明示，供根代理核对；没有把它包装成原文唯一规定。
- 先核对四例source snapshot/encoder/control、原RGB与完整参考mask哈希、原query H/W和未读GT标志，再合并[sealed4.json](real_smoke4_46466/sealed4.json)，然后读取真实原尺寸query annotation与**绑定packet.truth**评分。参考mask来自完整原annotation，不从coverage还原。
- 原annotation最近邻1024与packet.truth有370—2710边界像素差。最终1024统一使用历史绑定packet.truth，nearest-original-GT初始评分保留为`nearest_original_GT_variant_report.json`及相应逐例文件。原尺寸原annotation成绩不变；不是将1024 GT逆插值成原GT。

## 成本与资源

| 出现 | method六行总秒数 | 9次主分支编码秒数 | 编码次数 | 进程峰值RSS |
|---|---:|---:|---:|---:|
| 0_0_72 | 404.90 | 209.13 | 17 | 2.91GiB |
| 1_0_73 | 374.62 | 195.26 | 17 | 2.91GiB |
| 2_0_74 | 362.69 | 187.96 | 17 | 2.91GiB |
| 3_0_75 | 362.68 | 194.32 | 17 | 2.91GiB |

六行method时长中位368.66秒，9次主分支编码中位194.79秒。主分支**编码和**不是主法独立完整部署时间；没有另跑纯paired进程或重新测量native原始部署成本。第一完整进程包含加载/初始化/输出为417.34秒，四例各4线程，CPU、不下载、无GPU。

初始180秒screen在7次编码后停止，没有完整预测，记录保留。根代理澄清有进展首例允许12分钟后完整首例成功。随后按新并行授权保留已有case2进度、另启case3；guard仅在case1/2已关闭完整文件后冻结哈希并停止旧串行尾部，避免重复case3。[tail_stop_receipt.json](real_smoke4_46466/tail_stop_receipt.json):2-14。旧supervisor的`FAILED/-15`由这个授权的受控停止造成，不是算法失败；有效两例有明确partial seal。

四例预测及干预banks均在独立远端namespace：`/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/pro_environment_smoke4_v1`。核心实际DINO编码进程已结束；未启动600例、未按GT改变主方法、未改PLAN/STATUS、未进行GT参数搜索。

当前证据支持保留真实负结果及复用已封存干预banks检查具体失败，**不支持将当前M4-v1直接扩成大规模正结果验证**。组件的false-removal迹象不等于一个新的完整删除方法；未来若改变契约，应由根代理按用户当前指令另行定义与测量。
