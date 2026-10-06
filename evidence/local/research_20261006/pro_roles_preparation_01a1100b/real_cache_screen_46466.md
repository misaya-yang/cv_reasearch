# M3 真实缓存成本 screen 与固定24例复用结果

结论：真实4096活动分支已跑通，四行比较能够以约1GB RSS运行；24例尚未显示留出机制相对all-role的独立增益，不能据此称M3已成功。没有启动600例。

## 身份、授权与终点

- 本轮由根代理明确授权连接新46466并执行1例→固定4例→固定24例成本/复用验证；不恢复历史授权、旧队列或旧PID。
- 入口 `ssh -p46466 root@connect.westb.seetacloud.com`，实测cgroup CPU32核、内存64424509440 bytes。只启动本方法自有任务，无GPU、新编码、下载、远端Git、Git变更或共享PLAN/STATUS修改。
- 快照 `/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/code_318a5564bafd`；Python `/root/miniconda3/bin/python`，PYTHONPATH=`/root/demo4_cache/env:<snapshot>/src`。
- 所有推断均为workers1、threads4，taskset固定4CPU（128–131），CUDA_VISIBLE_DEVICES为空。自有process-group RSS监视上限8GiB，异常只停止自有组；三个推断、两个评分退出码均0，未触发资源停止。
- 本结果是**1024工作空间缓存适配版**，并非Pro从RGB/FP32原生编码到原H/W的完整版本。旧q/r/cov/full-FoRIS-score缓存与四个历史完整控制按bound600_v2逐例绑定；额外编码0。原尺寸第二段renderer未接入，不能跨口径搬分数。
- 所有数据是既有暴露/复用缓存，24例仅为`bound600_v2/rows.json`原顺序前24例，未按新法成绩选择；12类、24照片关联组。四例使用`bound600_v2/smoke4.json`，两组不合并为独立28例。

## 一例活动分支与四例 smoke

首例目录 `runs/pro_roles_screen1_v1`：4096 token、原Ward/Hungarian活动分支，无fallback；推断2.892896秒、4979次Hungarian，含进程启动的守卫窗口5.020256秒，自有RSS采样峰808337408 bytes。达到每例60秒内成本门槛后继续四例。

四例目录 `runs/pro_roles_smoke4_v1`：4/4活动，fallback0/4；四行推断守卫窗口16.074638秒，评分0.752754秒，自有RSS采样峰821764096 bytes。单例3.047/3.811/3.362/3.450秒；调用4979/6073/1953/11527次。

| 四例1024 class-summed mIoU | 结果 |
|---|---:|
| native | 36.316536 |
| M3 heldout | 32.755127 |
| M3 all-role | 33.983810 |
| M3 mean | 34.100029 |
| M3 zero | 35.052496 |
| stored MEAN | 35.684576 |
| fine64 | 35.907872 |

该小组结果反向；四例只核完整运行/成本，不单独决定整个方法失败，也不据此改参数。

## 固定24例结果

目录 `/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/pro_roles_reuse24_v1`。全部预测、字段、收据先封存，再调用独立score；fallback **0/24**，即24例全为活动结构分支，不能把收益归给native回退。

| 1024 class-summed mIoU | 结果 |
|---|---:|
| native | 63.882831 |
| M3 heldout | 66.297014 |
| M3 all-role | 66.217865 |
| M3 mean | 65.909079 |
| M3 zero | 63.020175 |
| stored MEAN | 66.744966 |
| RCG | 66.064126 |
| fine16 | 65.990688 |
| fine64 | 69.217451 |

| heldout相对 | 点差pp | 照片组配对95%区间 | 逐例上/下/平 |
|---|---:|---|---|
| all-role | +0.079149 | [-4.341902,0.738532] | 3/5/16 |
| mean | +0.387935 | [-4.200525,0.948760] | 4/9/11 |
| zero | +3.276839 | [-3.484143,5.432655] | 6/11/7 |
| stored MEAN | -0.447952 | [-6.998995,1.801211] | 5/18/1 |
| native | +2.414184 | [-4.524824,4.344609] | 6/17/1 |
| fine64 | -2.920437 | [-10.056541,0.543644] | 5/18/1 |

这些区间来自共享评分器的2000次RandomState(0)照片关联组配对重采样；不是独立确认。类别汇总主指标与逐例得失不是同一个量，因此6例上涨、17例下降与正的class-summed点差可以同时出现。小样本、稀疏类别重采样的宽且不对称区间不足以声称成功或等效。

heldout相对native四类像素变化：add_TP188554、delete_FP224842、delete_TP96858、add_FP201821。存在纠错，也有明显误删和新增FP；不能只选正变化讲机制。

## 实际成本与600外推

- 四行缓存推断（含其共享加载、Ward、所有matching/control、读出、封存）守卫总窗口 **117.032091秒**；runner封存时长116.754725秒；逐例计算窗口合计114.277201秒。
- 单例均值4.761550秒，median3.842932秒，p95 **10.308473秒**，最大 **12.062678秒**。平均Ward0.548910秒、平均matching1.467331秒。Hungarian共418420次，最大单例115708次。
- 自有进程组RSS按250ms采样峰 **999362560 bytes**；worker RSS high-water最大952590336 bytes。两种口径分别记录，均远低于8GiB。总score守卫窗口1.505644秒，RSS59142144 bytes。
- **计算性外推，非测量：**同一workers1/threads4与同批机器/共跑条件，按117.032091×600/24得2925.802279秒≈**48.76分钟**，仅为四行缓存推断。不是单独heldout部署时间，也不是新DINO编码成本；样本尾部、IO、候选数与共享CPU竞争可改变耗时。不同worker并发加速未经测量，不承诺线性提速。
- 本轮已满足有限成本screen交付；没有自动启动600例，没有为了保住方法名改seed、κ、候选树、角色数或温度。是否扩大由根代理结合本次不确定的机制结果与整体任务决定。

## 可重放证据与版本

远端每目录均保留`config.json`、`evaluation_manifest.json`、`inference_manifest.json`、`sealed.json`、`predictions/`、`fields/`、`receipts/`、`score/report.json`、`score/episode_metrics.json`、`score/counts.npz`与bootstrap抽样。外部守卫收据位于`runs/<run>.infer.guard.json`和`runs/<run>.score.guard.json`，守卫日志同目录。

24例身份SHA256：

- evaluation_manifest.json：`24e932c0691a4fd33a6514591122796f0348cbdf664a9e851324e0e35813e913`
- sealed.json：`27f1b5ceb0b45c2d4a830f7ad6bba06b2ddfa6134e6c26a37d41b2d4c6f99a27`
- config.json：`f37c3d0e18e37d0a799b2c44af0e9cb4bc86f42d42e0ce7c1e4a66626338c8ed`
- score/report.json：`3aef5bbd3ee2d7282a2baa246707010907206c56157aa3802959467b59ddf710`
- 方法源码：`cfdd881339202fa52205138e82afb26c4bd30af23924735d39a182f9ff1e1957`
- runner源码：`c4f8a6ecdd765d81b7e6cc445d98edbc8e0b7a715a06b65e8c2ae50392621c38`

执行期间未发现需修复的真实合同bug，未变更远端算法快照。报告数值从上述远端JSON经只读SSH摘要核对；未下载图像、模型、原始缓存或预测二进制。
