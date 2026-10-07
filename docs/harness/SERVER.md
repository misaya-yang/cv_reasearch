# 服务器操作说明

2026-10-06 CPU100任务的环境记录：用户当时为该项任务准备并授权新CPU实验服务器：
`ssh -p 56464 root@connect.westc.seetacloud.com`。本次新连接核实32 CPU、
`cpu.max=3200000 100000`、cgroup内存60GiB。root负责持续监控新作业、回收结果、
检查退出状态并接续审核放行该任务。此处仅记环境与当时资源收据，不构成后续任务授权。
不恢复旧PID/launch/队列。
新独占命名空间为 `/root/autodl-tmp/cpu100_20261006_01a1100b`；CPU实验不使用GPU、
不下载、不切端点。运行时及DINO/图像资产须按当前实际可读状态绑定。

这里只保存环境知识，不保存待办、当前进程或持续运行授权。当前阶段与资源状态见
[STATUS](STATUS.md)，唯一工作清单见 [PLAN](../research/PLAN.md)。
2026-10-06 当前授权与操作以STATUS/PLAN为准。新入口已经用户明确授权并验证：
`ssh -p 46466 root@connect.westb.seetacloud.com`，32 CPU、cgroup 60GiB、无可见GPU。
**旧46466用户已关机；其队列不恢复。当前新入口见页首。下方路径只作资产导航。**
当前命名空间 `/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b`；
旧缓存只作输入资产，不恢复旧PID/队列。

当前Python `/root/miniconda3/bin/python`，`PYTHONPATH=/root/demo4_cache/env`；
现存timm DINOv3权重、源码/位置基底及数据路径已核实，
[详细资产与哈希](../../evidence/local/research_20261006/parallel_review_01a1100b/server_assets.md)。
本轮不下载、不GPU、不新实例；合计线程不超过30且服从实际cgroup内存限制。

## 历史环境

曾用入口：`ssh -p 48002 root@connect.westd.seetacloud.com`。
曾用工作区：`/root/autodl-tmp/cvpr_single_ref_20261005_01a10ba9`。
曾记录 GPU 模式 RTX 4080 SUPER、32760 MiB、12 CPU、约 62 GiB 内存；
无卡模式仅 0.5 CPU / 2 GiB，`nvidia-smi` 可能是占位程序。授权执行时按实际 cgroup 和设备复核。
旧 PID、临时 DNS 地址和“队列正在运行”的快照已退出当前说明。

| 历史路径 | 用途 |
|---|---|
| `/root/autodl-tmp/demo9_extent` | 共享 FoRIS 入口、特征包和证据 |
| `/root/autodl-tmp/demo9_lang` | 其他代理的队列和报告 |
| `/root/autodl-tmp/demo9_transductive_ics` | 隔离清单与历史结果 |
| `/root/autodl-tmp/datasets/ics` | 共享数据 |
| `/root/autodl-tmp/demo4/INSID3` | INSID3 源码与加载器 |
| `/root/autodl-tmp/demo8_local_verification` | 历史 FoRIS/CRF 运行依赖 |
| `/root/demo4_cache` | Python 包、权重、COCO 标注 |

历史 Python 为 `/root/miniconda3/bin/python`；FoRIS 搜索路径曾为：

```text
/root/demo4_cache/env:/root/autodl-tmp/demo8_local_verification/crf_source/src:/root/autodl-tmp/demo8_local_verification/runtime/extensions
```

上表保留历史导航；本轮实际复核范围及路径见当前资产报告，缺失不授权下载或恢复旧队列。

## 授权运行时

按冻结清单及批准的预算执行，报告未知成本。检查设备、配额和所有权；共享特征与权重优先复用。
一个真实输入 smoke 要覆盖完整运行路径和 CRF，再完成该项评测。CPU 评分不需要保持 GPU 忙碌。
只处理已核对命令行和启动标识的自有进程，不按宽泛进程名杀任务或修改他人的运行快照。
仅在明确授权时关机；关机命令成功不等于租用计费已经停止。

## 已知兼容性问题

- INSID3 的配对/分离 BF16 编码可能改变聚类，保留被比较实现的批组成。
- FoRIS 与 INSID3 都有顶层 `models`、`utils`；既有入口先加载 FoRIS。
- 关闭 `np.load` 句柄，核对缓存是否覆盖全部折和案例。
- 以作业退出状态和预期产物判断完成；SSH 超时、普通 Error 文本或旧预检不能判断实验成败。

历史资源控制器和 launch 文件是实现/追溯材料，不能自行恢复旧队列。
