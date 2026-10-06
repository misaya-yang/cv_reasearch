# 服务器操作说明

这里只保存环境知识，不保存待办、当前进程或持续运行授权。当前阶段与资源状态见
[STATUS](STATUS.md)，唯一工作清单见 [PLAN](../research/PLAN.md)。
2026-10-06 文档整理没有访问服务器；之前用户报告已关机，现状未核实。

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

这些路径当前是否存在未核实；缺失不授权下载、重建缓存或恢复已删除的本地目录。

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
