# W71离线运行包：命令与验收状态

2026-10-16。按用户要求交付本地全自动命令，供Opus检查。**正式队列未启动，48小时期限尚未创建。** 用户最新资源预算：总内存25GiB，CPU/GPU持续负载预算80%。

## 启动

在仓库根目录执行下列依赖链。任一步失败，后面的启动不会执行；因此可以让它在断网后完成自检。

```bash
cd /Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch
~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w71_offline.py prepare && ~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w71_offline.py check && ~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w71_offline.py start
```

这条命令会冻结当前被审阅的代码、检查输入和来源哈希、运行14个真实输入的一致性预检；通过后才创建48小时时限并后台运行。无需在线模型、网络或后续交互。`caffeinate`防止闲置休眠；需接电保持机器开机。启动后终端可以关闭。

查看状态、主动停止、在原期限内恢复：

```bash
~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w71_offline.py status
~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w71_offline.py stop
~/miniconda3/envs/aidemo/bin/python -B scripts/x1012/w71_offline.py resume
```

运行中不要再执行`prepare`；已有session会阻止替换执行包。修改源码后必须重新prepare/check，过期的预检回执不能启动新版本。重复start不会产生第二个监督器；恢复不重置48小时。

## 已验证与边界

- 9项工程测试通过：精确同分阈值、因子统计、原子记录与损坏检测、持久化截止时间、网络拒绝、进程组清理、内存并发预算、资源等待上限、失败重试及独立后续任务继续。
- 科学计算路径已完成14例真实输入预检：血管111锚点6/6逐像素一致；正式头旧特征8/8的SH和SH+精确复现，新MPS特征通过既定余弦/L2及mask差异门槛。没有读取query GT或查看新干预臂准确率。
- 最初预检发现层列表顺序与编码器实际返回顺序不同，已修复为按网络顺序映射，随后真实输入检查通过。
- 用户后来补充25GiB预算，最后版本增加了RSS进程树加Metal分配的保守内存核算，并通过对应工程测试。**该最终包的完整14例预检由上述启动链再次执行；没有把旧版本的PASS冒充新包PASS。**
- 已封存408个机制任务、四个官方域的完整清单，原W30已有记录会校验后复用，实际剩余43,380例。输入清单共91,919条不同物理路径已做存在性/大小/mtime核对；RGB及mask内容在处理时验证已有身份hash。
- 核心机制、统计、阈值和数据范围见[固定计划](W71_weekend_offline_PLAN.md)。既有方法头和W57未修改，没有提交Git。

## 自动化行为

预测逐例原子写入、整批/整域封存后评分；评分也有逐例检查点。每阶段至多两次自动重试，并发上限从4降到2、1；持续资源不足120秒会退出该次尝试，不无限等待。失败阶段及其依赖评分有明确记录，独立任务继续。监督器在无心跳30分钟或总48小时截止时终止自己的进程组。

系统可用内存与25GiB作业预算共同决定并发；总预算按进程树RSS加Metal驱动分配保守计数，可能重复计算共享内存，为下一次编码预留3GiB。GPU用同步计时和间歇控制80%平均占空，**不声称能限制单个kernel的瞬时利用率**；CPU最多4个worker×2线程加主进程2线程，并在整机CPU采样超过80%时暂停提交。

自动恢复限于冻结程序内的可重试失败，不会自行改科学方案或修代码。系统关机后不会自行开机；重新运行时仍使用原48小时截止时间。

## 输出与时间

输出目录：`../cv_data/a/x1013_W71_offline/`。运行后查看`REPORT.md`、`result.json`、`progress.json`、`stages/`及`logs/`；原W30记录继续放在原目录。

真实预检中，MPS编码活动时间约1.73秒/张，血管每例两层16臂预测约4.5–8秒；这不是完整评分工期。按固定BFS顺序和24张内存缓存模拟，W30剩余任务约需39,846次新图编码，按80%占空折算编码部分约24小时，另加头部、评分、IO及资源等待。原4–8小时的机制预算偏保守；提前完成即结束，不补任务占满48小时。

## Opus审阅入口

- [监督器、冻结与官方队列](../../../scripts/x1012/w71_offline.py)
- [资源控制与离线编码](../../../scripts/x1012/w71_runtime.py)
- [固定因果对照与计分](../../../scripts/x1012/w71_mechanism.py)
- [工程测试](../../../scripts/x1012/test_w71_offline.py)

可重复工程测试：

```bash
~/miniconda3/envs/aidemo/bin/python -B -m unittest discover -s scripts/x1012 -p test_w71_offline.py -v
```
