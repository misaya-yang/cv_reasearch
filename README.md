# cv_reasearch

CVPR 2027 研究工作区：保存方法代码、实验协议、正负结果、失败分析和当前候选方向。此仓库是研究记录快照，尚不是已完成论文或统一的一键复现软件包。

## 从这里开始

- [当前结论与证据边界](RESEARCH_STATUS.md)：先了解哪些结果已经成立，以及下一步需要验证什么。
- [研究与 CPU 实验交接](demo_lists/research_decision_20261001/HANDOFF.md)：最新统计选择、区域判别及局部前景量方法的推导和局限。
- [实验协议](demo_lists/RESEARCH_PROTOCOL.md)、[方向筛选历史记录](demo_lists/DIRECTION_SCREENING_20261002.md)：保留当时判断；其中需要修正的解释见当前结论。
- [文献调研表](exa-results/cvpr2027-direction-review-2026-10-02.csv)：22 篇相关工作的来源、读取层次及其对项目的含义。
- [精简实验结果索引](evidence/manifest.json)：原始路径、字节数、SHA256 与导出文件路径。

## 研究目录

| 目录 | 内容 | 入口 / 状态 |
|---|---|---|
| `demo_lists/demo1_sam` | SAM/SAM2 解码复用、稀疏头、质量选择和蒸馏 | [结果](demo_lists/demo1_sam/RESULTS.md)、[实验记录](demo_lists/demo1_sam/GPU_RUNBOOK.md)；已停止方向 |
| `demo_lists/demo2_where_what_decoding` | 区域级语义解码与强像素集成对照 | [汇总](demo_lists/demo2_where_what_decoding/RESULTS.md)、[表格](demo_lists/demo2_where_what_decoding/RESULTS_TABLES.md) |
| `demo_lists/demo3_conditional_grouped` | 条件分组与生成模型中的动态算子 | [当前结果](demo_lists/demo3_conditional_grouped/RESULTS_CURRENT.md)；GPU 工作暂停 |
| `demo_lists/demo4_incontext_seg` | 冻结 DINOv3 / INSID3、区域证据、误差预算与规则探针 | [分析](demo_lists/demo4_incontext_seg/README.md) |
| `demo_lists/demo5_resolution_attention` | 分辨率变化下的注意力修正 | JSON 报告与探针脚本；通用修正假设未成立 |
| `demo_lists/demo6_instance_evidence` | DAVIS 对象身份、传播和缺席诊断 | 探针脚本；身份命中不等于分割质量 |
| `demo_lists/demo7_shared_state_ad` | 增强记忆库的共享成像状态异常检测 | [实验设计](demo_lists/demo7_shared_state_ad/README.md)；候选方向 |
| `demo_lists/demo8_local_verification` | FoRIS 强对照、独立 episode 清单与局部验证器 | [运行依赖与状态](demo_lists/demo8_local_verification/README.md)；开发中 |
| `demo_lists/research_decision_20261001` | CPU 判别实验、局部前景量决策与数学检查 | [交接](demo_lists/research_decision_20261001/HANDOFF.md) |

## 复现与依赖

各 demo 的 README、runbook、配置及脚本是对应实验的入口。部分脚本使用原服务器 `/root/...` 路径，运行前需要配置数据、缓存和模型位置；不应直接批量执行历史 GPU 队列。实验依赖按目录管理，SAM 的已测依赖见 `demo_lists/demo1_sam/requirements-tested.txt`。此快照发布时没有重新执行 GPU 实验，也没有验证所有历史脚本在新环境中的兼容性。

数据集、预训练模型、特征缓存、下载的第三方仓库、虚拟环境、张量、生成图像和原始 `results/` / `logs/` 留在本地，由 `.gitignore` 排除。现有分析中指向原始结果的路径是本地路径；在 GitHub 中可通过 `evidence/manifest.json` 找到相应精简快照。重新生成快照只需要 Python 标准库：

```sh
python3 tools/export_evidence.py
```

导出保留汇总字段，不重算指标；超过 64 项的数组会明确标记为省略，超过 200 KB 的导出文件只记录索引。小型文本日志保留原文。精简快照用于审阅结果，不能替代逐样本数据进行重新统计。原始文件不会被修改或删除。

第三方代码遵循各自许可证和版权声明。SAM 的保留源码及来源见 [THIRD_PARTY_NOTICES.md](demo_lists/demo1_sam/THIRD_PARTY_NOTICES.md)。本仓库未为全部内容统一指定新的许可证；使用第三方派生代码前应核对来源及许可。
