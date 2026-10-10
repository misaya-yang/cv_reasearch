# 05：旧参考判别读出与当前困难区域排序复核

2026-10-10。状态：执行中。写入范围仅本目录及对应的 `evidence/local/parallel_signal_20261010/05_reference_readouts/`；1 CPU、无新编码、无MPS、无进程池、无子代理。旧资产及主代理字段只读。

## 可回答问题

1. CPU100/native200 中 ridge、Huber、logistic 等哪些是真实已执行正结果，胜过哪些强控制；其它已执行历史判别读出是否仍有未被简单控制解释的收益。
2. 当前 Deep100/PACO100 的 APD ridge whole/local4 在同例同ROI上的排序增益有多一致；重复照片、同类/同折聚集是否改变结论；有效样本筛选是否造成宏平均误读。
3. 旧配方与当前 soft coverage 适配哪些同义、哪些变化；仅从参考未拟合patch选择IoU读出阈值时，连续coverage、balanced fitting、空间依赖和跨图迁移会带来哪些可验证陷阱。

## 已有数据与对照

- `evidence/local/cpu100_20261006/` 原评分JSON、方法/失败记录与Git中退役实现。
- `evidence/local/difficult_region_signal_20261010/{raw,existing}/episodes.jsonl`、sealed/report与 `study.py`。
- `cv_data/a/joint_role_pilot200_20261010/` 逐例输入身份、labels与封存代码，优先小字段；必要时只读labels。
- `cv_data/a/lvis_reference_discriminant100_20261009/` 与 matched-background 校准的现有参考自拟合/迁移证据。
- 主要同批控制：P0、FoRIS source response、旧whole竞争、同特征双原型、raw/APD ridge、Huber。

## 计划计算

- 对同例有效ROI做AUC/AP/TPR差值、胜/负比例、分位数与固定种子的配对bootstrap；照片或fold/class可识别时做聚类重采样并说明独立性边界。
- 检查episode、query/reference照片和fold/class重复，分别报告有效例及唯一身份数量；按原始已有分布字段给出少量预先有依据的边界检查，不搜索部署阈值。
- 从可用I/U重聚合旧正结果与最强简单控制，记录实际样本数/协议/执行文件，不把方法卡数当实验数。
- 对脚本采样/权重/标签/正则/读出进行逐项核对，并仅对现有coverage统计给出参考标定的最小建议。不生成或更改主候选mask。

## 交付

本目录 `FINDINGS.md` 记录重要中间证据及待整合项，最终 `REPORT.md` 给出来源、已执行/未执行、证据边界与可反驳条件；数值脚本和机器可读收据放对应evidence目录。所有当前200例均已曝光，置信诊断不称独立确认。

## 待整合

主代理继续合法参考读出与query上下文视图研究，本线只提供独立复核与具体陷阱证据，不重复实现候选。
