# 并行审查与实际结果

原Pro文档 `CVPR2027_five_original_segmentation_methods_20261006.md` SHA256：`72afe43605f762d6467df456275b00aa5bd87070cdd10f63fa48766be8a343bc`。
最初审查时仅本地准备，随后用户明确授权新46466服务器并行运行；各单篇报告的当时资源边界不覆盖后来的用户指令。

| 范围 | 已核对的核心结论 | 详细来源 |
|---|---|---|
| 1–3 | FF邻接合成量也可由通用同模式边解释；Huber只更改传播；颜色瓶颈不能删高置信错误种子 | [audit_1_3](audit_1_3.md) |
| 4–6 | 布局v1会删除未命中姿态的正确目标；形状方形正例不证明独有价值；协方差单位特征完整可实现例成立但不证明真实迁移 | [audit_4_6](audit_4_6.md) |
| 7–9 | 重复错误类别可以胜真类；比例估计小目标空输出；二次核首例可由子空间控制解决且shift误删 | [audit_7_9](audit_7_9.md) |
| Pro1–5 | 完整保留机制与renderer；M1/M5不能从末层cache反推raw状态；M3 Ward不可换single-linkage；M4需真实多环境编码 | [Pro审查](pro_contract_review.md) |
| 服务器 | 32CPU/60GiB，无GPU；实际Eva权重318key匹配，raw H20/QKV缓存不存在，现存RGB/参考mask/native basis可用 | [资产](server_assets.md) |

原九项已补真实600完整结果：[结果与资源](../server_prepared_01a1100b/README.md)。九项均未超过同批MEAN，这修正了此前仅有合成正例的质量期待。保留完整负结果；当前新版本只修正已证实局部误删，更多机制来自用户已重开开发授权。

Pro五项代码完成，本轮正在核对实际活动路径、完整输出与CPU成本；不将toy误差、首4结果或cache变体冒称完整原版方法成绩。
