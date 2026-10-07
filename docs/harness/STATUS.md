# 当前状态

更新：2026-10-07。分数和下一步分别只在 [CLAIM](../research/CLAIM.md) 和 [PLAN](../research/PLAN.md) 维护，本页只记环境。

- CPU 服务器：`ssh -p 25142 root@connect.westb.seetacloud.com`，旧 56464 的克隆，32 核 / 无显卡。旧 56464 和 46466 端点不再使用。
- 磁盘（2026-10-07 晚）：系统盘 30G 已用 8.5G，数据盘 70G 已用 40G。
  - 1200 例特征缓存 19G：`cvpr_single_ref_20261005_01a10ba9/outputs/confirm1200_conditional_v1/inputs`，
    三个子目录共 1201 例（FP16，已做位置去偏）。fresh600 和固定 600 例都在其中，两者重叠 305 例。
    `/root/claude_store/fresh600_cache` 现在是指向它的链接（原 9.4G 副本逐字节相同，已删）。
  - 固定 600 例的原始 FP32 特征（18G）已删；要用时从图像重提取，约 1.5 小时 CPU。`demo9_extent`（4.1G）被 116 个旧配置引用，保留。
  - COCO 图像与标注：`/root/demo4_cache/data`（6.5G）和 `datasets/ics`（8.1G）；DINOv3 权重在 `/root/demo4_cache/models`。
  - 已删：Astra300 / Pro30 / CPU100 的批量输出（评分报告留在 `kept_reports/`）、旧 demo2 的模型与 ADE 数据、pip 临时文件。
- 固定 600 例清单：`cpu100_20261006_01a1100b/fixed600_evaluation_rows.json`，80 类，四折各 150。
- 更早的状态记录见 [快照](../archive/2026-10-07-entry-docs-snapshot.md)，连接方式见 [SERVER](SERVER.md)。
