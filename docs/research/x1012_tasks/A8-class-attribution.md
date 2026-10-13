# A8：SUIM 类别差距归因

RO 一类贡献原净差距的 **60.08%**，RO+HD 两类合计贡献 **78.53%**。在同一批既有整数指标上，剔除 RO 后，IND0 相对 FoRIS 的 class mIoU 差距为 **−3.834119 pp**；剔除 RO+HD 后为 **−2.474523 pp**。损失明显集中于这两类，但其余类仍有剩余差距。

**这只是事后误差归因。不得据此删除正式基准类别、改变十域评分分母、更新候选选择或改变 C1 确认集。** 正式 SUIM 本批结果仍是 FoRIS 59.739778、IND0 51.507988、差值 −8.231790 pp。

## 来源与口径

- 唯一数据来源：[已完成 A8 原报告](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/x1012_A8/report.json) 的 `records`，筛选 `dataset == "suim"`，读取 `metrics.FoRIS/IND0.I/U` 等既有整数。
- 来源 SHA256：`684faefd4d1c884ebdc48b275b017405c0699276bbad1178fea1c00ef15f53f8`。
- 本批为固定 explore 的 SUIM **1200 例、770 张查询照片、7 个观察到的 (fold, global_class) 槽**；fold 均为 −1，类为 FV、HD、PF、RI、RO、SR、WR。口径为 CLI1024。
- 未重新读取查询 GT、RGB、预测掩码或 raw 特征，未执行实验，未改 A8 原报告。下述数值来自整数重聚合，不使用逐例 IoU 均值替代 class mIoU。

对每个观察槽 s=(fold, class)，先池化全部该槽的整数，再求 IoU：

`IoU_s(a) = 100 × ΣI_s(a) / max(ΣU_s(a), 1)`

`class_mIoU(a) = mean_s IoU_s(a)`

这与 `lab.Meter` 的观察槽等权口径一致；全 7 槽重算精确复现原报告的两个 `class_miou`。排除类仅用于下表诊断，两个方法使用完全相同的剩余槽及任务。

## 排除诊断

|诊断集合|例数|观察槽数|FoRIS class mIoU|IND0 class mIoU|IND0 − FoRIS pp|
|---|---:|---:|---:|---:|---:|
|原完整 SUIM 批|1200|7|59.739778|51.507988|−8.231790|
|仅诊断：排除 RO|1029|6|57.473868|53.639749|−3.834119|
|仅诊断：排除 RO+HD|858|5|54.321197|51.846674|−2.474523|

排除后重新对 6 或 5 个槽等权平均。因此，新的差距不能直接用“原差距减去被排除类的原分母贡献”代替：排除 RO 后，还要将原 7 槽分母下的剩余贡献乘以 7/6；排除 RO+HD 后乘以 7/5。

## 各类对原净差距的贡献

令 `Δ_c = IoU_c(IND0) − IoU_c(FoRIS)`。本批每类只有一个观察槽，故该类对原 7 槽总差距的贡献为 `Δ_c/7`；贡献之和为 −8.231790 pp。

下表“净损失占比”以原净损失 8.231790 pp 为分母。WR 的负占比表示它抵消了其他类的损失；这些占比合计 100%，不表示失败任务的比例。

|类|例数|FoRIS 池化 IoU|IND0 池化 IoU|Δ_c pp|原总差距贡献 Δ_c/7 pp|净损失占比|
|---|---:|---:|---:|---:|---:|---:|
|RO|171|73.335241|38.717426|−34.617815|−4.945402|60.0769%|
|HD|171|73.237221|62.605122|−10.632099|−1.518871|18.4513%|
|FV|173|68.568713|61.551055|−7.017658|−1.002523|12.1787%|
|SR|171|49.622361|44.176621|−5.445740|−0.777963|9.4507%|
|RI|171|61.656638|59.188098|−2.468540|−0.352649|4.2840%|
|PF|170|26.654699|24.748256|−1.906443|−0.272349|3.3085%|
|WR|173|65.103573|69.569339|+4.465766|+0.637967|−7.7500%|

RO+HD 在原 7 槽分母下贡献 −6.464273 pp。剩余类在原分母下贡献 −1.767516 pp，换成剩余 5 槽的诊断分母后得到 −2.474523 pp。FV 和 SR 是剩余损失的主要来源，WR 提供部分抵消。

RO 和 HD 的交集整数分别增加 252766、1211988，而并集整数分别增加 9840573、4616077。同一任务 GT 不变时，`U = GT + FP`，因此这些并集增量就是额外 FP；这两类的主要损失表现为误报扩张，同时 TP 并未减少。这一观察解释误差类型，不构成可部署的按类选择规则。

## 最小复算

以下仅解析原报告中的整数，不打开任何查询 GT：

```python
import json
from collections import defaultdict
from pathlib import Path

source = Path("/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/x1012_A8/report.json")
rows = [r for r in json.loads(source.read_text())["records"] if r["dataset"] == "suim"]
slots = defaultdict(lambda: {a: [0, 0] for a in ("FoRIS", "IND0")})
for r in rows:
    s = slots[(str(r["fold"]), str(r["class_id"]))]
    for a in ("FoRIS", "IND0"):
        s[a][0] += r["metrics"][a]["I"]
        s[a][1] += r["metrics"][a]["U"]

def slot_iou(s, a):
    return 100 * s[a][0] / max(s[a][1], 1)

for excluded in (set(), {"RO"}, {"RO", "HD"}):
    kept = [s for (_, c), s in slots.items() if c not in excluded]
    scores = {a: sum(slot_iou(s, a) for s in kept) / len(kept)
              for a in ("FoRIS", "IND0")}
    print(excluded, scores, scores["IND0"] - scores["FoRIS"])

for (_, c), s in slots.items():
    delta = slot_iou(s, "IND0") - slot_iou(s, "FoRIS")
    print(c, "class delta pp", delta, "original-gap contribution pp", delta / len(slots))
```

本补充没有为事后排除集合新增 bootstrap；没有把排除后的分数作为性能提升或独立确认结果。
