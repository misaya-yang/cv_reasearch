# 已封存LVIS1400：目标粒度与图误删

标签侧诊断及独立核查完成，未编码、生成新预测或新增mIoU。使用原1400字段和pixel mask，逐例核对query身份、全部原交集以及graph gross edits；GT与实际错误区域相交后再16×16 pooling。8worker耗时3.839秒，session79360 exit0。

| MEAN最终零交集的互斥来源 | 全1400 | 其中小目标591 | 设计1300 |
|---|---:|---:|---:|
| rank已命中，graph后全部丢失 | 51 | 38 | 45 |
| pre已命中，rank后全部丢失 | 3 | 2 | 2 |
| FG前缀已命中，pre/rank无命中 | 33 | 22 | 33 |
| 最早FG前缀无命中 | 8 | 5 | 8 |
| 合计 | 95 | 67 | 88 |

这是按既有前缀定义的标签侧分类。最早FG前缀无命中不等于原始表征没有任何目标信息；先前有交集也不证明身份判断完全正确。

小目标定义为query GT面积<1%，共591例。graph从rank输入删真194012、补真14783、添假1418065。被删真区域的面积加权读数：原s .659260，原unary .661373，最终z .453442，参考guide全图rank .974942。这里是错误区域的面积加权场值，不能当作AUC、概率、分数增量或部署时可读取的筛选条件。

173例query不存在覆盖≥90%的16×16块，全部位于小目标组；该组MEAN zero-IU31/173，其余64/1227。小目标graph删真区域73.6%质量来自mixed块，但全1400最终FN有80.9%质量位于pure块。因此几何混合是部分困难，纯度没有给出模型或mIoU上限。

独立核查使用逐像素坐标bincount重算全部1400 GT覆盖及5600个实际region；3例12区域另以逐块枚举核对。全部baseline交集、graph编辑、FN、token质量、加权场值及六组归并一致。未调用模型、CG或新mask生成。

原图会同时补真和添假，不能从上述误删直接推断关闭graph能涨分。下一同100研究将保持原图与unary，检验参考支持的positive-unary fidelity，并与同位置、同总增量控制比较；不追加照片或恢复全量。

产物：[report.json](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_resolution_loss1400_20261009/report.json)、[逐例诊断](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_resolution_loss1400_20261009/episodes.jsonl)、[独立审查](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_resolution_loss1400_20261009/reviews/independent_audit.md)。
