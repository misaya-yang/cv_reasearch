# 统计口径版本说明

本文只澄清估计对象，不改任何冻结预测、原始I/U、旧报告数值或方法选择。

## 当前合同

对共享support/query照片形成的episode连接组重采样，2000次，RandomState(0)，numeric fold/e/c稳定排序。每次对重复抽中的episode按次数累计各类I与U，再对本次出现的类别计算I/U宏平均。没有抽到的类别在该次宏平均中省略。

若某类只有一例，该例被抽中w次，则wI/wU=I/U，重复次数约掉是该统计量的正确结果。若某类有多例，应计算Σw_eI_e/Σw_eU_e；此时不能省略每例重复权重。所有臂使用相同抽样。

当前通用实现：`validation121_locked/statistics_randomstate.py`。之前`photo_group_bootstrap.py`和`photo_group_statistics.py`同属照片组→类I/U估计量，但使用default_rng，随机序列不同；原结果保留。RandomState补充另列，不能说逐draw相同。

## 历史口径

1. `corrected_real20_statistics.{md,json}`及`correct_real20_statistics.py`使用20个逐例IoU配对差的重复均值。这保留了抽样重复episode的权重，但估计量与当前按类ΣI/ΣU再平均不同。点估计因原20每类恰好一例而相等，不意味着bootstrap分布相同。历史把权重抵消统一称为bug不适用于当前合同。
2. `new40_class_statistics*`、`repri40_class_statistics*`和`new40_statistics_protocol.md`曾按类别或关联类别组抽样。这是早期协议，后来被用户明确的照片组primary替代。
3. 像素质量加权AUC的逐例宏平均，是另一种已明确标注的目标量；其照片组bootstrap仍保留重复episode均值权重，不进行类I/U聚合。AUC区间不能填入mIoU栏。

canonical ledger中的当前mIoU统计引用照片组→类I/U文件；历史文件保留作为协议演变记录，不再称为当前合同的修复版本。固定OOF监督预测的bootstrap不包含重新训练带来的不确定性。
