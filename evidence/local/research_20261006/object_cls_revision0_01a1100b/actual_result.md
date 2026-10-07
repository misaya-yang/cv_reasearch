# Object CLS revision0：单例真实完整链阴性封存

**决定：关闭这项固定 revision0，不扩展到 4/24/600，不调整参数，不计独立新方法。** 已完成的单例显示，CLS 没有恢复目标，同裁剪 patch 均值给出几乎相同的完整场修正；这不是数学等价证明，也不能排除所有 CLS 路线。新 crop 编码真实存在并已计成本。[actual_single/report.json:24-76](actual_single/report.json)、[actual_single/report.json:115-145](actual_single/report.json)

## 固定合同与历史边界

实际全局 CLS 和 RGB crop 重新编码此前已存在：旧源码直接取最终 norm 后 token0，并使用已知 RFG/BG；旧 Q 构造含 GT/pred box 特权。因此本项仅为合法全覆盖及对称中性隔离的 revision0，而非新增 observable。[scripts/diagnose_object_cls_dev241.py:39-55](../../../../scripts/diagnose_object_cls_dev241.py)、[scripts/diagnose_object_cls_dev241.py:263-285](../../../../scripts/diagnose_object_cls_dev241.py)

旧 1200 unchanged 诊断的 CLS margin AUROC 0.433959 [0.340099,0.527200] 是适用构造的强反证；不同 natural reference 的条件信号 0.641561 改变了参考内容/背景且仍含 GT boxes，不能外推为本项合法完整方法增益。[RESULTS.md:1342-1353](../../../../RESULTS.md)、[RESULTS.md:1716-1736](../../../../RESULTS.md)

本项只读取一张 R 原 RGB、完整原尺寸 mask、一张 Q 原 RGB及已有 native q/r 与同 MEAN 基场。Q 采用固定相邻 Ward 的 4/8/16/32 四个完整分区，每个 token 每尺度一票；RFG 用完整前景，RBG 仅来自完整 mask 已知背景。所有 R/Q 视图采用同一中性隔离、原图物理 aspect、最长边 128、居中 padding，不用 Q GT 或 GT/pred box 筛选。[src/ics/methods/object_crop_cls.py:52-79](../../../../src/ics/methods/object_crop_cls.py)、[src/ics/methods/object_crop_cls.py:132-215](../../../../src/ics/methods/object_crop_cls.py)

实际 encoder 调用冻结 Eva `forward_features`，要求 FP32 `(1,69,1024)`，取 CLS index0，跳过四个 register，并同次获得 final patch tokens。强控制同时包括同裁剪全 patch 均值和几何 coverage 加权 patch 均值，使用相同 RFG/RBG banks、margin 和完整读出；另有 nativeROI 均值与辅助 FG-only 行。[src/ics/methods/object_crop_cls.py:234-261](../../../../src/ics/methods/object_crop_cls.py)

receipt 的继承 `producer_binding.producer` 字符串带 `RGB1024`，描述的是原 native cache/同一冻结 checkpoint 的历史 producer 名称，不能作为新 crop 尺寸证据。新 crop 尺寸以实际调用 `view_size=[128,128]`、源码 `forward_features` 的 69-token shape 断言和本报告的新增 forward 计数为准；新 crop 未复用原 1024 特征代替编码。[actual_single/receipt.json:52-68](actual_single/receipt.json)、[src/ics/methods/object_crop_cls.py:246-261](../../../../src/ics/methods/object_crop_cls.py)

固定 margin 为 `cos(QROI,RFG)-max_BG cos(QROI,RBG)`；主/同 crop 控制均以 `0.05*(四尺度平均 tanh(m_crop/0.07)-四尺度平均 tanh(m_native/0.07))` 修正完整 MEAN。推导的绝对场改动上限 0.1，因此 coarse 基场位于 [0.4,0.6] 外的 token 不会翻转；这是有限修正的能力边界，不是概率校准或优化保证。先 bilinear 场到 1024 严格 >0.5，再 binary1024 bilinear 到真实 Q 原 H/W 严格 >0.5。[src/ics/methods/object_crop_cls.py:89-129](../../../../src/ics/methods/object_crop_cls.py)

## 合成完整正负例

确定性非 DINO encoder 的合法 RGB 测试验证四完整分区、aspect/padding、调用计数与完整场上界。正例靠全局颜色排列线索恢复 64 个弱目标 token、删除 64 个 distractor，完整 work TP 12996；同类 180° 外观变化负例把原先正确的 64 个目标 token 全删，完整 work TP 0。toy patch 描述符人为混淆，不能据此声称 DINO CLS 胜过 patch 均值。[check.json:1-23](check.json)

## 已完成真实单例

固定首例 `0_0_72` 为已暴露样本，未以 GT 选样或调参。远端独立 namespace 为 `/root/autodl-tmp/cvpr_prepared9_20261006_01a1100b/runs/object_cls_revision0_single1_v1/single`，已有完整 work/original masks 与 fields；本地保存 receipt、seal、质量 report、descriptors 与 ROI ids。seal 保存所有预测/场/hash，并标记推断未打开 Q GT；score 先核 seal/source hashes 后读真实原尺寸 annotation 与既有 work packet truth，原尺寸 GT 没有从 1024 逆插值。[actual_single/sealed.json:1-15](actual_single/sealed.json)、[scripts/run_object_crop_cls.py:45-81](../../../../scripts/run_object_crop_cls.py)、[scripts/run_object_crop_cls.py:99-167](../../../../scripts/run_object_crop_cls.py)

**实测成本：** CPU 2 线程，69 个逻辑视图（1 RFG +8 RBG +60 QROI）中 12 个 QROI 成员完全重复，故 48 个唯一 QROI、57 次新的物理 128 RGB encoder forward。`exact_view_reuse=0` 仅计 RGB+coverage hash 层复用，12 次成员复用已在此之前避免调用，不代表 69 次 forward。方法链 36.532 s、同 MEAN 重算 1.069 s、完整进程 46.091 s，supervisor 48.543 s，峰值 RSS 3,239,079,936 bytes（3.017 GiB），正常 exit0。没有 GPU、下载或零前向声明。[actual_single/receipt.json:52-57](actual_single/receipt.json)、[actual_single/receipt.json:2057-2073](actual_single/receipt.json)、[actual_single/supervisor.json:12-19](actual_single/supervisor.json)

|固定行|原尺寸 IoU|新增 TP|新增 FP|删除 TP|删除 FP|1024 IoU|1024 删除 FP|
|---|---:|---:|---:|---:|---:|---:|---:|
|同 MEAN|0|0|0|0|0|0|0|
|CLS 主方法|0|0|0|0|1511|0|3623|
|同 crop 全 patch 均值|0|0|0|0|1504|0|3608|
|同 crop masked patch 均值|0|0|0|0|1430|0|3423|
|CLS FG-only 辅助|0|0|0|0|335|0|755|
|nativeROI 均值控制|0|0|301|0|311|0|800|

上表为实测完整输出，全部六行 work/original IoU 均 0，无目标恢复。nativeROI work 另新增 FP 700。CLS 相对最强全 patch 均值仅多删除原尺寸 FP 7，没有质量增益证据。[actual_single/report.json:6-113](actual_single/report.json)

CLS 与全 patch 场最大差 `1.1350043e-4`，与 masked patch 最大差 `0.0023280`；数值近似，非代数同一。封存后 48 个重叠层级 ROI 的 margin/GT purity Spearman 分别 CLS 0.0831、全 patch -0.00389、masked -0.1140，仅辅助诊断，不是 48 个独立样本或方法增益。[actual_single/report.json:115-145](actual_single/report.json)

**解释与未知：** 单例否定了这一固定 revision 在该暴露例上恢复完整目标或胜过同信息简单控制的期待；不能估计总体泛化、类均值增益或所有 CLS 的适用性。完整 masks 已产生并评分，主链无回退。结合既有严重反证，保持关闭，不因只删 FP 就扩样或再调 tanh/权重/分辨率。

本次Git提交包含该单例的摘要、评分/receipt/seal和ROI ID；较大的descriptor NPZ与完整mask/field输出仍保留在本地/服务器。

## 并行元数据冲突

本报告写入时 `method.json` 已被并行编辑为 `actual_quality_unmeasured`、未运行及当前 commit 请求未授权。未覆盖该文件或其授权文字；上述 actual receipts 是已完成事实，主代理已收到冲突通知，应将状态与本报告核对。此报告不创造任何后续远端运行授权。
