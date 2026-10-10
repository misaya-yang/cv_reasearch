# 给02研究线的追加任务：冻结新global判别规则，检验COCO迁移

主代理已读完02完整报告。旧whole竞争在COCO200负迁移明确，而新的APD-ridge+reference标定+bounded anchor在PACO100优于旧whole、FoRIS及MEAN，Deep也优于FoRIS。现在需要检验这种新信号是否跨到COCO，避免只在已诊断100上停下。继续沿用户“已有缓存、部分成功方法和数据集差异”的授权执行，不开新数据或编码。

请02线读取最新`ROOT_RESEARCH.md`，自行在自己的TASK中写追加任务和冻结协议，完成真实缓存推断及全部200原图输出。新增产物只放自己02目录的新`ridge_coco_validation/`，另写`COCO_RIDGE_REPORT.md`；保留已完成REPORT/旧候选原记录，不改主代理代码或封存结果。

配方严格使用主代理 `evidence/local/difficult_region_signal_20261010/candidate_reference_v1/` 的 **anchor.global**，不扫参数、不加dataset规则或额外标定。正式配置和冻结源在该目录config.json/frozen/；实际函数入口在 `evidence/local/difficult_region_signal_20261010/candidate.py`，核心旧ridge/原投影在 `raw/frozen/study.py`。主代理正在做PACO剩余500的同法验证，入口`validate_paco.py`及其config也是可读参照。

1. 同已有COCO200 manifest与matched_raw完整FoRIS、MEAN、旧whole竞争，不换样本/顺序。使用现有whole R/Q两份原始O24，严格验证实际输入key、profile、payload和数组；任何缺失都报明，不补编码。global128特征是raw whole64 bilinear→单位化→同官方whole语义APD分支，顺序与冻结脚本相同。
2. 参考coverage、128/role quantile拟合子集、旧ridge lambda=.01、未入子集reference pixel-mass IoU阈值、reference类均值差缩放、clip[-1,1]、原图g0+.5h和平局继承FoRIS全部保持。不要把连续coverage硬化。可避免计算被丢弃的Huber，但必须验证ridge与冻结核心一致。
3. 正式COCO生成前，用原PACO pilot第一例做一次同global规则逐位mask对齐（只作实现身份，不另挑样本）。这不是COCO调参。零h必须精确还原COCO完整FoRIS。
4. 全200预测与源身份封存后才读query GT评分。比较同批完整FoRIS63.358900、MEAN64.006821和旧whole60.895815；按实际fold0/1官方40槽位保留缺类，原图为主，不能跟PACO100绝对值混减。记录四动作和实际增量CPU/IO成本、reference校准回退。
5. 输出不论正负均保留，不据此修改公式。报告它支持/否定的迁移命题，以及单张reference标定/anchor所仍不能解决的问题。

资源：只需whole两份，不读取九窗，单数值进程1CPU（若严格复现用2CPU，标明）。无MPS/DINO/新raw写入。主代理的PACO job是2CPU，整机14核/36GiB有足够余量。仍仅通过MD交互；本次调度只用一次唤醒通知传此路径，后续证据写MD即可。
