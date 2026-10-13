# 五数据集逐例机制证据：参考 Mask、真覆盖保留与部件误报


完成，2026-10-11，`codex_m4`。范围严格为 COCO200、PACO600、LVIS1400、原生 PASCAL2000、原生 SUIM2000，共6200现有完整记录。没有混入 Deep100 或旧 PASCAL40/SUIM40，也没有继续评价九窗方法。先封存全队列身份、原六臂整数和官方 class 贡献，再固定20解释案例，最后仅读相应GT、真实annotation与小字段。

**现有小收益主要来自对原覆盖的再分配；参考任务信号存在具体有效例，也会把正确的单臂真覆盖擦掉。** 高参考纯度/好参考拟合并不保证 query 迁移，小参考也不等于前景bank为空。完整队列和反例不支持按数据集、面积、GT形状或同批最优臂指定新规则。本线提供机制证据；统一方案由主Sol/设计线推导，没有生产新预测或公式。

**全队列与指标。** [6200逐例身份/整数账本](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/integer_episodes.jsonl)、[官方逐class贡献](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/official_class_contributions.jsonl)、[库存验收](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/inventory_report.json)。每例保存B/G/R0/C/N/K的原图与1024两个独立frame、真实R/Q照片与crop、Mask SHA和对应预测/小字段文件SHA。每class先池化I/U，再按官方fold与class槽平均；缺槽记0，没有以episode平均或总像素IoU冒充mIoU。

|队列|n|观察/官方 fold-class 槽|FoRIS B|MEAN G|原R0|C|N|同预算K|N−B|N−K|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|COCO|200|39/40|63.358900|64.006821|62.459822|64.080288|64.140667|64.127715|+0.781767|+0.012952|
|PACO|600|264/303|38.460868|39.352045|40.774944|39.664615|39.684486|39.678407|+1.223618|+0.006079|
|LVIS|1400|876/920|43.912229|43.872534|42.771402|44.519548|44.438002|44.435575|+0.525773|+0.002427|
|PASCAL 原生|2000|56/56|55.496178|55.715834|56.534176|56.255810|56.394769|56.386686|+0.898591|+0.008082|
|SUIM 原生|2000|7/7|59.660311|59.549329|59.169070|59.907918|60.095403|60.083051|+0.435092|+0.012352|

上表只比较各行自己的原图协议，不对不同数据集绝对值做因果相减。1024各臂数值全部另存inventory，不与原图混写。B是原完整FoRIS，G是原完整MEAN；这里的R0是原有界ridge全图anchor，尚未包含后来单次finalCRF。C是B/G/R0多数；N只在`B=BG、R0=BG、G=FG、原H>0`区域向C加像素；K在同一个合法新增域里加同样数量、按MEAN自身margin排序。此处读取原封存mask，没有重新执行这些配方，也没有用full-reference/case-deleted校准替换原R0。

**官方收益与覆盖代价。** [全6200固定分母机制摘要](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/mechanism_summary.json)同时保存逐例赢/输、class贡献及两个frame的净TP/FP。以下是N相对B的原图端点差；“净”不能当作gross编辑数。

|队列|逐例IoU提高/下降/不变|净TP|净FP|N−C官方mIoU|
|---|---:|---:|---:|---:|
|COCO|98/98/4|-28,437|-135,888|+0.060379|
|PACO|373/212/15|-33,687|-247,778|+0.019871|
|LVIS|669/656/75|-194,313|-1,215,111|-0.081546|
|PASCAL 原生|1235/726/39|-56,822|-404,863|+0.138959|
|SUIM 原生|1004/960/36|+230,814|-1,822,421|+0.187485|

LVIS的N−C为−.081546点：188个官方class贡献改善、328个变差、404个不变/缺槽。N−K虽有+.002427点官方class均值，全像素却净少1432TP、多1432FP；这不是账本矛盾，而是class加权与像素池化不同。五域N−K均只剩.0024–.0130点，不能把所有N−B收益归于参考新增排序。SUIM N−C七个class均值都改善，同时仍有490/2000例变差。全队列均已曝光，不是独立验证或泛化证明。

**参考Mask在token层的有效监督。** [全6200合法参考支持标量](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/reference_support.jsonl)从5423个既有reference-only小cache读取64²真实coverage和原128/role质量分位fit IDs。没有新拟合或打开O24 payload。coverage是最近邻1024后每16×16块的FG比例；`sum(c²)/sum(c)`描述FG像素所在patch的FG比例，不是视觉feature纯度，也不是FoRIS官方bilinear角色采样。

|队列|R FG面积%中位|FG质量纯度中位|≥90% FG token中位|没有该token的例数|旧分离度≤0例数|
|---|---:|---:|---:|---:|---:|
|COCO|4.906|0.9230|156|0/200|0/200|
|PACO|10.881|0.9446|381|0/600|0/600|
|LVIS|1.361|0.8499|35.5|191/1400|8/1400|
|PASCAL 原生|14.769|0.9637|546.5|4/2000|1/2000|
|SUIM 原生|13.080|0.9464|461|34/2000|4/2000|

原小目标与混合patch困难在LVIS更常见，但“无≥90%token”组里LVIS N−B逐例仍111改善/66下降/14不变，SUIM则9/25/0。这只是已曝光描述，不能由同名参考条件推出统一臂选择。LVIS8、PASCAL1、SUIM4个旧H非正分离度例，R0/C/N/K原图均回到B；零证据是方法退化事实，不能解释成query目标不存在。LVIS还有1例参考角色退化。

FoRIS实际Mask路径已按[官方downsample_mask](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/third_party/foris_official/utils/data.py)逐案核对：raw二值mask先nearest到1024，再bilinear64且>.5；若无FG改nearest64；仍无FG才注入FG坐标均值所在的centroid token。**不能用面积coverage>.5代替该路径，也不能把centroid回退后说成bank为空。** 下述bank coverage仅是实际标签与patch面积的诊断，不保证视觉feature只含这些像素。R0保留连续coverage后，query迁移仍可能失败。旧H校准与同图参考拟合的高分不构成独立query任务正确证据；本线没有改校准，主线既有6380校准修正验收未支持推广。

**案例选择与原图逐阶段追踪。** [冻结20例成员/选择规则](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/diagnostic_case_selection.frozen.json)在本线新query像素读取前固定：每域取最大/最小N−B官方class贡献，再取该class内最大/最小逐例差；追加每域最小R FG质量例、首个旧H零证据例及全部真实Mask-only成员。选择使用既有评估整数只为解释，绝不进入后续推理。[逐例gross add/delete与来源](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_diagnostics.jsonl)保存两frame六臂TP/FP/FN、原query GT、每阶段新增/删除的真/假像素、GT诊断ROI原source/mean/H均值及既有零点计数。20张联系图只展示已经保存的原图mask：绿TP、红FP、黄FN。

下表“加TP/FP；删TP/FP”都是原图gross数，不是净变化。普通正负案例彼此使用不同R/Q，属于跨episode比较，不能当成固定R/Q只改Mask的因果对照。

|固定普通例|B→R0 加TP/FP；删TP/FP|R0→C 加TP/FP；删TP/FP|C→N 加TP/FP|B→N IoU百分点|
|---|---|---|---|---:|
|dev_s1/coco/1/13|0/0；488/15717|38/151；0/0|0/0|+18.523|
|dev_s1/coco/1/36|0/0；4490/10235|282/1234；0/0|0/0|-3.746|
|dev_s1/paco_part/1/75|0/0；411/3022|371/40；0/0|0/0|+36.595|
|dev_s1/paco_part/0/29|0/0；7923/2884|3106/1228；0/0|0/0|-15.947|
|dev_s1/lvis/8/82|0/0；172/5448|161/89；0/0|0/0|+57.961|
|dev_s1/lvis/2/105|3/1378；0/0|0/0；0/964|0/6|-47.566|
|dev_s1/pascal_part/2/63|0/0；2/315|0/0；0/0|0/0|+26.238|
|dev_s1/pascal_part/1/178|0/0；315/251|144/229；0/0|0/0|-12.082|
|dev_s1/suim/-1/1873|0/595；8/361|6/0；0/595|22/7|+14.729|
|dev_s1/suim/-1/428|0/0；45792/96|5695/2；0/0|0/0|-31.093|

**COCO1/13，杯子：** [原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/000213.png)。R/Q都是单个二值组件，R硬bank1254tokens、平均真实FG覆盖98.82%。Q周围非目标包括鼠标和杯座；FoRIS有20438FP。R0删15717FP同时损488TP，C补回38TP/151FP，N没有任何新增，最后N比B少15566FP、少450TP，IoU64.127→82.651。此例收益来自已有共识/删错，不能归为N参考新增。COCO1/36的[原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/000236.png)显示多个碗的语义union：R/Q为3/8组件，真实实例数未知；R0删4490TP和10235FP，C只补282TP/1234FP，N不补，最终少4208TP，IoU16.677→12.931。不能把减少FP直接叫作任务辨别成功。

**PACO1/75，category2097的车辆部件：** [原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/000570.png)。父bbox是实际合法输入crop；父segmentation另来自真实`obj_segmentation`，只用于此处评估。FoRIS3284/3291FP在同父非目标区，图上可见车轮附近错扩。R0删3022FP，其中3021父内；同时删411TP。C用B/G共识补回371TP、只补40FP，最后N保1312TP、303父内FP/6父外FP，IoU26.756→63.351；C→N为0。MEAN单臂有1670TP/225FP，当前组合反而少358TP、多84FP。并非“父内一律拒绝”就能保住目标。该GT仍有7px在parent外，保留原标注，没有裁掉。

**PACO0/29，category2080的盒体目标区域：** [原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/000329.png)。R真实FG覆盖97.66%且旧reference IoU.913，Q同父目标20140px。FoRIS6770/7165FP父内；R0删7923TP，只删2884FP，C补3106TP/1228FP，N没有补回。最终11981→7164TP、7165→5509FP，IoU43.878→27.931。同一“有纯参考/有父内干扰”的输入条件也出现目标严重丢失；不能从PACO1/75推广一个父内清除器。Q target外于真实parent的489px原样计分。最小参考PACO0/40另有R5556 canonical FGpx、20硬FGtoken；Q608px有4组件却只有1个真正part annotation，B490TP/2461FP→N424TP/2388FP。

**LVIS8/82，class1098小目标：** [原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/001631.png)。真实annotation R/Q各1实例，Q目标582px。R只有4个≥90%token、16硬FGtoken，原量分位拟合已吸收99.24%参考FG质量，余下校准FG极少；旧reference IoU.00379不能当query排序收益。B532TP/5559FP，R0删172TP/5448FP，C补161TP/89FP，N不补，最后521TP/200FP，IoU8.663→66.624。[原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/001830.png) 的LVIS2/105则R有2真实实例、Q有1实例78px；R纯度.757、10硬FGtoken，旧separation仅.0031848，FG/BG参考均值−.9076785/−.9108633，旧H有截断放大的条件。R0添加3TP却加1378FP；C拒964FP，N再加6FP，B61.983→N14.418。这是旧任务证据在query上错扩的具体例，不能要求H正号自动覆盖原B。

**PASCAL2/63，person+NECK：** [原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/003443.png)。R98硬FGtoken、平均真实FG覆盖92.69%，Q只149px；真实目标是所选person的1个NECK annotation，不是全person。MEAN仅7TP、FoRIS131TP/430FP；R0删2TP/315FP，其中312FP在真实其它标注部件union中，C/N保129TP/115FP，IoU22.625→48.864。现有C可以凭B/R0两票保留MEAN弱覆盖；不应只把MEAN作为唯一候选源。PASCAL1/178 pottedplant+POT的[原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/003058.png)却R908硬FGtoken、FG覆盖98.82%、旧reference IoU.921；Q盆的可见部分被植物与画面边缘遮挡，R0删315TP/251FP，C补144TP/229FP，最终B449TP/371FP→N278TP/349FP，IoU32.583→20.501。只有9/371原FP落在其它PLANT标注域里，失败主要是目标保留，不能统称父物体误报。

固定PASCAL例的源JSON相关对象均只有`name,bndbox,parts`，**没有完整对象segmentation**。本线以真实名字+真实bbox+目标RLE mask精确锁定所选对象，bbox从未填为parent GT。上述“其它标注部件union”仅是部分可证实的对象区域，允许与目标mask重叠；不能将其父外补集解释成真实parent外背景。各例actual region、重叠与FP计数都显式保留。

**SUIM-1/1873，RO：** [原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/006253.png)。R114硬FGtoken，Q1076GTpx/2组件，实例数未知；G923TP/183FP，B786TP/565FP。R0额外加595FP同时删361FP、8TP；C回拒这595FP并补6TP；N再补22TP/7FP，得到806TP/211FP。但同预算K从C补28TP/1FP，原图IoU比N高.762点。参考新增不是此例最佳排序。SUIM-1/428 HD的[原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/004808.png)有参考353硬FGtoken、真实FG覆盖95.29%、旧reference IoU.863、separation1.6825，Q仍B102875TP/462FP→R057083TP/366FP；C仅补5695TP/2FP，N无法补。最终损40097TP只拒94FP、IoU79.866→48.772。原B独有而MEAN为BG的41853真像素，其既有H零点正证据计数为0；“其它两臂不同意”没有说明它们是误报。N的补集要求B原为BG，结构上不能追回C已删的B-FG真覆盖。

**小参考、分裂结构与零证据的追加固定例。** [原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/001585.png) 的LVIS8/30有2真实R annotation但只13 canonical FGpx，bilinear/nearest64均无FG，centroid回退1token；该token coverage13/256=5.078%、约94.92%为实际BG像素。Q有2真实实例、GT3组件（一个实例本身分裂成2组件）；1837px目标在六臂均0TP，FoRIS/MEAN/R0/C/N各FP5016/845/691/1006/1006。合法Mask信息在角色硬化时被迫集中到高度混合token；回退不是空bank，也未创造正确可迁移目标证据。

PASCAL3/2 aeroplane+WHEEL的[原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/003882.png)中R457 canonical FGpx、2硬token、bank平均FG覆盖52.15%；query真实JSON有8个WHEEL annotation entries，却只有7个GT组件，不能用组件数称8个轮子实例。B1207TP/5732FP→R01151TP/2873FP→C/N1196TP/4625FP；C补45TP同时补1752FP，IoU虽17.245→20.299仍远低R027.802。COCO1/78最小R有23硬token、保71.30%参考FG质量，B4261TP/1215FP→N4172TP/947FP，改善只有+.160点。SUIM-1/504 FV的[原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/004884.png)中R只有198 canonical FGpx、1硬token覆盖60.16%，Q11579px/37组件而真实实例数未知；R0补22TP/6FP但C又删掉，N补204FP/0TP，最后IoU5.427→5.224。薄/分裂/小面积并未对应同一种正确动作。

三个首个非正旧校准例 [LVIS2/13图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/000951.png)、[PASCAL3/258图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/004138.png)、[SUIM-1/375图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/004755.png)都有实际FG bank（6/5/8tokens），旧H退化为0后R0/C/N/K都与B完全相同。目标仍分别有92/337/51006px，B有63/280/12633TP；不能把H=0叫作不存在目标，或把维持B当作任务新增收益。

**唯一真实同R/Q改Mask对照。** [6200中精确Mask-only库存](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/mask_only_inventory.json)只找到PASCAL一组：`dev_s1/pascal_part/3/312 bus+WINDOWS` 与 `3/422 bus+VEHICLE_SIDE`。真实照片分别2008_008469/2008_005360、父crop完全相同；R/Q decoded RGB SHA相同。20例审计核对同一profile及两个role的entry bytes、input/key、继承tensor/payload SHA完全同，未打开rawpayload。APD两任务均false。此处推理合法输入差别是R Mask，query GT随任务改变只供评估。普通10个正负案例不具备这个固定输入条件。

该对照的[完整逐阶段/跨任务GT小收据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/existing_mask_only_pair_diagnostics.json)显示B/C/N两任务原图mask XOR54281/55440/54501px，说明输出确实响应参考任务。**WINDOWS与SIDE并不互斥：42940/94662目标px中42845px重叠，WINDOWS独有95、SIDE独有51817。** 不能把全部SIDE覆盖都称为WINDOWS误报。WINDOWS的[原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/004192.png)里N相对C补1100TP/80FP，同预算K补424TP/756FP，N比K多676TP、少676FP；任务参考在这个区域有具体增量。与此同时WINDOWS的N仍错误覆盖4865个SIDE独有px；SIDE的[原图联系图](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_images/004302.png)里C→N只加83TP却加182FP，N与KTP/FP相同。该正对照支持任务响应和局部辨别，不支持普遍可靠性，更不是新Mask干预实验。

**真实多目标与父mask证据的边界。** [LVIS1400真实instance annotation数/ID](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/lvis1400_genuine_instance_counts.jsonl)依据[实际lvis_val.pkl](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/datasets/ics/LVIS/lvis_val.pkl)，全1400中R624、Q639例有多个同类实例annotation，最大102/113；这是实际annotation计数，selected案例又逐位decode union与保存目标相等。PACO通过category+真实obj_ann_id锁定一个父实例，并union相应part annotation；selected三例的query真实父segmentation与既有GT-only parent cache逐位同。PASCAL的所选parent object和parts真实可核，完整父mask缺失；COCO/SUIM当前输入只提供semantic masks，实例数未知。没有用二值组件、多个RLE片段或bbox填充冒充实例/父GT。

SUIM原生2000的R有35、Q有32例源mask与RGB尺寸不同；其它四域均0。严格沿封存官方alignment将raw GT nearest直接映到每个预测frame，元数据保留原尺寸，未修补输入或改协议。全6200原图query GT面积中位COCO16129、PACO2090、LVIS3972、PASCAL3101、SUIM52850px；这些是诊断标签，不能用来决定推理方法。

**可以支持统一机制推导的事实。** 本线没有把旧报告的面积/纯度相关重新当选择器。需要分别处理：Mask在role采样/混合token中的监督真实性；跨参考到query的任务迁移；多臂单独覆盖的保留；真实同父非目标的辨别；最后组件/像素readout的增删代价。PACO正例说明拒同父错扩可明显奏效，PACO负例和SUIM潜水员说明有好参考也可错删大块真目标；小LVIS说明连续FG质量与硬回退不能凭空生成任务支持；公交Mask-only说明相同视觉特征的任务响应存在，同时同预算新增选择既有成功又有失败。把其中任一项变成面积、calibration或dataset开关都会超过当前证据。

所有source/H/MEAN字段以及R Mask可计算的面积/coverage属于推理时可得信号；按GT分出的ROI真/假、query面积/组件、真实R/Q annotation实例数及parent mask属于oracle诊断。二值R Mask输入不额外提供其真实实例分组。父分割、case角色、class贡献和GT阈值绝不进入将来规则。可审计的已存stage只有B/G/R0/C/N/K及少量continuous fields；本线没有恢复全6200的Part2邻居票、聚类中间态或CRF前全链路。缺少这些保存中间量时只能定位“在哪个有效保存阶段目标丢失”，不能声称已证明DINO表示、邻居选择或CRF中的唯一因果。主线Pro200另做已授权机制验证，本线不重复。

**输入封存、检查与成本。** [reference小cache输入SHA](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/reference_support_inputs.json)、[20例实际读输入SHA](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_diagnostic_input_receipt.json)、[库存protocol](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/protocol.json)保存实际路径、字节、SHA、code及来源。全6200原mask/小字段的来源SHA继承完整6380验收，当前inventory检查其文件存在；本线新开箱SHA的是5423个reference小cache和固定20例的262个输入文件，不宣称重新校验全raw特征池。raw tensor/payload SHA为封存metadata继承，selected20只开箱entry/profile JSON。

|本线脚本阶段|范围/检查|测得wall秒 / CPU秒|新增读取|
|---|---|---:|---|
|inventory.py|6200身份/原整数/官方class槽，297660检查|1.128 / 未单独计CPU|封存binding与scoring JSON；0GT/raw|
|reference_support.py|6200，68184检查|4.060 / 2.810|5423既有小cache，共353,775,576压缩字节；0GT/raw|
|cases.py成功pass|20例两帧六臂、1292SHA/整数/逻辑检查|5.210 / 5.136|20Rmask+20QGT、40案例RGB、120既存预测packet读取（240packed arrays）、77小score array+20coverage、80entry/profile metadata；0raw|
|summarize_mechanisms.py|6200账本，179804检查|0.296 / 0.296|已有JSONL；0GT/raw|

以上均单worker/CPU；新DINO/encoder、raw payload、fit、候选mask、CRF、下载、队列扩展均0，没有dense字段落盘。20联系图约6.6MB，整数/身份/标量账本只保留紧凑文本。源RGB/Mask/raw/旧stage全部保留。[首次metadata序列化失败收据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_attempt1_receipt.json)与失败源单独保存：第1例before stage masks遇到重复validation键，修为显式dictionary合并后按同一冻结20成员重启；额外2个annotation pickle、1R/1QGT及2RGB读取未算作免费命中，失败wall/CPU未捕获。测时只覆盖表内脚本；手工schema核读、联系图视觉查看与文档生成没有单独计时。没有因修复变更成员、score、参数或预测。

|主要封存对象|SHA256|
|---|---|
|[6380来源binding](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/consensus_reference_calibration6380_v1_20261011/source_bindings.json)|`1b301d5d44d7416ea11afda414d97e2d7e74f7723c6bf357f493134309fa1ac1`|
|[6380完整评分](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/consensus_reference_calibration6380_v1_20261011/scored_episodes.json)|`31e4df5b7198cbc936baaae4a405f5afc6e6f4756ab9804fdc4624a95a86b6a3`|
|[6200整数/身份](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/integer_episodes.jsonl)|`35e04a5964e299f93078d6a45757544ad80c788190aa944f825ec919605460a3`|
|[官方class贡献](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/official_class_contributions.jsonl)|`4f296efc902c5f060d9645e8b24fcaa3beb262f49e4a80c8247ab50c22107c6d`|
|[6200参考标量](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/reference_support.jsonl)|`a46cba44ce6fa374e7ea608be033effa5438ce41f5df86bc428acf1df62cfd06`|
|[冻结20成员](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/diagnostic_case_selection.frozen.json)|`87c9d75b49ae67b3f34bc573e784cc492785123c83ad5fd5be74a8514a2aefd5`|
|[20例逐阶段机制](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/case_diagnostics.jsonl)|`b66abe38d347e646b0f48bf30df0233a7491470aa09d7dfca255995c3626439e`|
|[唯一Mask-only pair](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_case_mechanisms_20261011/existing_mask_only_pair_diagnostics.json)|`9cae6c49986f6d758a44bae0aac8bbd9d86ee82b24adc45b228134872a0e41e5`|

旧[04数据集差异报告](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/parallel_signal_20261010/04_dataset_differences/REPORT.md)仅作为已知失败与任务定义历史来源；其中Deep、旧PASCAL/SUIM及九窗结论没有转入本6200主队列，也没有重开已关闭方法。本报告不把source-reference同图拟合、full/case正仿射校准或已有case故事当新query收益。
