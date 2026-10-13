# 原参考Mask到官方hard role：6200现有输入的唯一别名库存

完成，2026-10-11，`codex_m4`。**770对合法同R、不同任务、不同原Mask中，D64完全碰撞为0，联合(N64,D64)碰撞也为0。** 本次没有找到自然任务别名见证，不能把现有6200的首次全路径不可恢复位置定在hard64。COCO没有符合条件的配对，0/0不构成充分阴性。零结果不证明硬化没有损失，也不证明任何后续阶段可恢复query目标。

唯一假设来自[此前20例机制报告](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/FIVE_DATASET_CASE_MECHANISMS_20261011.md)中LVIS8/30：13个canonical FG像素经centroid回退成为一个仅5.078%真实FG覆盖的role token。此前只证明参考标签与patch面积高度混合，没有证明另一真实任务Mask压缩成相同完整有效角色。本次只查这个自然别名问题；不新增Q任务、Mask干预、算法、头、阈值或队列。

**范围先封存。** COCO200/PACO600/LVIS1400/原生PASCAL2000/原生SUIM2000=6200，排除Deep和旧P/S40。[冻结protocol](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/protocol.frozen.json)、[全6200 R-only身份index](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/reference_index.frozen.jsonl)、[冻结同R候选组](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/candidate_groups.frozen.json)在打开任何R Mask前写好。同一R要求decoded RGB SHA、实际crop、输入尺寸和raw encoded profile/input/key/payload/tensor身份全部相同；4679个既有R entry及1个profile的metadata字节与binding逐项核验，raw payload未打开。没有按原Mask得分、query GT或结果筛成员。

主语义别名要求同一数据集global task/真实part类别不同、原合法R二值Mask不同。相同类别重复Mask不计；同CID换selected parent/instance或不同支持Mask也不自动要求不同Q语义答案，只单列support variation。不同R crop或编码身份不计。跨COCO/LVIS命名空间仅有3个物理/encoded重合组，数字CID不同不能自动证明语义类别不同，已列于冻结index并排除主语义分母。

|队列|现有n|同R且不同task/Mask候选组|涉及episode|唯一必要R Mask|有效语义variant对|D64碰撞|N64＋D64联合碰撞|1024预处理碰撞|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|COCO|200|0|0|0|0|0|0|0|
|PACO|600|40|89|84|49|0|0|0|
|LVIS|1400|78|163|159|84|0|0|0|
|PASCAL原生|2000|157|354|322|173|0|0|0|
|SUIM原生|2000|340|1088|741|464|0|0|0|

候选615组涉及1694个episode，重复episode先按同类别＋原Mask去重，得到770个不同语义variant对，另3个SUIM同类别不同支持Mask对。后3对D及联合alias均为0；没有PACO同CID异parent语义alias计入主结果。其余4506个episode已参与身份/分组库存，因无同R异Mask比较条件无需转换，不能说重新读取或转换了全6200张Mask。

**精确转换和已有字段复用。** 候选引用的1334个reference-only小cache仅包含`reference_score,coverage,fit_ids,deleted_score,leverage`，检查ZIP成员名后确认没有保存D/N hard数组；未打开其中score数组。[既有20例支持实现的冻结副本](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/previous20_hard_support.source.frozen.py)保持原SHA，并逐案核对stage、FG token数和centroid位置。真正的D数组执行的是从[实际官方utils源码](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/third_party/foris_official/utils/data.py)AST原样提取的[冻结downsample_mask函数](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/downsample_mask.function.frozen.py)，没有重新实现centroid的Torch求均值。

raw二值Mask严格`>0`，依实际`set_reference`先Torch nearest到1024再`>.5`；D64由bilinear64、`align_corners=False`和严格`>.5`产生，空时nearest64，再空才Torch centroid fallback。N64为该1024 Mask另行nearest64的严格`>.5`数组。1306个唯一合法原Mask各转换一次；所有实际原二值array SHA/形状均与manifest一致，当前canonical coverage数组SHA又与既有R-only标量cache一致。按相同原Mask复用小角色结果，1024场逐例释放，未落盘dense数组、Mask图或预测。

D/N都保留64² uint8数组的SHA和token数；对比采用内存中的实际packed role字节完全相等，不只比较FG数量或以相似度/阈值替代。原Mask不同但canonical1024相同属于更早预处理别名；canonical不同但D相同才归为新增D压缩别名。本次两项皆0。[逐R角色hash记录](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/reference_role_records.jsonl)和[全部773个variant对](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/candidate_variant_pairs.jsonl)可逐项复核；[alias案例收据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/alias_cases.json)明确为空，而非跳过输出。

**具体零结果。** 下列每域例为冻结index顺序的首个有效语义对，只用已有记录作解释，不按hard结果挑案例。完整原Mask、canonical、D、N SHA均保留在variant对JSONL。

|队列|同一实际参考|两任务/episode|原MaskSHA前12位|D64 SHA前12位|D FG token数|
|---|---|---|---|---|---|
|PACO|train2017/000000194499.jpg|2440 / 2390；dev_s1/paco_part/0/2 / dev_s1/paco_part/2/11|47a21589165c / d92cbaaa4a41|227a56235a41 / f816e15e9258|579 / 454|
|LVIS|train2017/000000429335.jpg|243 / 708；dev_s1/lvis/0/19 / dev_s1/lvis/8/53|e07c66c35c8d / 3637a90a51d2|6577c0530d9c / 99e08e1da14c|76 / 238|
|PASCAL原生|2008_008354|bird+TAIL / bird+TORSO；dev_s1/pascal_part/0/3 / dev_s1/pascal_part/0/66|d9a675618bf7 / f7d0728c67fd|05d50992e767 / 3a2063f3bde6|97 / 1757|
|SUIM原生|f_r_1063_|FV / PF；dev_s1/suim/-1/0 / dev_s1/suim/-1/1507|a79ff9ece766 / fcf36cd02317|76b0b62214df / 1b36f00014ad|122 / 292|

更接近原假设的自然小参考对是LVIS `4/11 class146` 与 `6/148 class746`：原Mask17/38px、canonical104/240px，前者centroid fallback、后者bilinear成功。两者D64都只有**1个**FG token，但D完整hash不同（`a571d7852cc5a0f8fa4a75d528dfcb99443634ef2e24a4ba42e14e41b5affe06`、`53833a46825a7fea02dc1c026566d6d6340107533af5e7f543f5548c39ae2397`），N分别0/1个token，未产生任务alias。token个数相同不是role数组相同。候选共1691个bilinear正常episode、2个nearest fallback、1个centroid fallback。

触发假设的LVIS8/30实际R是`train2017/000000142822.jpg`，class410；6200里只有该一个R task成员，所以不能将它变成“已有自然两任务完全碰撞”故事，也没有为它构造第二Mask或Q任务。原先13px/混合token的事实保留，但完整任务别名假设此次未获支持。

**原FoRIS的全路径事实。** 执行线独立源码审计 [Mask压缩切点报告](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/docs/research/FIVE_DATASET_EFFECTIVE_READOUT_20261011.md)、[计算图收据mask_compression_cut](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_effective_readout_20261011/path_evidence.json)确认原模型不是只消费一个D64：[APD:304](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/third_party/foris_official/models/foris.py) 与stage1 gate:430另用N64；stage2 FG/hardBG prototype:485、NN reference标签:591、seed prior:656使用D64。D相同而N不同可能改变整个R/Q APD投影；即使旧episode APD flag同，也不能从理想标量单位化推断float32后续聚类严格逐位同。

连续coverage是final之后trace诊断，不是原segment预测的额外R-Mask旁路。原像素Mask在set_reference的1024预处理中及D的centroid回退参与；当N/D已相同后，原模型没有额外使用它恢复任务差别。**在ordered R集合、同R/固定Q真实输入与特征、basis、参数、初态/随机状态和确定数值执行都相同的条件下，联合(N,D)相同是原FoRIS预测相同的充分条件。** 这是一条计算图条件证明，本库存没有自然联合alias见证，更没有读取Q GT/特征或验证固定Q两任务预测相等；它也不适用于软coverage MEAN/ridge/composition。返回的完整reference `mask_model`诊断tuple仍可不同，充分条件只针对预测。

**可下的结论。** 此6200自然库存在所有770个主语义variant对保住至少一种完整hard角色差别，没有提供“此处已不可恢复全部任务区分”的实证见证。小Mask仍可能被混合、遗漏边界或标签化得不可靠，但这些部分损失不等于两个完整有效输入表示完全相同。COCO无匹配分母、跨命名空间未证实语义对应、未固定Q、未跑下游模型，都已保留边界。本次唯一假设完成后关闭，不扩大搜寻、扰动Mask或创建新查询任务。

**来源SHA与成本。** [R-only metadata收据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/reference_metadata_receipt.json)记录43402个metadata身份检查；[R Mask/纯R-cache实际读取收据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/conversion_input_receipt.json)记录所有PNG文件SHA、binary array SHA及header来源。准备1.067s wall/0.756s CPU；唯一转换pass 9.907s wall/9.662s CPU，14851检查，单CPU线程、单worker。0 query GT、Q特征、raw payload、encoder、fit、模型/预测、CRF、下载、新样本或阈值扫描。只保存身份、hash、整数和源代码，不写主state，不改旧atlas/Mask/raw或已封存预测。

|封存对象|SHA256|
|---|---|
|[protocol](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/protocol.frozen.json)|`37828aca70babb442fe6c7c37c20976587937e6bf593e90c55712db20d274975`|
|[6200 R-only身份](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/reference_index.frozen.jsonl)|`69327fdcef949e5838e93755c38d296311e0d628778fa7b6dd249e01a9c60bd4`|
|[role记录](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/reference_role_records.jsonl)|`9bd29f4f1516f5ca48a6b45adf35de331d4bd4fd019e71accef9548392a905b5`|
|[variant对](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/candidate_variant_pairs.jsonl)|`8eda97e23a00f5fb8429f6cb4f9dd9978aa3329d784118c50c511de00d26645c`|
|[官方downsample源码](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/third_party/foris_official/utils/data.py)|`37211b2f55391694023cad806cdc9ec210944d32346ac1de0f9502df39731b53`|
|[沿用20例支持源码](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/reference_mask_hard_alias6200_20261011/previous20_hard_support.source.frozen.py)|`2fcf36a9dcaf557284180cc7d3d999ac167af677589b95554d887d16eed22ff6`|
|[原FoRIS源码](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/third_party/foris_official/models/foris.py)|`982cf4aa91ac90370151c353efb37240d7d4d5c35a0dfe9cc9509cd554c85fa9`|
|[执行线计算图事实](/Users/misaya.yanghejazfs.com.au/paper_project/cv_reasearch/evidence/local/five_dataset_effective_readout_20261011/path_evidence.json)|`2130502b347e42630a38a45f5808e2a3fc7016ebe406c7225e96adeb9fb8bd9e`|
