# 九窗接受争议区的参考信号：固定只读诊断

2026-10-11。

**状态：whole保存场1400 source-arm已完成；真实窗固定同头诊断已完成。**

真实窗同固定参考头在Deep接受少数区增加了排序（宏AUC .7545对whole .6401），LVIS增幅较小（.7175对.6918）；PACO反而较差（.6956对.7637）。原零点不能直接替换既有M：分别会丢39.29% / 30.23% / 57.00%的已有TP，AU4仍保1,191,816 / 197,286 / 1,980,719个FP。新视野存在条件互补，但尚未证成跨三域的统一完整判别组合。

已有whole ridge提供一些额外真／假排序，但原参考零点不能同时保证追回少数真支持和保护现有正确覆盖。full-reference和case-deleted校准复用同一个query ridge，并未产生新的query任务排序。真实窗阶段只改变查询RGB视野，保留同一whole参考、固定ridge头、APD和旧校准；其唯一统计为原接受支持视图上的最大H。没有产生新预测、mIoU、CRF或DINO前向。

## 输入、真实参考视野与可用字段

完整读取旧九窗队列的1400份source-arm：1300个不同episode ID，Deep同100有原／快两份输出。所有旧九窗在已完成6380校准队列中都精确匹配R/Q decoded RGB、两mask hash、原crop和原尺寸；PACO重复物理输入以dataset+episode ID消歧。这里的物理输入一致不等于局部读出所用encoded R一致。九窗另裁R在下面单列；诊断H始终来自与原whole ridge完全一致的whole R。

|队列|source-arm n|同whole R|另裁local R|保存逐窗reference/query pooled-margin|旧非正分离度|
|---|---|---|---|---|---|
|LVIS原600|600|364|236|200|5|
|Deep原100|100|100|0|0|0|
|Deep快100|100|100|0|0|0|
|PACO快600|600|587|13|0|0|

全1300不同episode均有保存whole ridge128、source64、mean64及旧/full/case校准。真实逐窗连续source/H没有保存；全队列固定同头CPU阶段从实际O24重新计算瞬时H并释放。保存的逐窗reference/query margin只覆盖LVIS最初200：它按整窗原mask池化，并非每组件reference判据。其余1200 source-arm不存在这个小字段，未填零或伪造为逐组件参考证据。6380队列其余5080行不在已绑定真实九窗队列中，本次不扩队列。

LVIS/Deep的whole guide是旧原图FoRIS mask按原bilinear/strict>.5回到canonical1024；PACO另外保存了真实canonical FoRIS端点。本次所有ROI统一使用旧guide定义，不把回渲染guide AUC称为原图性能。以前52.005138、37.344028、34.971403等原图完整输出分数保留其原口径，本次没有新完整输出评分。

## 固定ROI、分母与信号

`M=(2*accepted_votes>coverage)`；`A=(accepted_votes>0)`；接受少数区是`A & ~M`；四覆盖全接受区是`coverage==4 & accepted_votes==4`（AU4）。每像素原覆盖数只为1/2/4，无支持仍进入原投票分母。truth按旧脚本由raw query mask直接nearest到1024；计数为canonical唯一像素，不是窗口发生量、原图像素或mIoU权重。

AUC是每例ROI内FG/BG像素排序，再宏平均；必须同时有FG/BG才有效，未满足记null。另存pooled pixel AUC、精确零点TP/FP、均值与有效分母，二者不能混成一种权重。AUC固定.0005分箱仅作统计压缩；ridge先按例正仿射minmax到[-1,1]，没有阈值搜索。所有H零点计数来自未分箱的`H>0`。guide FG/BG和同R/另裁R是预声明成员分层，不是方法选择器。

`H=clip((saved ridge128-reference threshold)/reference separation,-1,1)`在128网格先算，再bilinear到1024；非正分离度清零。旧/full/case在相同ridge、正分离度且未clip的范围只是正仿射变换。排序差异来自非正分离度清零、截断平台和插值顺序；不算新query信息。whole source是原source64按原minmax到[0,1]后映射到[-1,1]；whole mean为`2*mean64-1`；两者是现有whole连续基线，不冒充九窗证据。组件context取原支持组件最大margin。

|队列|接受少数TP / FP|AUC有效例|M TP / FP|AU4 TP / FP|AU4 AUC有效例|
|---|---|---|---|---|---|
|LVIS原600|2,355,649 / 9,278,811|551|23,521,901 / 13,658,522|9,031,671 / 3,281,377|367|
|Deep原100|467,882 / 1,927,930|100|2,415,571 / 2,016,376|517,242 / 307,677|73|
|Deep快100|492,434 / 2,218,125|98|2,565,972 / 2,363,158|553,368 / 369,684|73|
|PACO快600|21,873,324 / 36,977,239|580|62,317,217 / 41,022,467|14,427,677 / 6,128,630|309|

## 保存whole信号实际有多少额外区别

|ROI / 宏AUC信号|LVIS原600|Deep原100|Deep快100|PACO快600|
|---|---|---|---|---|
|accepted_minority / ridge_rank|0.7009|0.6503|0.6411|0.7647|
|accepted_minority / H_old|0.6918|0.6496|0.6401|0.7637|
|accepted_minority / H_full_reference|0.7009|0.6502|0.6410|0.7647|
|accepted_minority / H_case_deleted|0.6996|0.6503|0.6410|0.7647|
|accepted_minority / whole_source|0.7032|0.6372|0.6191|0.7329|
|accepted_minority / whole_mean|0.7179|0.6095|0.5960|0.7540|
|accepted_minority / component_context|0.6809|0.5889|0.5866|0.6462|
|accepted_minority_guide_fg / ridge_rank|0.6047|0.6279|0.6137|0.6778|
|accepted_minority_guide_fg / H_old|0.5945|0.6296|0.6157|0.6777|
|accepted_minority_guide_fg / H_full_reference|0.6046|0.6289|0.6150|0.6778|
|accepted_minority_guide_fg / H_case_deleted|0.6044|0.6290|0.6149|0.6778|
|accepted_minority_guide_fg / whole_source|0.5804|0.5949|0.5709|0.6148|
|accepted_minority_guide_fg / whole_mean|0.6040|0.5825|0.5600|0.6523|
|accepted_minority_guide_fg / component_context|0.5558|0.4859|0.4775|0.5381|
|accepted_minority_guide_bg / ridge_rank|0.6808|0.6667|0.6459|0.7366|
|accepted_minority_guide_bg / H_old|0.6749|0.6659|0.6444|0.7355|
|accepted_minority_guide_bg / H_full_reference|0.6808|0.6665|0.6455|0.7368|
|accepted_minority_guide_bg / H_case_deleted|0.6802|0.6667|0.6455|0.7367|
|accepted_minority_guide_bg / whole_source|0.7003|0.6609|0.6269|0.6999|
|accepted_minority_guide_bg / whole_mean|0.7224|0.6328|0.6033|0.7437|
|accepted_minority_guide_bg / component_context|0.7036|0.6218|0.5996|0.6405|
|accepted_unanimous_coverage4 / ridge_rank|0.7831|0.6392|0.6588|0.7544|
|accepted_unanimous_coverage4 / H_old|0.7595|0.6387|0.6583|0.7525|
|accepted_unanimous_coverage4 / H_full_reference|0.7831|0.6381|0.6577|0.7545|
|accepted_unanimous_coverage4 / H_case_deleted|0.7842|0.6381|0.6577|0.7544|
|accepted_unanimous_coverage4 / whole_source|0.7760|0.6286|0.6427|0.7468|
|accepted_unanimous_coverage4 / whole_mean|0.7944|0.6522|0.6619|0.7834|
|accepted_unanimous_coverage4 / component_context|0.5398|0.4878|0.4940|0.5129|

PACO minority ridge宏AUC .7647高于已接受组件context .6462，说明whole参考判别在这一区域有额外关联；条件化guide=FG后降低到.6778。Deep快minority ridge .6411、guide=FG .6137，同样只有有限区分。LVIS minority ridge .7009，接近whole source .7032且低于whole mean .7179。组件context在已接受域的零点全部为正，其已有筛选不能再在同域拒绝。AU4组件context接近随机，但ridge仍有一定排序；这两种现象不能省略其真目标分母。

|队列|full H>0 minority TP / FP|minority TP保留率 / FP保留率|若用full H删M，丢TP|full H>0 AU4 TP / FP|
|---|---|---|---|---|
|LVIS原600|385,329 / 496,916|16.36% / 5.36%|10,846,621|5,733,187 / 757,030|
|Deep原100|110,837 / 234,776|23.69% / 12.18%|913,828|368,727 / 133,604|
|Deep快100|103,025 / 239,979|20.92% / 10.82%|1,025,298|382,098 / 156,282|
|PACO快600|4,594,380 / 2,239,399|21.00% / 6.06%|31,345,396|8,556,323 / 2,005,481|

因此既有whole校准零点不能直接充当M保护／少数恢复的统一门控。full H会在PACO minority只保21.00% TP；若据它删M会丢31,345,396 TP，同时AU4仍保2,005,481 FP。LVIS/Deep同样有明显正确覆盖损失。这里是固定ROI的条件统计，并未创建或评价一个删M候选。6380原完整校准实验也没有支持推广校准修正。

保存的整窗reference-margin在LVIS最初200的minority宏AUC .6679（179有效例），零点留1,053,483 TP / 2,687,791 FP；query-window margin为.6479，留1,050,532 TP / 2,933,164 FP。在这200的AU4，它们的宏AUC只有.5392/.5285（117有效例），两者都保全部2,969,683 TP及950,064 FP。真实参考池化margin有定义，也不能从这一子集证明能拒绝一致误报。

## 真实窗同头唯一诊断

冻结规格：`evidence/local/nine_window_composition_20261011/native_diagnostic_protocol.frozen.json`，SHA `413ba8ec9c88ebd0e89b2dec81e3541b1e6971986e58cdd1f8adeadb9060c0ef`。candidate仍与这份诊断规格分开；本分支不生产候选。固定ridge只复原旧whole R的quantile128/role、balanced weights、y=2c−1、lambda=.01和未惩罚intercept，APD沿旧apd_applied。所有真实Q窗也保持原whole `raw64 bilinear128→normalize→fixed APD normalize→coef→H128→bilinear512`，再按真实窗口坐标回到1024。

`L_A(p)=max{H_v(p): 原接受组件在真实窗v支持p}`。A中集合非空；无支持像素不假称低分。在旧H非正分离度例，所有H_v精确清零，不把缺身份证据写成负证据。只报告该一个L_A、原零点、预声明ROI及whole-H/L_A符号四格；不添median、view阈值或融合配方。原组件另裁R分层仅记录原九窗生产条件；固定H头始终同whole参考，隔离的是Q视野变化。

|队列 / ROI|L_A宏AUC (有效例)|whole H宏AUC|L_A>0 TP / FP|TP保留率 / FP保留率|
|---|---|---|---|---|
|LVIS原600 / accepted_minority|0.7175 (551)|0.6918|458,590 / 1,438,628|19.47% / 15.50%|
|LVIS原600 / accepted_majority|0.7720 (571)|0.7507|14,279,873 / 3,920,008|60.71% / 28.70%|
|LVIS原600 / accepted_unanimous_coverage4|0.7626 (367)|0.7595|6,583,666 / 1,191,816|72.90% / 36.32%|
|Deep快100 / accepted_minority|0.7545 (98)|0.6401|184,204 / 309,927|37.41% / 13.97%|
|Deep快100 / accepted_majority|0.7783 (100)|0.6791|1,790,247 / 921,881|69.77% / 39.01%|
|Deep快100 / accepted_unanimous_coverage4|0.7772 (73)|0.6583|449,356 / 197,286|81.20% / 53.37%|
|PACO快600 / accepted_minority|0.6956 (580)|0.7637|3,140,143 / 1,997,406|14.36% / 5.40%|
|PACO快600 / accepted_majority|0.7043 (583)|0.7374|26,798,662 / 8,890,535|43.00% / 21.67%|
|PACO快600 / accepted_unanimous_coverage4|0.7088 (309)|0.7525|8,087,823 / 1,980,719|56.06% / 32.32%|

以下四格是精确唯一像素TP / FP，`whole−`与`native−`均包含零点；不是四种新方法。whole−/native+直接量化新视野提供的零点互补，whole+/native−同时保留其失去的真支持和拒绝的假支持。

|队列 / ROI|whole− / native−|whole− / native+|whole+ / native−|whole+ / native+|
|---|---|---|---|---|
|LVIS原600 / accepted_minority|1,789,503 / 7,660,193|61,296 / 193,105|107,556 / 179,990|397,294 / 1,245,523|
|LVIS原600 / accepted_majority|8,231,999 / 9,415,562|827,683 / 530,276|1,010,029 / 322,952|13,452,190 / 3,389,732|
|LVIS原600 / accepted_unanimous_coverage4|2,225,549 / 1,998,613|328,704 / 151,222|222,456 / 90,948|6,254,962 / 1,040,594|
|Deep快100 / accepted_minority|288,209 / 1,823,832|94,331 / 122,542|20,021 / 84,366|89,873 / 187,385|
|Deep快100 / accepted_majority|602,948 / 1,317,936|359,729 / 223,851|172,777 / 123,341|1,430,518 / 698,030|
|Deep快100 / accepted_unanimous_coverage4|82,888 / 158,914|80,430 / 46,777|21,124 / 13,484|368,926 / 150,509|
|PACO快600 / accepted_minority|16,763,211 / 34,295,191|510,993 / 462,401|1,969,970 / 684,642|2,629,150 / 1,535,005|
|PACO快600 / accepted_majority|29,130,966 / 30,315,985|2,400,809 / 1,229,220|6,387,589 / 1,815,947|24,397,853 / 7,661,315|
|PACO快600 / accepted_unanimous_coverage4|5,128,607 / 3,809,517|665,073 / 257,721|1,211,247 / 338,394|7,422,750 / 1,722,998|

|队列 / guide=BG ROI|原TP / FP|L_A>0 TP / FP|whole H>0 TP / FP|
|---|---|---|---|
|LVIS原600 / accepted_minority_guide_bg|823,362 / 5,946,800|106,113 / 863,160|86,478 / 794,009|
|LVIS原600 / accepted_majority_guide_bg|719,716 / 2,292,253|194,315 / 513,827|127,551 / 463,545|
|LVIS原600 / accepted_unanimous_coverage4_guide_bg|162,404 / 363,122|56,228 / 77,902|34,538 / 62,561|
|Deep快100 / accepted_minority_guide_bg|401,025 / 1,778,273|149,930 / 244,586|75,941 / 183,847|
|Deep快100 / accepted_majority_guide_bg|647,967 / 1,014,053|403,080 / 342,431|236,781 / 238,084|
|Deep快100 / accepted_unanimous_coverage4_guide_bg|82,840 / 107,294|61,239 / 55,384|29,363 / 35,705|
|PACO快600 / accepted_minority_guide_bg|6,192,895 / 22,709,923|613,060 / 699,091|547,399 / 478,131|
|PACO快600 / accepted_majority_guide_bg|3,040,685 / 8,181,615|551,029 / 489,654|408,004 / 329,395|
|PACO快600 / accepted_unanimous_coverage4_guide_bg|287,839 / 505,273|105,309 / 59,719|78,491 / 33,233|

|队列 / 参考视野|n|minority L_A AUC (有效例)|minority L_A>0 TP / FP|AU4 L_A>0 TP / FP|
|---|---|---|---|---|
|LVIS原600 / same_whole_R|364|0.7246 (343)|343,725 / 798,250|5,201,564 / 834,266|
|LVIS原600 / cropped_local_R|236|0.7058 (208)|114,865 / 640,378|1,382,102 / 357,550|
|Deep快100 / same_whole_R|100|0.7545 (98)|184,204 / 309,927|449,356 / 197,286|
|PACO快600 / same_whole_R|587|0.6929 (569)|3,136,493 / 1,989,651|7,975,413 / 1,978,364|
|PACO快600 / cropped_local_R|13|0.8372 (11)|3,650 / 7,755|112,410 / 2,355|

固定例 `dg18-0000`：

|ROI|原TP / FP|L_A>0 TP / FP|whole H>0 TP / FP|
|---|---|---|---|
|accepted_minority|20,637 / 60,832|18,867 / 21,350|18,333 / 33,337|
|accepted_majority|79,242 / 32,456|77,402 / 22,174|78,099 / 25,413|
|accepted_unanimous_coverage4|0 / 0|0 / 0|0 / 0|

固定例 `dev_s1/paco_part/0/0`：

|ROI|原TP / FP|L_A>0 TP / FP|whole H>0 TP / FP|
|---|---|---|---|
|accepted_minority|0 / 197,563|0 / 0|0 / 0|
|accepted_majority|23,513 / 267,105|97 / 0|3,479 / 0|
|accepted_unanimous_coverage4|0 / 40,518|0 / 0|0 / 0|

固定例 `dev_s1/paco_part/0/14`：

|ROI|原TP / FP|L_A>0 TP / FP|whole H>0 TP / FP|
|---|---|---|---|
|accepted_minority|381,794 / 18,490|207,831 / 1,350|6,453 / 0|
|accepted_majority|244,553 / 2,912|163,665 / 1,036|8,934 / 0|
|accepted_unanimous_coverage4|0 / 0|0 / 0|0 / 0|

全部匹配旧零H LVIS成员（未根据新分数挑例）保存在`native_fixed_head/report.json:zero_evidence_examples`；非正分离度例不会被full-reference修正替换成本次native头。

PACO固定`0/14`的局部正例被全600统计明确限制：其381,794个少数区TP中native零点补回207,831，带入1,350 FP，但全PACO少数区native零点只保14.36% TP，whole原零点保21.03%。PACO少数区native-only（whole≤0、native>0）仍有510,993 TP / 462,401 FP；同时whole-only包含1,969,970 TP / 684,642 FP，说明两场有条件互补，不能以较差的单场排序证明任意组合都无用，也不能据一个正例选择native场。

三域M且guide=BG的原TP为719,716 / 647,967 / 3,040,685，native正号分别只保26.999% / 62.207% / 18.122%；不能把低于零点直接当已知错误而擦掉独有真覆盖。AU4宏AUC有效例为367 / 73 / 309，其真／假像素均保留在全部收据内。主Sol独占封存AUC对的query-photo及R/Q连通单位配对区间审计；本线没有bootstrap或阈值／系数选择。

复原的1300份saved whole ridge128最大数值误差 `5.10702591327572e-15`；旧reference calibration最大误差 `0.0`，均小于冻结1e−12。所有匹配保存的reference score另做bitwise比较。H头使用同一个whole参考编码，不将组件额裁R影响归因于H的query视野。

主Sol的[唯一配对区间审计](../../evidence/local/sol_validation_20261010/nine_window_paired_auc_intervals.json)已完成，直接读取封存标量，零GT/raw/fit/pred读取或生成。以下为accepted-minority的 L_A−whole H 宏AUC差；query-photo与R/Q连通两种单位均支持LVIS/Deep正方向和PACO负方向。它仍是已曝光队列内条件排序区间，不能当新完整mIoU或Mask-only因果结论。本分支未重复bootstrap。

|队列|有效paired例|ΔAUC|R/Q连通95% CI|query-photo95% CI|
|---|---|---|---|---|
|LVIS原600|551|0.025730|0.011818 / 0.038851|0.012498 / 0.040158|
|Deep快100|98|0.114402|0.089341 / 0.138674|0.089878 / 0.141043|
|PACO快600|580|-0.068046|-0.081575 / -0.053934|-0.083590 / -0.052155|

## Whole独有支持的旧完整R0补充边界

设计分支追加冻结了一次必要ROI（[冻结规格](../../evidence/local/nine_window_composition_20261011/whole_only_diagnostic.frozen.json)，SHA `26cddde071429481cdd2c968a6983e1b167c0a8ca534b5e7088bae25ee925727`）。只读取6380 source_bindings绑定的旧完整`anchor.global` mask：LVIS600来自`consensus_lvis1400_dual_frame_v1_20261011`；Deep/PACO700来自`consensus_dual_frame1020_v1_20261011`。旧R0使用原`B符号×|2*minmax(source64)−1| + .5*H_old`，严格>0、精确tie继承原B。没有重新拟合、重算native头或生产候选。

ROI是`B_only=B & ~A`，A为任何接受组件支持。canonical B严格承接九窗的原图FoRIS回渲染guide，R0取真实保存的cli直接场mask；original B取旧完整FoRIS原图mask，A按原bilinear/strict>.5回原尺寸，R0取真实保存的原图直接场mask。原图GT仍直接nearest raw mask到该帧。两个帧是明确各自成员域的诊断，不把原图直接R0冒充cli回渲染，也不把这块过滤计数当全方法mIoU。

|队列 / 帧|B-only TP / FP|R0保TP / 拒FP|TP保留率 / FP拒绝率|损TP / 剩FP|
|---|---|---|---|---|
|LVIS原600 / canonical|987,426 / 5,965,999|642,883 / 3,201,322|65.11% / 53.66%|344,543 / 2,764,677|
|LVIS原600 / original|264,295 / 1,575,113|173,675 / 839,157|65.71% / 53.28%|90,620 / 735,956|
|Deep快100 / canonical|61,807 / 753,385|45,806 / 412,070|74.11% / 54.70%|16,001 / 341,315|
|Deep快100 / original|61,807 / 753,385|45,806 / 412,070|74.11% / 54.70%|16,001 / 341,315|
|PACO快600 / canonical|9,442,502 / 14,210,845|6,294,600 / 8,541,815|66.66% / 60.11%|3,147,902 / 5,669,030|
|PACO快600 / original|156,973 / 445,952|101,528 / 290,851|64.68% / 65.22%|55,445 / 155,101|

旧R0在Deep B-only确实拒绝412,070/753,385 FP，同时损16,001/61,807 TP；在PACO原图拒290,851 FP但损55,445 TP。它比盲删整个B-only更能保留whole独有真支持，也仍有明确的真目标代价，需作为完整组合约束。有效FG/BG例数、按例宏比例、同R/另裁R和positive-calibration/zero-H分组均在小收据中。

全部1300旧B_original与九窗B_original逐位一致。旧Bcli与旧guide的XOR分别为LVIS615,339、Deep0、PACO3,769,620像素，LVIS/PACO全部600例都有边界差异。5旧zero-H成员的原图R0等于B，保全617 TP/23,564 FP；canonical少84 TP/863 FP只来自Bcli/guide帧别名，不能称为零头提供了语义拒绝。

补充诊断 11.35 wall / 10.79 CPU秒，单线程／worker、7,803 内部检查，零fit/raw/native/CRF/encoder/新mask。`whole_only_roi/verification.json`独立核对逐例／汇总分区；report SHA `33b9f78d57fdf5c6d34042a061fd503a89b5815c0f894e4994e05d7640995eb5`。本分支不再增诊断前提；候选由设计分支冻结、执行分支生产与完整评分。

## 收据、成本与限制

产物根：`evidence/local/nine_window_signal_20261011/`。`analyze.py`、`protocol.json`、`bindings.json`、`source_receipt.json`、`episodes.jsonl`、`report.json`保留固定逻辑、输入SHA、逐例标量和有效分母。保存whole诊断耗时 383.57 wall / 381.24 CPU秒；单CPU线程、单worker、零raw payload读取。内部记录计 107,000 项检查，另完整复原旧mask端点；`verification.json`独立核对319,332个标量分母／分区条件。whole报告SHA `1ff1b2a7bf0bf6060328d4a7037f463d353c71dc0014c559a43229e3e1495c36`。

真实窗CPU阶段成功1300 pass耗时 1862.58 wall / 4520.92 CPU秒；4 Torch线程、单worker。实际复原旧reference ridge 1,300 次，全部算本次CPU工作，不当旧免费cache fit命中。加载 14,255 次已有raw payload，按压缩文件大小计加载量约 214.00 GiB（SHA流读取另有开销，不冒充实测物理SSD I/O）；有效窗头应用 11,655 次。阶段细账及全部消费payload/entry/tensor SHA在`native_fixed_head/source_receipt.json`。IO计时包含于fit/query大阶段，不相加成总耗时。原始features和dense中间场均逐例／逐窗释放，最终只保标量。

首轮在第一条标量写出前遇NumPy bool序列化错误；仅改builtin bool，失败代码及工作收据保留。额外工作是1次R拟合、1次whole重放、9次窗头应用、11次payload加载（171,153,032 B）和1次GT读取；未及时发出失败wall/CPU计时，标unknown，不填0或计成旧缓存命中。该工作额外于成功1300 pass，详见`native_fixed_head/attempt1_work_receipt.json`。

运行源码及实际读取诊断规格的同SHA冻结副本别名见`native_fixed_head/immutable_sources.json`；`native_fixed_head/native.frozen.py`保留实际代码。最终native完整报告SHA `0662354654f1ecfa5702e5041128fd688d2d0167e38b84a543314076d082d960`。

`verify_native.py`追加核对 43,918 项捕获SHA／冻结预期、3,900 项原始逐窗key／文件／R-box以及 195,810 项逐例／汇总分母、guide分区、零点四格和AUC宏均值，不重新读取raw payload或GT。`paired_auc_records.jsonl`及其receipt导出真实R/Q照片、RGB/crop/size、binding index与同ROI两场AUC（约2.46 MB），专供主Sol唯一配对区间审计。

保存whole阶段峰值RSS 2.08 GiB；native阶段 1.59 GiB。本分支全部收据、代码和逐例标量约 61.1 MiB，没有保存raw复制、dense评分场或新完整预测。此为缓存追加CPU工作量，不能当冷端到端耗时。

这些输入都是已曝光开发队列；条件AUC与零点TP/FP不等于完整输出增益或新独立验证。[无GT输入库存](../../evidence/local/sol_validation_20261010/nine_window_mask_switch_coverage.json)证明本队列同R/Q仅切换Rmask的组数为0；PACO570个照片标识对应580个实际Q输入、595个实际R/Q输入，不能据此主张已通过同图mask切换的因果鉴别。没有按dataset、query GT或native结果选择方法。没有改变主state、原始缓存或他线文件。候选性能、Strong成功和可部署规则仍需其实际完整输出证据，本只读分支不替代该验收。
