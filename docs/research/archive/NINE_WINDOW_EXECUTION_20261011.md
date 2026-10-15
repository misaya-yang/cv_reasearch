# 真实九窗缓存执行与验证

2026-10-11。**固定1300例、三臂、canonical/原图共7800张最终位图已封存、评分和独立核验；九窗方法路线按最新人类指令关闭，不加变体、不扩批。原始缓存保留。**

主候选未完成 Strong 目标：原图 LVIS600 49.748627 低于旧区域52.005138，Deep100 36.426043低于旧区域37.344028；PACO600 从旧区域34.971403回升到38.348501，仍低于完整FoRIS38.460868和匹配whole ridge40.774944。Pro200固定六臂作为独立机制验证顺序执行，不延续九窗方法路线。

## 输入与冻结协议

执行证据根为 `evidence/local/nine_window_execution_20261011/`。`input_binding_receipt.json` SHA `8c18e34d4eb12a121638af156a6374a7dce18fade8047246e1db1db153b15fc0`；`input_bindings.jsonl` SHA `c4e51ba23defae7182685e7aa99f2224ac6e5b68627a1f1adb4965bb00d67d57`。复用已经验收的重建，不重跑旧整份诊断、M∪(B∩A)、两个CRF位置或6380校准。

三组同原R/Q decoded RGB、mask、crop、源manifest和封存SHA：LVIS600/600 query照片、Deep100/100、PACO600/570。额外组件R crop分别236/0/13。共14065不同raw输入元数据完整、缺失0；O/24 float32[4096,1024]，固定profile `0a555915a7972480b74fcce11c876a79c4e776c93db8573d31f8b0e01c1c6158`。绑定阶段只检查元数据和存在性；消费时逐次核payload/tensor SHA。空photo ID以已验证RGB哈希识别，本1300无空query ID。

唯一protocol SHA `c620555f9b5ee3d30b3547ba3f5c6300db94d6dd8b919e1e183bbca6c844f83c`，`protocol.frozen.json`保持其字节。真实9个512窗x/y∈{0,256,512}，coverage1/2/4固定，拒绝组件不减少分母。A=accepted>0，M=2accepted>coverage，D=A&(M xor B)。valid源使用 `(M&B)|(B&~A&R0)|(D&((W>0)|(L>0)))`；L仅原accepted支持位内取真实窗H最大值。invalid源三臂均用 `M|(B&~A)`，不可把零H当BG。whole臂仅D内用W；同预算臂在两个固定eligible域按未裁whole ridge排序、row-major平票。未修改层、头、阈值或权重。

R0是绑定的实际direct CLI `old_arms.R.cli_key`；whole ridge/old calibration和全1300whole sentinel从signals已完成的冻结诊断复用。本次仅1295 valid R系数复原和11655真实窗应用，0 wholeQ raw重放。最终整张canonical用float32 bilinear align_corners=False、严格>.5导出原图，未拼接R0_original。

## 完整同批指标

|口径/帧|FoRIS|旧区域 M|未筛选|几何|R0 whole|native_or|whole|canonical 同预算|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|LVIS600 observed/原图|47.779371|52.005138|34.128915|49.744750|46.024083|49.748627|49.542621|49.680751|
|LVIS600 observed/canonical1024|47.583914|51.702409|33.977336|49.479666|45.925918|49.565256|49.375082|49.498583|
|LVIS600 fixed920/原图|31.160459|33.916394|22.257988|32.442228|30.015706|32.444757|32.310405|32.400490|
|LVIS600 fixed920/canonical1024|31.032987|33.718963|22.159132|32.269348|29.951686|32.325167|32.201141|32.281685|
|Deep fast100/原图|29.377542|37.344028|35.035001|36.537657|32.014976|36.426043|34.248939|35.761222|
|Deep fast100/canonical1024|29.377542|37.344028|35.035001|36.537657|32.014976|36.426043|34.248939|35.761222|
|PACO600 official303/原图|38.460868|34.971403|29.538809|—|40.774944|38.348501|38.202012|38.333703|
|PACO600 official303/canonical1024|38.438053|34.178532|29.398761|—|40.441563|37.679934|37.576170|37.668689|
|PACO pilot100 observed87/原图|45.563915|40.956998|33.938013|—|49.227401|45.177269|44.677610|45.145865|
|PACO pilot100 observed87/canonical1024|45.629834|40.914271|34.022009|—|49.296898|45.254348|44.755362|45.217920|

主口径为原图。LVIS observed600与fixed920分列；PACO600官方303槽位（264观察）；PACO pilot100仅87观察槽位；Deep为道路总I/U，不跨域平均mIoU。PACO的FoRIS canonical是真实CLI；LVIS/Deep FoRIS canonical是旧原图mask回渲guide，不冒充原始CLI。Deep query本身1024，canonical与原图终点在本组完全相同；其他组分帧报告。`archived_R0_comparison.json`复用主Sol已独立验收6380的1300 old.R整数，0 GT/mask/raw重读、0旧头/6380重跑；源FoRIS原图I/U与旧ledger全1300一致。

旧restore.accepted/raw_support/whole的完整I/U与指标仍在`report.json`和`archived_R0_comparison.json`，只复用旧已验收整数。

## 编辑、预算与区间

|组/新臂（原图相对M）|补真|补假|删真|删假|
|---|---:|---:|---:|---:|
|LVIS600 observed/native_or|292985|889038|144361|462523|
|LVIS600 observed/whole|283793|877835|163220|479319|
|LVIS600 observed/canonical_same_edit_budget|292574|889451|146049|461074|
|Deep fast100/native_or|87903|438467|225590|643295|
|Deep fast100/whole|79759|429219|411186|775969|
|Deep fast100/canonical_same_edit_budget|85459|440911|255791|613094|
|PACO600 official303/native_or|172700|196118|68282|300517|
|PACO600 official303/whole|155938|192332|74918|307430|
|PACO600 official303/canonical_same_edit_budget|172640|196300|68789|300044|

|组|D添加 K|D删除 K|共同B-only添加|invalid B-only添加|原图native/budget面积不同例数|
|---|---:|---:|---:|---:|---:|
|LVIS600 observed|1132991|2279839|3408507|106438|283|
|Deep fast100|139249|868885|387121|0|0|
|PACO600 official303|6278260|10083021|11963630|0|195|

`K_add_D=|F∩Eadd|`、`K_del_D=|Edel∩¬F|`，共同B-only独立记账。invalid5例D增删0，但其相对M的共同B-only添加可非零。全1300每例canonical native/budget添加删除数完全相等；**原图不声称同K**，完整实际面积在逐例production records及completion.json。TP/FP/FN、每帧I/U和四类编辑在`episodes.jsonl`，逐例closure已独立核验。

|native_or 原图相对|Δpp|query-photo 95%|connected R/Q 95%敏感度|
|---|---:|---|---|
|LVIS600 observed/region|-2.256511|[-2.256511, -2.256511]|[-1.912320, -0.918922]|
|LVIS600 observed/foris|1.969256|[1.969256, 1.969256]|[0.874295, 1.605012]|
|Deep fast100/region|-0.917984|[-1.940207, 0.242690]|[-1.955157, 0.196020]|
|Deep fast100/foris|7.048502|[5.453926, 8.751060]|[5.473757, 8.782659]|
|PACO600 official303/region|3.377098|[2.856305, 3.750290]|[1.855757, 3.186301]|
|PACO600 official303/foris|-0.112367|[-0.850442, 0.180279]|[-0.945256, 0.331375]|

query-photo5000及connected R/Q2000，seed0，仅新臂相关比较；`sensitivity_intervals.json`保存所有冻结控制，`archived_R0_comparison.json`另列新臂对R0。LVIS600每个观察class/fold只有1例，分层query-photo区间退化为点，不能当显著性。复用的connected实现让重采样未出现的声明槽位保持0，区间可能偏离原point；只解释为条件固定队列敏感度，不是独立新图验证。PACO native相对R0为−2.426443pp、photo95%[−2.865044,−1.988185]。

## 封存与独立验收

`sealed.json` SHA `bac5059e75d04949f1bd3469a9ae06faefa763ce2380fdff716741b6e4b1d36b`；`report.json` SHA `48197239445a0922622d98a0f04f4ed54b301e11521e3f77ab3f3e48d3628291`；`episodes.jsonl` SHA `34a8d535ff7a6db6bbb8bebd16c5d5d4754ef7e83cc156fb51030ae3673deb89`。query mask guard贯穿完整7800输出封存，访问尝试0，评分先全文件gate再第一次GT。

生产时3900独立Boolean/稳定排序公式重建通过；随后独立direct AND/OR：7800 I/U、7800四编辑、3900canonical→original、7800padding、17000旧整数身份和96聚合值均通过。`verification.json`固定相应SHA，`completion.json`封闭引用所有必要报告和每组预算。

## 实际成本

本候选单worker、4Torch线程，wall 1647.769547s、CPU 4066.349044s、peakRSS 1600733184 bytes。reference fit 99.064287s；真实窗读出/支持 1503.564538s；组成与独立公式核验 13.282790s；导出/记录写入 11.315994s。12950 raw请求/12678不同输入，每次payload+tensor SHA核验；请求payload容器体积 209356348932 bytes，IO wall 588.241296s、IO CPU 568.049418s，IO已包含于总wall且与阶段分段重叠，不能再次相加。该字节数是逻辑请求容器体积，不是OS实测SSD物理字节。

评分wall 14.102937s/CPU 13.498613s；独立验收wall 22.308513s/CPU 22.303180s；新控制区间wall 1.524422s/CPU 1.433323s；匹配R0整数控制与新对比区间wall 1.620785s/CPU 1.517409s。输入绑定另记wall40.852175s/CPU38.423073s。0新encoder、0CRF、0raw写入、0新dense归档。

历史真实native编码与保存流程记录绑定于`historical_cost_receipt.json`，不重复编码。LVIS最后200的seal wall是收尾进程，不能当完整批成本；Deep是已有raw的重读出；历史cache-aware时间分段不构成匹配冷启动延迟。signals先行1300诊断和其失败attempt成本由其native_fixed_head报告单列，不混入本候选新增成本。缓存CPU时间不称冷端到端速度。

## 顺序Pro200

原Pro包仅去除`payload['field/'+name]=field`归档行，核心源码与metrics字节不变，收据`pro_adapter_receipt.json`。固定Deep100+PACO100、whole+真实四角O24、六臂/rank64/T=.07/role128、signed128→原图FP64 bilinear严格>0；保存mask已是实际PACO crop，不能重复裁剪。输出`../cv_data/a/scene_reconstruction_20261011`，全六臂封存后才GT，缺cache报错，不触发编码。属于已授权的独立机制验证，结果另归档，不扩展。

Pro200已完成：六臂1200原图mask封存、同批完整评分/独立I/U/四编辑/照片区间通过。scene_reconstructed Deep11.499408/PACO observed87 29.657600，低于同批FoRIS29.377542/45.563915、MEAN26.465369/46.687946和快九窗37.344028/40.956998；固定机制实验关闭不扩展。完整原图结果、源码/I/O适配、两次清单适配失败成本与真实历史编码边界详见 [`Pro200 REPORT.md`](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/scene_reconstruction_20261011/REPORT.md)。
