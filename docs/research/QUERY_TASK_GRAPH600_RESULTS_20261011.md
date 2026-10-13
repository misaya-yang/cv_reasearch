# 查询双角色任务图：600完整实验已淘汰

原图完整方法49.876534，比FoRIS47.779371高2.097163pp，但低于已有最强S52.005138，也低于去RGB50.257765与uniformgeometry51.218818。预先S+1pp/CLI/完整对照门槛未通过，配方淘汰，不扩6200、不扫参数、不追加本配方变体；总研究goal active且未达成。

## 完整方法与新证据

原历史LVIS600身份/类/尺寸/crop已逐个exactmatch，不取1400前600。旧S全处理直接保留，不重跑窗口/编码；从S/B/G/R0四方共同FG与共同BG形成query64格软面积种子，在非一致域用DINO O24与实际RGB构8邻接图，CPU64解(L+diag(f+b))u=f。原图和CLI各自FP64直接渲染，严格.5、tieS；1111/0000保护，无旧conf/alpha移植。完整公式见[冻结协议](QUERY_TASK_GRAPH600_PROTOCOL_20261011.md)。

设计依据为真实损失结构，而非全图AUC：600空间动作中S误删B真675694px，641500仍邻接真S；失败修复attachedTP233138但FP731995。附件关系不是语义证书。固定3案例/对应又反驳更纯reference或更完整BG的简单补法：两个坏例约98%/94%误恢复的原近邻来源是纯RFG；fullBG只翻17%/20%错却翻最佳例100%正确新增。它们是像素池/选定案例证据，不是各class贡献、五域重复或理论原创。

## 同批完整终点

| 输出 | CLI观察600 | 原图观察600 | 原图官方920 | 原图对S pp |
|---|---:|---:|---:|---:|
| FoRIS完整 | 47.619569 | 47.779371 | 31.160459 | -4.225767 |
| MEAN完整 | 47.580880 | 47.684953 | 31.098882 | -4.320185 |
| 参考R0完整 | 45.925918 | 46.024083 | 30.015706 | -5.981055 |
| 多数C | 48.445810 | 48.562655 | 31.671297 | -3.442482 |
| 支持前景N | 48.255239 | 48.381993 | 31.553474 | -3.623144 |
| 原最强完整S | 51.702409 | 52.005138 | 33.916394 | +0.000000 |
| DINO+RGB任务图完整 | 49.754918 | 49.876534 | 32.528175 | -2.128603 |
| 去RGB完整消融 | 50.129924 | 50.257765 | 32.776804 | -1.747372 |
| uniformgeometry完整消融 | 51.046163 | 51.218818 | 33.403577 | -0.786320 |
| 同S实际编辑预算完整控制 | 49.702102 | 49.832123 | 32.499211 | -2.173015 |

600观察槽位和920固定槽位是同mask的两种声明统计，不混比；每fold60各不同类，920固定缺项0，pool I/U后class/fold均值。此为曝光开发池，不能称未见验证。失败点值已足以淘汰，不重复退化的class/photo bootstrap。

| 原图相对S四编辑 | 补真TP | 补假FP | 删真TP | 删假FP |
|---|---:|---:|---:|---:|
| DINO+RGB任务图完整 | 517324 | 1168571 | 288337 | 829601 |
| 去RGB完整消融 | 523552 | 1139554 | 293652 | 836990 |
| uniformgeometry完整消融 | 387311 | 744848 | 401469 | 976252 |
| 同S实际编辑预算完整控制 | 508097 | 1177798 | 296661 | 821277 |

主臂恢复517324TP却新增1168571FP，又删除288337TP/829601FP，净TP+228987、净FP+338970，class mIoU仍退步。RGB移除+ .381231pp；uniformgeometry比full高1.342283pp；full对同预算只+.044412pp。主臂对B/N/去RGB/几何/预算逐例有限四编辑完整保留，不能拿净像素或对FoRIS正差替代最强同批比较。

消融验证的是这些完整算子的后果：RGB因子和DINO亲和同时改变边连接强度/分布以及相对seed unary的平滑强度。不能据此宣称RGB或DINO表征本身没有任务信息，也不能把geometry胜出当可接纳方法。候选所有图都低S。

## 一个可定位的条件恢复上界

若图边非负，某锚定连通组件的精确解满足：min_i f_i/(f_i+b_i) ≤ u ≤ max_i f_i/(f_i+b_i)，仅在f+b>0节点取比率。最大值处Lu≥0，结合(L+diag(f+b))u=f；无锚节点沿连通边继承该界。convex bilinear继续保持上界，真正无锚组件由support禁编辑。

所以当seed ratio最大值≤.5，任意该拓扑上的非负亲和都不能产生严格>.5恢复，不是source目标不存在。这不是全模型不可识别定理；也不是source/crossdataset新理论。4个实际full低field例仅2个满足这个图无关证书：3/42有6个共同FG像素，ratio上界.029412；7/29有15个，ratio上界.217391。5/47和5/52虽field低.5但ratio上界1，不能称结构性不可能。完整核对见[seed ceiling](../../evidence/local/query_task_graph600_20261011/seed_ceiling_case4.json)。

该证据把下一问题定位到任务role面积压缩/label confidence和读出，不能用本轮强行推广调阈值、class平衡或细网格版本。先分开角色约束、同一cell内FG/BG混合及已有fine mask的位置证据，理论要直接约束下一个完整设计。

## 执行、失败与成本

全4800mask先seal，全SHA/schema/padding gate才600GT。四臂直接I/U、TP/FP/FN、S-relativegross、四方保护、actualframe增删预算和row-major位置全部通过；root另70887项整数/官方class/solver residual核验，新增mask/GT/raw读取0，0.015040s。

生产90.380443s/CPU84.585991s/RSS1.112GiB，600均MPS32邻距离+CPU64CG，无backendfallback。600次旧Q raw加载/600unique，容器请求9,335,035,979 bytes（不是新磁盘占用），R raw读取0，Q RGB600。IO40.0558s，edge+三CG18.6201s，frame/budget15.5924s。评分22.138788s/CPU23.374703s/RSS0.833GiB。新encoder/CRF/照片/views/densearchive0，逐例场丢弃。

每图593成功solve/7缺共同query role回S，共1779次实际线性推理；full/noRGB/geometry总iterations24501/17870/20711，最大1269/175/107。full8个无锚节点禁止编辑；其余两图0。CG实际residual满足rtol1e-7，不能把每例线性推理说成0fitting或拿最大原则作语义正确保证。

一次首启动在0mask/0raw阶段ics模块src路径缺失退出：完整源码、旧协议、config/log保留failed_start_module_path/；toolwall.807280s，CPU/RSS未测明确unknown，读取Q RGB1。仅修import路径，方法、solver、600清单均未改，再冻结实际协议37992a5e。不是观察超时重启。

- [当前协议与17源SHA](../../evidence/local/query_task_graph600_20261011/protocol.json)，SHA256 `37992a5e51b79b8fd90d25ff34290ee25b8403cf69d50e7cc5b35957ad1d3d9c`。
- [最终seal](../../evidence/local/query_task_graph600_20261011/sealed.json)，SHA256 `affda3be415549cc6a3a59449722269ba2cf6b7da7a89af453ba6295d49c51ed`。
- [完整报告JSON](../../evidence/local/query_task_graph600_20261011/report.json)、[逐例I/U/四编辑](../../evidence/local/query_task_graph600_20261011/episodes.jsonl)、[root核验/solver账本](../../evidence/local/query_task_graph600_20261011/root_scalar_verification.json)。
- [真实case3图/annotation](../../evidence/local/strong600_case_conflicts_20261011/report.json)、[source最近对应3](../../evidence/local/strong600_witness_provenance3_20261011/report.json)、[空间600](../../evidence/local/strong600_spatial_conflicts_20261011/report.json)。
- [决定](../../evidence/local/query_task_graph600_20261011/screening_decision.json)、[成本](../../evidence/local/query_task_graph600_20261011/production_receipt.json)、[完成与资产SHA/bytes](../../evidence/local/query_task_graph600_20261011/completion.json)。

旧强S及数据/权重/raw/源码/清单/必要最终mask和逐例整数全部保留。当前没有producer/scorer/GPU任务，下一候选未冻结；主goal仍active，按600明确淘汰规则继续。
