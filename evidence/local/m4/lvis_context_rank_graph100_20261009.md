# LVIS 同100条件排序图传播

固定100例已封存、评分并通过独立审查。原参考guide、unary、置信权重、CG和输出流程均保持；只在至少一端位于原候选集合 `G=s>.5` 的图边上，按参考guide的条件CDF距离衰减。对照使用相同边集合，统一衰减并匹配总边权。全部预测完成后才读取query GT评分。

| 方法 | CLI mIoU | 原图 mIoU | CLI零交集 |
|---|---:|---:|---:|
| 完整FoRIS | 47.704029 | 47.873899 | 1 |
| 原MEAN | 47.109370 | 47.182508 | 7 |
| 同E等边权对照 | 47.472579 | 47.573653 | 6 |
| 条件排序图 | 47.701403 | 47.789883 | 6 |

CLI候选较原MEAN+.592033（7折涨、3折跌），较同量对照+.228825（6折涨、4折跌）；较完整FoRIS−.002626。原图仍比FoRIS低.084016。候选有匹配控制的同批正差，但未超过完整FoRIS。旧FG-anchor仍为50.047216，零交集9例；本轮没有与anchor叠加。

候选相对同E等边权对照的精确CLI贡献为：补真+.432754、补假−.207758、删真−.120871、删假+.124700，合计+.228825。相对MEAN为补真+.911474、补假−.358272、删真−.148723、删假+.187555。新增真前景提供主要正贡献，但仍伴随新增假前景及误删。

封存后40个query面积<1%的案例，沿用fold/class聚合：候选21.439252、原MEAN20.424772、对照21.055882、FoRIS22.139022；较MEAN+1.014480、较对照+.383370，仍低FoRIS.699770。其余60例候选较MEAN+.135958、较对照+.063607。子组fold/class权重不同，不可用例数加权平均还原总体分数。

实际全图边权保留中位.995259，受改动E占全图质量中位.041801，E内统一对照η中位.871383。E外边及unary保持不保证G外最终图解保持，保存场证实解向外传播。

独立审查覆盖100例CDF半ties及直接计数、原s/guide/unary/A、同E权重与总质量、200个实际H差分与方程残差、全部最终mask/IU、基线和编辑归因。质量匹配最大误差4.55e−13，原对角舍入残差保持误差1.42e−14。原H实际FP64，未强制降精度或重建父图。

aidemo、codex_m4；session70181 exit0，进程结束快于PID快照，未编造PID。编码0、推理图像解码0、新query照片0，评分阶段读取100个query mask。8worker逐例时间累计7.753秒，日志创建到完整report5.369秒（含启动/推理/评分）。无后台推理或自动扩量。

这100例标签此前已暴露，是开发探索；经典bilateral rank gating不构成新参考信息或原创性证明。主线继续检查小目标漏检和参考粒度的迁移，不据此恢复旧全量或加入默认入口。

产物：[report.json](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_context_rank_graph100_20261009/report.json)、[封存后诊断](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_context_rank_graph100_20261009/postseal_diagnostics.json)、[独立审查](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_context_rank_graph100_20261009/reviews/independent_audit.md)、[执行收据](/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/lvis_context_rank_graph100_20261009/execution_receipt.json)。
