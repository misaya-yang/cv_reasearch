# 原35个连续公式：两批曝光开发的完整并表

**1360和500均是曝光开发队列。本次完整评价相同35个预封存公式；不新增网格、不按域选权重、不改来源，不是再次独立验证。先封存全部17,500自然mask，再统一评分；新confirm1000未访问。**

此前只复现第一批赢家b0m1u2t1的结果按原字节保留；本次所有四个纯源及该先前公式的自然mask和评分整数全部复现。

|公式|B/M/U/T|直接来源|总组件依赖|1360十域均值|500十域均值|两批等权均值|500−network4|
|---|---|---|---|---:|---:|---:|---:|
|b0m2u2t0|0/0.5/0.5/0|M,U|FoRIS_score64,MEAN_rank_and_query_graph,UN_reliability_graph,VP_component_signals,independent_c32|54.959639|55.304008|55.131823|+0.454710|
|b0m2u1t1|0/0.5/0.25/0.25|M,U,T|FoRIS_score64,MEAN_rank_and_query_graph,UN_reliability_graph,VP_component_signals,independent_c32|55.181100|54.980630|55.080865|+0.131332|
|b0m1u3t0|0/0.25/0.75/0|M,U|FoRIS_score64,MEAN_rank_and_query_graph,UN_reliability_graph,VP_component_signals,independent_c32|55.024588|54.831256|54.927922|-0.018042|
|b0m1u2t1|0/0.25/0.5/0.25|M,U,T|FoRIS_score64,MEAN_rank_and_query_graph,UN_reliability_graph,VP_component_signals,independent_c32|55.188602|54.411900|54.800251|-0.437397|
|b1m0u2t1|0.25/0/0.5/0.25|B,U,T|FoRIS_final_CRF,FoRIS_score64,UN_reliability_graph,VP_component_signals,independent_c32|54.836433|54.759866|54.798149|-0.089432|
|b1m0u3t0|0.25/0/0.75/0|B,U|FoRIS_final_CRF,FoRIS_score64,UN_reliability_graph,VP_component_signals,independent_c32|54.465612|54.812492|54.639052|-0.036805|
|b0m0u4t0|0/0/1/0|U|UN_reliability_graph,VP_component_signals,independent_c32|54.726007|54.230904|54.478456|-0.618394|
|b0m3u0t1|0/0.75/0/0.25|M,T|FoRIS_score64,MEAN_rank_and_query_graph,independent_c32|54.089427|54.558848|54.324137|-0.290450|
|b0m3u1t0|0/0.75/0.25/0|M,U|FoRIS_score64,MEAN_rank_and_query_graph,UN_reliability_graph,VP_component_signals,independent_c32|53.790726|54.797904|54.294315|-0.051393|
|b0m0u3t1|0/0/0.75/0.25|U,T|UN_reliability_graph,VP_component_signals,independent_c32|54.712728|53.795228|54.253978|-1.054069|
|b1m1u1t1|0.25/0.25/0.25/0.25|B,M,U,T|FoRIS_final_CRF,FoRIS_score64,MEAN_rank_and_query_graph,UN_reliability_graph,VP_component_signals,independent_c32|53.910108|54.301877|54.105993|-0.547421|
|b1m0u1t2|0.25/0/0.25/0.5|B,U,T|FoRIS_final_CRF,FoRIS_score64,UN_reliability_graph,VP_component_signals,independent_c32|54.535163|53.622419|54.078791|-1.226878|
|b1m1u2t0|0.25/0.25/0.5/0|B,M,U|FoRIS_final_CRF,FoRIS_score64,MEAN_rank_and_query_graph,UN_reliability_graph,VP_component_signals,independent_c32|53.519705|54.087943|53.803824|-0.761355|
|b0m2u0t2|0/0.5/0/0.5|M,T|FoRIS_score64,MEAN_rank_and_query_graph,independent_c32|54.024356|53.364736|53.694546|-1.484562|
|b1m1u0t2|0.25/0.25/0/0.5|B,M,T|FoRIS_final_CRF,FoRIS_score64,MEAN_rank_and_query_graph,independent_c32|53.868544|53.475681|53.672113|-1.373617|
|b0m1u1t2|0/0.25/0.25/0.5|M,U,T|FoRIS_score64,MEAN_rank_and_query_graph,UN_reliability_graph,VP_component_signals,independent_c32|54.087416|53.203925|53.645671|-1.645373|
|b0m0u2t2|0/0/0.5/0.5|U,T|UN_reliability_graph,VP_component_signals,independent_c32|53.881029|52.715529|53.298279|-2.133769|
|b1m0u0t3|0.25/0/0/0.75|B,T|FoRIS_final_CRF,FoRIS_score64,independent_c32|53.282763|52.339401|52.811082|-2.509897|
|b1m2u0t1|0.25/0.5/0/0.25|B,M,T|FoRIS_final_CRF,FoRIS_score64,MEAN_rank_and_query_graph,independent_c32|52.305633|53.001591|52.653612|-1.847707|
|b1m2u1t0|0.25/0.5/0.25/0|B,M,U|FoRIS_final_CRF,FoRIS_score64,MEAN_rank_and_query_graph,UN_reliability_graph,VP_component_signals,independent_c32|52.155106|52.941860|52.548483|-1.907438|
|b0m4u0t0|0/1/0/0|M|FoRIS_score64,MEAN_rank_and_query_graph|51.586447|53.207143|52.396795|-1.642154|
|b1m3u0t0|0.25/0.75/0/0|B,M|FoRIS_final_CRF,FoRIS_score64,MEAN_rank_and_query_graph|51.824664|52.616366|52.220515|-2.232932|
|b0m1u0t3|0/0.25/0/0.75|M,T|FoRIS_score64,MEAN_rank_and_query_graph,independent_c32|52.573148|51.812266|52.192707|-3.037032|
|b2m0u1t1|0.5/0/0.25/0.25|B,U,T|FoRIS_final_CRF,FoRIS_score64,UN_reliability_graph,VP_component_signals,independent_c32|51.725289|52.464008|52.094649|-2.385289|
|b2m0u2t0|0.5/0/0.5/0|B,U|FoRIS_final_CRF,FoRIS_score64,UN_reliability_graph,VP_component_signals,independent_c32|51.725289|52.464008|52.094649|-2.385289|
|b2m1u0t1|0.5/0.25/0/0.25|B,M,T|FoRIS_final_CRF,FoRIS_score64,MEAN_rank_and_query_graph,independent_c32|51.725289|52.464008|52.094648|-2.385290|
|b2m1u1t0|0.5/0.25/0.25/0|B,M,U|FoRIS_final_CRF,FoRIS_score64,MEAN_rank_and_query_graph,UN_reliability_graph,VP_component_signals,independent_c32|51.725289|52.464008|52.094648|-2.385290|
|b2m2u0t0|0.5/0.5/0/0|B,M|FoRIS_final_CRF,FoRIS_score64,MEAN_rank_and_query_graph|51.725289|52.464008|52.094648|-2.385290|
|b3m0u0t1|0.75/0/0/0.25|B,T|FoRIS_final_CRF,FoRIS_score64,independent_c32|51.725289|52.464008|52.094648|-2.385290|
|b3m0u1t0|0.75/0/0.25/0|B,U|FoRIS_final_CRF,FoRIS_score64,UN_reliability_graph,VP_component_signals,independent_c32|51.725289|52.464008|52.094648|-2.385290|
|b3m1u0t0|0.75/0.25/0/0|B,M|FoRIS_final_CRF,FoRIS_score64,MEAN_rank_and_query_graph|51.725289|52.464008|52.094648|-2.385290|
|b4m0u0t0|1/0/0/0|B|FoRIS_final_CRF,FoRIS_score64|51.725289|52.464008|52.094648|-2.385290|
|b2m0u0t2|0.5/0/0/0.5|B,T|FoRIS_final_CRF,FoRIS_score64,independent_c32|51.725288|52.464008|52.094648|-2.385289|
|b0m0u1t3|0/0/0.25/0.75|U,T|UN_reliability_graph,VP_component_signals,independent_c32|52.405237|51.629360|52.017299|-3.219938|
|b0m0u0t4|0/0/0/1|T|independent_c32|51.083567|50.279233|50.681400|-4.570064|

**依赖不等于混合权重：M即使在B权重0时仍依赖FoRIS score64。各来源共享DINO与部分计算，不能把分支旧耗时直接相加；最终选择还需实测完整管线延迟。**

两批等权均值仅将两个开发macro平均，不是把1860例池化，也不是新确认统计。完整每公式每域两批分数、对FoRIS/U/network4差值和500逐例增删整数均在JSON。

## 三种开发排序首位的逐域取舍

|公式|域|1360分数|1360−B|500分数|500−B|500−network4|
|---|---|---:|---:|---:|---:|---:|
|b0m1u2t1|coco|64.1256|-0.6603|61.7459|-0.6574|-0.3847|
|b0m1u2t1|lvis|47.7374|+1.2078|50.3047|-2.6404|-0.1797|
|b0m1u2t1|pascal_part|59.3503|+3.7602|55.0758|+3.4497|-1.1317|
|b0m1u2t1|paco_part|49.1970|+6.7911|54.5658|+2.9133|+0.0663|
|b0m1u2t1|suim|58.6045|-0.2440|52.2415|-3.9486|-0.7166|
|b0m1u2t1|isaid|53.7755|+3.3499|56.7325|+0.8399|-0.4505|
|b0m1u2t1|isic|62.7979|-0.3240|54.2317|-2.7441|-2.9018|
|b0m1u2t1|lung|89.9531|+2.7969|89.0070|+2.8484|+0.2421|
|b0m1u2t1|fundus|34.4982|+15.4867|34.9821|+15.9300|+0.3016|
|b0m1u2t1|deepglobe_road|31.8464|+2.4689|35.2319|+3.4882|+0.7810|
|b0m2u2t0|coco|66.0402|+1.2543|65.7414|+3.3380|+3.6107|
|b0m2u2t0|lvis|48.2525|+1.7230|52.1325|-0.8126|+1.6481|
|b0m2u2t0|pascal_part|58.3478|+2.7577|55.9626|+4.3365|-0.2449|
|b0m2u2t0|paco_part|48.3134|+5.9074|54.5982|+2.9456|+0.0987|
|b0m2u2t0|suim|59.8438|+0.9953|54.2218|-1.9683|+1.2637|
|b0m2u2t0|isaid|52.2209|+1.7953|56.6826|+0.7899|-0.5004|
|b0m2u2t0|isic|63.8207|+0.6987|57.5758|+0.6000|+0.4423|
|b0m2u2t0|lung|89.4423|+2.2860|88.3196|+2.1610|-0.4453|
|b0m2u2t0|fundus|33.3289|+14.3173|33.5425|+14.4904|-1.1380|
|b0m2u2t0|deepglobe_road|29.9859|+0.6084|34.2631|+2.5194|-0.1878|

[全部35公式完整结果](../../evidence/research/x1012s2_soft500_all35.json)。
封存SHA `e2b342c6f44996797308636f1489adf3bdf1fe78a68718f7dd1cdf0cb289f0a6`；配置SHA `35a7a8bf999c67bd78ee985fca32335d0eec4eb9a0bf8352d084375c7eccce63`。
本任务不自行冻结最终部署方法；两批开发结果与完整延迟应共同用于之后的选择。

## 完整开发比较的直接结果

1360首位仍是b0m1u2t1（.25M+.5U+.25T）：55.188602；其500均值54.411900。500首位及两批等权首位均为b0m2u2t0（.5M+.5U）：1360为54.959639、500为55.304008，两批等权55.131823。第二名b0m2u1t1（.5M+.25U+.25T）分别55.181100/54.980630，两批等权55.080865。

b0m2u2t0在1360十域对FoRIS点估计均为正；500仅LVIS−0.8126、SUIM−1.9683pp为负。对500 network4，它在COCO+3.6107、LVIS+1.6481、SUIM+1.2637pp，但Fundus−1.1380、iSAID−0.5004、Lung−0.4453pp，需作为完整取舍保留。相对network4四项合计为补回TP1,840,747、新增FP4,581,028、删除TP1,211,252、移除FP2,572,716；不能用单项增删替代完整class IoU排序。

本批是对已有35公式的完整曝光开发搜索；所有公式和两批结果一起用于后续方法选择。此前单公式复现的原结果没有被覆盖，新confirm1000尚未读取。
