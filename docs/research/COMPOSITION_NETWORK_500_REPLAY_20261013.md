# 分支AND/OR网络在已曝光500配对上的固定复现

**同一已曝光R5开发复现，不是新盲测；两条网络先冻结并封存全部自然输出，再读GT评分。不修改原r00097，不访问新confirm1000。**

`network4=fce0fce0fce0fce0`：`(B&U)|(M&U)|(M&T)|(U&T)`，相对原r00097只新增`M&T&~U`。
`network6=fee0fee0fee0fcc0`：`(M&U)|(M&T)|(U&T)|(B&U&I)|(B&T&I)|(B&U&J)|(B&T&J)`。

|域|FoRIS|MEAN|UN|原r00097|network4|network6|network4−原|network6−原|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|coco|62.4033|64.3254|62.2353|62.2899|62.1307|62.1461|-0.1592|-0.1437|
|lvis|52.9451|52.6973|49.6511|49.9599|50.4845|51.1077|0.5246|1.1478|
|pascal_part|51.6261|53.8425|55.3130|55.4742|56.2075|55.8850|0.7333|0.4108|
|paco_part|51.6526|52.8329|53.2989|54.4112|54.4995|54.6967|0.0883|0.2855|
|suim|56.1901|55.7495|52.2067|52.3477|52.9581|53.2141|0.6103|0.8664|
|isaid|55.8927|56.7934|56.9621|57.0543|57.1830|57.0689|0.1287|0.0147|
|isic|56.9758|59.7366|55.2798|56.7225|57.1335|56.5237|0.4110|-0.1988|
|lung|86.1587|87.2603|88.4980|88.7826|88.7650|88.5991|-0.0176|-0.1835|
|fundus|19.0521|17.5938|34.6805|34.6805|34.6805|34.6805|0.0000|0.0000|
|deepglobe_road|31.7436|31.2397|34.1837|34.3539|34.4509|34.6576|0.0969|0.3037|
|十域均值|52.4640|53.2071|54.2309|54.6077|54.8493|54.8580|0.2416|0.2503|

## 对原r00097的真实增删代价

|域|网络|补回TP|新增FP|删除TP|移除FP|对原分数pp|
|---|---|---:|---:|---:|---:|---:|
|coco|network4|30770|46540|0|0|-0.1592|
|coco|network6|43745|62505|135|2446|-0.1437|
|lvis|network4|94093|150657|0|0|+0.5246|
|lvis|network6|128313|174167|2002|3664|+1.1478|
|pascal_part|network4|409862|309611|0|0|+0.7333|
|pascal_part|network6|445367|367999|5658|67116|+0.4108|
|paco_part|network4|88915|146534|0|0|+0.0883|
|paco_part|network6|100594|170416|5807|12553|+0.2855|
|suim|network4|212451|87780|0|0|+0.6103|
|suim|network6|270889|113704|5263|61003|+0.8664|
|isaid|network4|83023|168205|0|0|+0.1287|
|isaid|network6|109750|198655|34096|15264|+0.0147|
|isic|network4|156550|171552|0|0|+0.4110|
|isic|network6|207102|429681|85|4752|-0.1988|
|lung|network4|15844|20691|0|0|-0.0176|
|lung|network6|27760|60847|3259|3553|-0.1835|
|fundus|network4|0|0|0|0|+0.0000|
|fundus|network6|0|0|0|0|+0.0000|
|deepglobe_road|network4|6704|7794|0|0|+0.0969|
|deepglobe_road|network6|8775|14821|2922|34273|+0.3037|

## 固定规则照片配对区间

|范围|网络|对照|差值pp|95%区间|
|---|---|---|---:|---|
|coco|network4|FoRIS|-0.2727|[-3.4055,+2.1254]|
|coco|network4|MEAN|-2.1948|[-4.3717,+0.1916]|
|coco|network4|UN_midrange|-0.1046|[-0.4844,+0.2296]|
|coco|network4|r00097|-0.1592|[-0.5299,+0.1903]|
|coco|network6|FoRIS|-0.2572|[-3.3695,+2.1749]|
|coco|network6|MEAN|-2.1793|[-4.3497,+0.1868]|
|coco|network6|UN_midrange|-0.0892|[-0.5349,+0.3349]|
|coco|network6|r00097|-0.1437|[-0.5924,+0.2732]|
|coco|network6|network4|+0.0155|[-0.0899,+0.1566]|
|lvis|network4|FoRIS|-2.4606|[-4.7757,-0.0936]|
|lvis|network4|MEAN|-2.2128|[-4.6173,-0.0696]|
|lvis|network4|UN_midrange|+0.8334|[+0.0273,+1.5567]|
|lvis|network4|r00097|+0.5246|[-0.1257,+1.0151]|
|lvis|network6|FoRIS|-1.8374|[-4.0220,+0.4336]|
|lvis|network6|MEAN|-1.5896|[-4.1380,+0.8714]|
|lvis|network6|UN_midrange|+1.4567|[-0.0461,+2.6823]|
|lvis|network6|r00097|+1.1478|[-0.3245,+2.3105]|
|lvis|network6|network4|+0.6232|[-0.3305,+1.4087]|
|pascal_part|network4|FoRIS|+4.5814|[+1.9237,+6.8727]|
|pascal_part|network4|MEAN|+2.3650|[+0.0080,+4.2865]|
|pascal_part|network4|UN_midrange|+0.8945|[-0.4253,+1.9535]|
|pascal_part|network4|r00097|+0.7333|[-0.4810,+1.7107]|
|pascal_part|network6|FoRIS|+4.2589|[+1.7786,+6.3967]|
|pascal_part|network6|MEAN|+2.0425|[-0.5596,+4.5283]|
|pascal_part|network6|UN_midrange|+0.5719|[-1.1829,+2.2654]|
|pascal_part|network6|r00097|+0.4108|[-1.2313,+2.0631]|
|pascal_part|network6|network4|-0.3225|[-1.0263,+0.6098]|
|paco_part|network4|FoRIS|+2.8469|[+0.0159,+5.4771]|
|paco_part|network4|MEAN|+1.6666|[-1.2076,+4.3678]|
|paco_part|network4|UN_midrange|+1.2006|[-0.0366,+2.1420]|
|paco_part|network4|r00097|+0.0883|[-0.2172,+0.3808]|
|paco_part|network6|FoRIS|+3.0442|[+0.1936,+5.7076]|
|paco_part|network6|MEAN|+1.8638|[-0.9948,+4.3875]|
|paco_part|network6|UN_midrange|+1.3978|[-0.1013,+2.5392]|
|paco_part|network6|r00097|+0.2855|[-0.2472,+0.7568]|
|paco_part|network6|network4|+0.1972|[-0.1301,+0.4541]|
|suim|network4|FoRIS|-3.2320|[-8.5925,+1.0509]|
|suim|network4|MEAN|-2.7914|[-8.2072,+0.9619]|
|suim|network4|UN_midrange|+0.7514|[+0.1669,+1.4553]|
|suim|network4|r00097|+0.6103|[+0.0846,+1.2694]|
|suim|network6|FoRIS|-2.9760|[-8.2959,+1.3897]|
|suim|network6|MEAN|-2.5353|[-7.9563,+1.3600]|
|suim|network6|UN_midrange|+1.0074|[+0.3809,+1.7045]|
|suim|network6|r00097|+0.8664|[+0.2179,+1.5651]|
|suim|network6|network4|+0.2561|[-0.0505,+0.5387]|
|isaid|network4|FoRIS|+1.2903|[-0.8147,+4.0653]|
|isaid|network4|MEAN|+0.3896|[-3.2467,+3.7958]|
|isaid|network4|UN_midrange|+0.2209|[-0.7419,+1.3354]|
|isaid|network4|r00097|+0.1287|[-0.7208,+1.2848]|
|isaid|network6|FoRIS|+1.1763|[-0.8699,+3.9907]|
|isaid|network6|MEAN|+0.2755|[-3.3834,+3.7493]|
|isaid|network6|UN_midrange|+0.1069|[-0.8767,+1.3912]|
|isaid|network6|r00097|+0.0147|[-0.9034,+1.3347]|
|isaid|network6|network4|-0.1141|[-0.3674,+0.2850]|
|isic|network4|FoRIS|+0.1577|[-1.8674,+2.1484]|
|isic|network4|MEAN|-2.6032|[-6.3967,+0.4496]|
|isic|network4|UN_midrange|+1.8537|[-0.3442,+5.5421]|
|isic|network4|r00097|+0.4110|[-0.5840,+1.3485]|
|isic|network6|FoRIS|-0.4520|[-2.1389,+1.2229]|
|isic|network6|MEAN|-3.2129|[-7.5948,+0.4922]|
|isic|network6|UN_midrange|+1.2440|[-1.9706,+5.2669]|
|isic|network6|r00097|-0.1988|[-2.4862,+1.6214]|
|isic|network6|network4|-0.6097|[-2.0747,+0.5464]|
|lung|network4|FoRIS|+2.6063|[+1.3705,+4.1387]|
|lung|network4|MEAN|+1.5047|[+0.4749,+2.8782]|
|lung|network4|UN_midrange|+0.2669|[-0.2035,+0.7509]|
|lung|network4|r00097|-0.0176|[-0.1677,+0.1126]|
|lung|network6|FoRIS|+2.4404|[+1.1993,+3.9182]|
|lung|network6|MEAN|+1.3388|[+0.2569,+2.7484]|
|lung|network6|UN_midrange|+0.1011|[-0.3686,+0.5934]|
|lung|network6|r00097|-0.1835|[-0.4315,+0.0590]|
|lung|network6|network4|-0.1659|[-0.3765,+0.0070]|
|fundus|network4|FoRIS|+15.6284|[+14.0715,+17.2089]|
|fundus|network4|MEAN|+17.0867|[+15.6511,+18.6002]|
|fundus|network4|UN_midrange|+0.0000|[+0.0000,+0.0000]|
|fundus|network4|r00097|+0.0000|[+0.0000,+0.0000]|
|fundus|network6|FoRIS|+15.6284|[+14.0715,+17.2089]|
|fundus|network6|MEAN|+17.0867|[+15.6511,+18.6002]|
|fundus|network6|UN_midrange|+0.0000|[+0.0000,+0.0000]|
|fundus|network6|r00097|+0.0000|[+0.0000,+0.0000]|
|fundus|network6|network4|+0.0000|[+0.0000,+0.0000]|
|deepglobe_road|network4|FoRIS|+2.7072|[+0.1505,+4.9141]|
|deepglobe_road|network4|MEAN|+3.2111|[+0.2145,+5.8355]|
|deepglobe_road|network4|UN_midrange|+0.2672|[+0.0354,+0.5976]|
|deepglobe_road|network4|r00097|+0.0969|[-0.0494,+0.3494]|
|deepglobe_road|network6|FoRIS|+2.9140|[+0.2563,+5.3849]|
|deepglobe_road|network6|MEAN|+3.4179|[+0.4797,+6.1350]|
|deepglobe_road|network6|UN_midrange|+0.4740|[+0.0725,+1.2077]|
|deepglobe_road|network6|r00097|+0.3037|[-0.0757,+0.9703]|
|deepglobe_road|network6|network4|+0.2068|[-0.0742,+0.7965]|
|macro10|network4|FoRIS|+2.3853|[+1.4434,+3.1581]|
|macro10|network4|MEAN|+1.6422|[+0.6844,+2.5594]|
|macro10|network4|UN_midrange|+0.6184|[+0.2279,+1.0519]|
|macro10|network4|r00097|+0.2416|[+0.0183,+0.4541]|
|macro10|network6|FoRIS|+2.3940|[+1.5278,+3.1825]|
|macro10|network6|MEAN|+1.6508|[+0.6862,+2.6256]|
|macro10|network6|UN_midrange|+0.6271|[+0.1660,+1.1380]|
|macro10|network6|r00097|+0.2503|[-0.0956,+0.5911]|
|macro10|network6|network4|+0.0087|[-0.1985,+0.2456]|

全部对FoRIS/MEAN/UN/原规则的四项整数与逐例评分保留在JSON；CI为曝光数据上固定规则的描述性区间，不校正此前开发选择。
[可复算结果](../../evidence/research/x1012s2_combo500_network_v2.json)。
封存SHA `33d51613f23280bcae6e4338a2312dfbb86a17edaffd76de2dd123dea8a428d6`；配置SHA `c763da1ca2856a7ebcce51bab9942ecb864f94d9eb9269723c85eed16684811a`。
新增编码0、拟合0。执行时间只覆盖既有mask重放，不能冒充完整方法延迟。

## 真实补回收支

四源网络的M&T补回分支共补回1,098,212个TP、新增1,109,364个FP，没有删除像素；对原r00097的十域均值增益+0.24162pp，95%区间[+0.01825,+0.45413]。PASCAL-Part+0.7333、SUIM+0.6103、LVIS+0.5246、ISIC+0.4110pp，COCO−0.1592、Lung−0.0176pp，Fundus恒等。像素增删不能简单等权替代完整class IoU。

六源网络相对原规则补回1,342,295 TP、新增1,592,795 FP，同时删除59,227 TP、移除204,624 FP。它比四源均值仅+0.00867pp，区间[−0.19847,+0.24563]，未建立额外来源的稳定收益。四源相对FoRIS仍有COCO/LVIS/SUIM三个负域，不能报告逐域全面领先；ISIC在这一曝光500上转为+0.1577pp。

所有结论来自既有500发展复现；未修改两条冻结网络，也未读取新confirm1000。原v1输出、配置和报告保持原字节。
