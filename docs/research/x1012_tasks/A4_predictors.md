# A4_predictors 实验报告

只评估A4原1360个固定已曝光开发任务；参考input hash与R mask hash为缓存键。七个描述量全部先封存，生产阶段不读取query GT或A4 metric report。
按查询照片的固定SHA256切半；所有同照片任务属于同一半。每个单变量只在训练半拟合一个跨十域的方向与21分位候选阈值，不按域分支；缺失走训练半较优常数。
该表仅为留出报告半：选择结果的整数指标直接复用原A4已评分两臂，未重新读GT。阈值与report choices在留出评测前分别封存。
AUROC按|INDN IoU-IND0 IoU|加权；IoU ties为零权，descriptor score ties获0.5配对质量；方向只从训练半取得。缺少hardFG/hardBG的LOO、空IND0的面积比均记缺失。
所有选择器均报告；不按留出分数挑最佳，也不把该开发切半视为confirm。只输出.65/.75判据，不启动连续调制或生成新配方。
固定描述量定义及参数见本任务config.json；FP16 log_e/back_f按现存字段使用，E<median可能近似常数仍如实保留。

主指标沿用计划lab.Meter：每个(fold,global_class)池化I/U，再对观察槽等权平均；CLI1024。不同于先平均fold的指标。
GT/预测忽略iSAID ignore像素。FN/GT、FP/GT用像素总量之比；面积误差为空预测时为∞。95%区间按query照片配对bootstrap，seed20261012，1000次，每次重新聚合。全部为开发/诊断结果，未评分confirm。

|域|方法|n|class mIoU|逐例均值|FN/GT|FP/GT|面积误差中位|对FoRIS pp|配对95%区间|
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
|coco|FoRIS|74|64.492|65.303|0.205|0.271|0.188|+0.000|[+0.000,+0.000]|
|coco|IND0|74|52.589|56.848|0.167|0.599|0.426|-11.904|[-14.879,-6.209]|
|coco|INDN|74|60.332|61.143|0.376|0.181|0.289|-4.161|[-8.828,+0.462]|
|coco|selector_ref_loo_auroc|74|53.578|57.361|0.166|0.518|0.426|-10.914|[-13.668,-5.836]|
|coco|selector_ref_background_fg_mass|74|57.937|60.185|0.261|0.475|0.322|-6.556|[-11.016,-1.154]|
|coco|selector_rho|74|57.948|59.859|0.186|0.527|0.305|-6.544|[-11.524,-1.535]|
|coco|selector_ref_fg_area|74|56.765|58.608|0.178|0.548|0.422|-7.727|[-12.879,-2.581]|
|coco|selector_query_e_below_median_fraction|74|52.589|56.848|0.167|0.599|0.426|-11.904|[-14.879,-6.209]|
|coco|selector_indn_ind0_area_ratio|74|59.057|59.853|0.187|0.369|0.383|-5.436|[-10.736,-0.586]|
|coco|selector_j_difference|74|60.438|62.650|0.185|0.477|0.295|-4.054|[-7.846,+0.740]|
|lvis|FoRIS|75|44.551|44.551|0.184|0.467|1.038|+0.000|[+0.000,+0.000]|
|lvis|IND0|75|38.949|38.949|0.197|0.714|0.735|-5.602|[-8.784,-2.765]|
|lvis|INDN|75|43.946|43.946|0.341|0.265|0.537|-0.604|[-3.910,+2.725]|
|lvis|selector_ref_loo_auroc|75|38.517|38.517|0.216|0.695|0.697|-6.034|[-9.240,-3.158]|
|lvis|selector_ref_background_fg_mass|75|42.953|42.953|0.261|0.477|0.537|-1.597|[-5.213,+2.154]|
|lvis|selector_rho|75|42.176|42.176|0.223|0.578|0.557|-2.375|[-5.602,+0.838]|
|lvis|selector_ref_fg_area|75|39.472|39.472|0.208|0.668|0.654|-5.079|[-8.363,-2.098]|
|lvis|selector_query_e_below_median_fraction|75|38.949|38.949|0.197|0.714|0.735|-5.602|[-8.784,-2.765]|
|lvis|selector_indn_ind0_area_ratio|75|42.554|42.554|0.223|0.474|0.679|-1.997|[-5.112,+0.804]|
|lvis|selector_j_difference|75|41.701|41.701|0.201|0.596|0.735|-2.850|[-6.184,+0.191]|
|pascal_part|FoRIS|80|52.293|55.858|0.243|0.201|0.387|+0.000|[+0.000,+0.000]|
|pascal_part|IND0|80|57.537|57.650|0.211|0.177|0.289|+5.244|[+0.230,+7.967]|
|pascal_part|INDN|80|49.291|45.875|0.525|0.088|0.572|-3.002|[-10.462,-0.755]|
|pascal_part|selector_ref_loo_auroc|80|56.473|54.572|0.266|0.170|0.328|+4.180|[-1.213,+6.907]|
|pascal_part|selector_ref_background_fg_mass|80|56.694|56.798|0.228|0.151|0.255|+4.400|[-0.520,+7.302]|
|pascal_part|selector_rho|80|55.723|55.973|0.232|0.169|0.282|+3.430|[-1.342,+6.153]|
|pascal_part|selector_ref_fg_area|80|57.612|57.840|0.211|0.178|0.289|+5.319|[+0.178,+7.969]|
|pascal_part|selector_query_e_below_median_fraction|80|57.537|57.650|0.211|0.177|0.289|+5.244|[+0.230,+7.967]|
|pascal_part|selector_indn_ind0_area_ratio|80|54.284|51.752|0.354|0.146|0.391|+1.991|[-4.359,+5.033]|
|pascal_part|selector_j_difference|80|57.118|57.186|0.223|0.163|0.304|+4.825|[-0.582,+7.875]|
|paco_part|FoRIS|78|41.937|41.744|0.299|0.525|0.882|+0.000|[+0.000,+0.000]|
|paco_part|IND0|78|46.893|46.939|0.354|0.327|0.531|+4.956|[+1.737,+8.131]|
|paco_part|INDN|78|41.996|42.269|0.599|0.184|0.639|+0.060|[-4.717,+4.773]|
|paco_part|selector_ref_loo_auroc|78|46.893|46.939|0.354|0.326|0.512|+4.956|[+1.737,+8.131]|
|paco_part|selector_ref_background_fg_mass|78|46.897|46.984|0.408|0.298|0.484|+4.960|[+1.081,+8.647]|
|paco_part|selector_rho|78|47.013|47.291|0.372|0.302|0.569|+5.077|[+1.419,+8.657]|
|paco_part|selector_ref_fg_area|78|47.345|47.327|0.355|0.327|0.512|+5.408|[+1.936,+8.647]|
|paco_part|selector_query_e_below_median_fraction|78|46.893|46.939|0.354|0.327|0.531|+4.956|[+1.737,+8.131]|
|paco_part|selector_indn_ind0_area_ratio|78|43.306|43.058|0.504|0.251|0.658|+1.369|[-3.046,+5.592]|
|paco_part|selector_j_difference|78|47.544|47.789|0.356|0.314|0.522|+5.607|[+2.399,+8.886]|
|suim|FoRIS|81|59.215|59.277|0.263|0.290|0.328|+0.000|[+0.000,+0.000]|
|suim|IND0|81|53.399|53.965|0.275|0.344|0.485|-5.816|[-12.012,+0.044]|
|suim|INDN|81|48.412|50.267|0.541|0.137|0.671|-10.803|[-15.519,-5.871]|
|suim|selector_ref_loo_auroc|81|53.679|54.254|0.278|0.324|0.473|-5.536|[-11.740,+0.309]|
|suim|selector_ref_background_fg_mass|81|53.206|53.984|0.354|0.280|0.521|-6.009|[-11.145,-1.359]|
|suim|selector_rho|81|50.917|53.679|0.294|0.320|0.508|-8.298|[-13.786,-2.679]|
|suim|selector_ref_fg_area|81|50.962|53.338|0.289|0.335|0.571|-8.253|[-13.689,-2.388]|
|suim|selector_query_e_below_median_fraction|81|53.399|53.965|0.275|0.344|0.485|-5.816|[-12.012,+0.044]|
|suim|selector_indn_ind0_area_ratio|81|55.240|55.167|0.337|0.250|0.473|-3.975|[-9.575,+0.939]|
|suim|selector_j_difference|81|57.314|56.874|0.283|0.301|0.411|-1.901|[-6.341,+2.589]|
|isaid|FoRIS|43|58.985|56.693|0.207|0.407|0.450|+0.000|[+0.000,+0.000]|
|isaid|IND0|43|59.921|56.982|0.162|0.481|0.381|+0.936|[-2.415,+3.704]|
|isaid|INDN|43|62.187|57.624|0.282|0.205|0.304|+3.202|[-2.414,+7.984]|
|isaid|selector_ref_loo_auroc|43|59.876|56.831|0.169|0.473|0.381|+0.891|[-2.403,+3.546]|
|isaid|selector_ref_background_fg_mass|43|57.001|55.488|0.208|0.462|0.413|-1.984|[-5.997,+2.555]|
|isaid|selector_rho|43|61.663|57.206|0.204|0.303|0.381|+2.678|[-1.199,+6.521]|
|isaid|selector_ref_fg_area|43|59.646|56.862|0.166|0.476|0.381|+0.662|[-2.753,+3.538]|
|isaid|selector_query_e_below_median_fraction|43|59.921|56.982|0.162|0.481|0.381|+0.936|[-2.415,+3.704]|
|isaid|selector_indn_ind0_area_ratio|43|61.996|57.348|0.166|0.364|0.381|+3.011|[-0.411,+6.333]|
|isaid|selector_j_difference|43|64.571|58.972|0.165|0.375|0.259|+5.586|[-0.731,+9.273]|
|isic|FoRIS|75|64.409|65.766|0.217|0.227|0.382|+0.000|[+0.000,+0.000]|
|isic|IND0|75|61.358|61.511|0.189|0.333|0.338|-3.051|[-8.064,+1.693]|
|isic|INDN|75|52.698|59.669|0.423|0.092|0.486|-11.711|[-18.312,-4.420]|
|isic|selector_ref_loo_auroc|75|60.719|60.776|0.202|0.324|0.338|-3.690|[-8.922,+1.237]|
|isic|selector_ref_background_fg_mass|75|55.621|60.726|0.270|0.290|0.333|-8.788|[-13.809,-3.798]|
|isic|selector_rho|75|60.914|61.339|0.194|0.331|0.338|-3.495|[-8.568,+1.362]|
|isic|selector_ref_fg_area|75|58.793|61.214|0.220|0.322|0.330|-5.616|[-9.034,-2.678]|
|isic|selector_query_e_below_median_fraction|75|61.358|61.511|0.189|0.333|0.338|-3.051|[-8.064,+1.693]|
|isic|selector_indn_ind0_area_ratio|75|58.221|60.765|0.297|0.194|0.387|-6.188|[-13.360,+0.370]|
|isic|selector_j_difference|75|60.618|63.681|0.240|0.258|0.326|-3.791|[-6.909,-0.844]|
|lung|FoRIS|74|87.287|87.444|0.078|0.056|0.072|+0.000|[+0.000,+0.000]|
|lung|IND0|74|86.393|86.404|0.092|0.051|0.074|-0.894|[-1.745,+0.131]|
|lung|INDN|74|86.142|86.203|0.113|0.030|0.105|-1.145|[-2.208,+0.014]|
|lung|selector_ref_loo_auroc|74|86.393|86.404|0.092|0.051|0.074|-0.894|[-1.745,+0.131]|
|lung|selector_ref_background_fg_mass|74|86.393|86.404|0.092|0.051|0.074|-0.894|[-1.745,+0.131]|
|lung|selector_rho|74|86.393|86.404|0.092|0.051|0.074|-0.894|[-1.745,+0.131]|
|lung|selector_ref_fg_area|74|86.393|86.404|0.092|0.051|0.074|-0.894|[-1.745,+0.131]|
|lung|selector_query_e_below_median_fraction|74|86.393|86.404|0.092|0.051|0.074|-0.894|[-1.745,+0.131]|
|lung|selector_indn_ind0_area_ratio|74|86.393|86.404|0.092|0.051|0.074|-0.894|[-1.745,+0.131]|
|lung|selector_j_difference|74|86.393|86.404|0.092|0.051|0.074|-0.894|[-1.745,+0.131]|
|fundus|FoRIS|60|19.225|20.001|0.282|2.732|1.778|+0.000|[+0.000,+0.000]|
|fundus|IND0|60|28.086|27.578|0.541|0.633|0.218|+8.861|[+7.463,+10.021]|
|fundus|INDN|60|27.416|26.696|0.569|0.573|0.283|+8.191|[+6.855,+9.359]|
|fundus|selector_ref_loo_auroc|60|28.086|27.578|0.541|0.633|0.218|+8.861|[+7.463,+10.021]|
|fundus|selector_ref_background_fg_mass|60|28.086|27.578|0.541|0.633|0.218|+8.861|[+7.463,+10.021]|
|fundus|selector_rho|60|27.416|26.696|0.569|0.573|0.283|+8.191|[+6.855,+9.359]|
|fundus|selector_ref_fg_area|60|28.098|27.596|0.541|0.634|0.218|+8.873|[+7.463,+10.037]|
|fundus|selector_query_e_below_median_fraction|60|28.086|27.578|0.541|0.633|0.218|+8.861|[+7.463,+10.021]|
|fundus|selector_indn_ind0_area_ratio|60|28.108|27.533|0.543|0.625|0.218|+8.883|[+7.464,+10.049]|
|fundus|selector_j_difference|60|28.086|27.578|0.541|0.633|0.218|+8.861|[+7.463,+10.021]|
|deepglobe_road|FoRIS|56|33.089|33.787|0.455|0.647|0.657|+0.000|[+0.000,+0.000]|
|deepglobe_road|IND0|56|33.921|31.455|0.403|0.760|0.719|+0.833|[-2.904,+4.596]|
|deepglobe_road|INDN|56|34.327|33.242|0.483|0.505|0.587|+1.238|[-3.232,+5.329]|
|deepglobe_road|selector_ref_loo_auroc|56|33.693|31.160|0.419|0.726|0.688|+0.604|[-3.355,+4.465]|
|deepglobe_road|selector_ref_background_fg_mass|56|33.129|31.178|0.450|0.659|0.727|+0.040|[-4.019,+3.980]|
|deepglobe_road|selector_rho|56|34.327|33.242|0.483|0.505|0.587|+1.238|[-3.232,+5.329]|
|deepglobe_road|selector_ref_fg_area|56|33.357|31.430|0.456|0.631|0.746|+0.268|[-3.792,+4.196]|
|deepglobe_road|selector_query_e_below_median_fraction|56|33.921|31.455|0.403|0.760|0.719|+0.833|[-2.904,+4.596]|
|deepglobe_road|selector_indn_ind0_area_ratio|56|35.194|32.509|0.407|0.684|0.668|+2.105|[-1.501,+5.638]|
|deepglobe_road|selector_j_difference|56|34.820|32.307|0.403|0.715|0.665|+1.732|[-1.888,+5.320]|

|域|方法|<2% n/均值|2–20% n/均值|>20% n/均值|
|---|---|---|---|---|
|coco|FoRIS|23 / 48.804|34 / 73.797|17 / 70.637|
|coco|IND0|23 / 42.779|34 / 58.625|17 / 72.329|
|coco|INDN|23 / 51.698|34 / 71.658|17 / 52.889|
|coco|selector_ref_loo_auroc|23 / 42.951|34 / 59.565|17 / 72.448|
|coco|selector_ref_background_fg_mass|23 / 49.513|34 / 66.145|17 / 62.702|
|coco|selector_rho|23 / 48.250|34 / 62.232|17 / 70.818|
|coco|selector_ref_fg_area|23 / 47.565|34 / 59.647|17 / 71.470|
|coco|selector_query_e_below_median_fraction|23 / 42.779|34 / 58.625|17 / 72.329|
|coco|selector_indn_ind0_area_ratio|23 / 50.063|34 / 61.416|17 / 69.970|
|coco|selector_j_difference|23 / 48.950|34 / 68.094|17 / 70.295|
|lvis|FoRIS|40 / 28.978|29 / 60.783|6 / 69.912|
|lvis|IND0|40 / 25.188|29 / 54.960|6 / 53.303|
|lvis|INDN|40 / 31.109|29 / 60.909|6 / 47.542|
|lvis|selector_ref_loo_auroc|40 / 24.812|29 / 54.361|6 / 53.303|
|lvis|selector_ref_background_fg_mass|40 / 29.289|29 / 61.006|6 / 46.795|
|lvis|selector_rho|40 / 29.403|29 / 57.492|6 / 53.297|
|lvis|selector_ref_fg_area|40 / 26.258|29 / 54.836|6 / 53.297|
|lvis|selector_query_e_below_median_fraction|40 / 25.188|29 / 54.960|6 / 53.303|
|lvis|selector_indn_ind0_area_ratio|40 / 30.060|29 / 57.563|6 / 53.303|
|lvis|selector_j_difference|40 / 28.774|29 / 57.132|6 / 53.303|
|pascal_part|FoRIS|7 / 22.917|37 / 48.175|36 / 70.159|
|pascal_part|IND0|7 / 35.705|37 / 48.383|36 / 71.442|
|pascal_part|INDN|7 / 44.670|37 / 42.940|36 / 49.125|
|pascal_part|selector_ref_loo_auroc|7 / 35.705|37 / 46.985|36 / 66.039|
|pascal_part|selector_ref_background_fg_mass|7 / 44.670|37 / 44.596|36 / 71.697|
|pascal_part|selector_rho|7 / 36.213|37 / 46.454|36 / 69.599|
|pascal_part|selector_ref_fg_area|7 / 36.151|37 / 48.709|36 / 71.442|
|pascal_part|selector_query_e_below_median_fraction|7 / 35.705|37 / 48.383|36 / 71.442|
|pascal_part|selector_indn_ind0_area_ratio|7 / 42.757|37 / 44.903|36 / 60.541|
|pascal_part|selector_j_difference|7 / 42.757|37 / 46.835|36 / 70.630|
|paco_part|FoRIS|11 / 29.798|45 / 36.304|22 / 58.842|
|paco_part|IND0|11 / 40.878|45 / 45.173|22 / 53.584|
|paco_part|INDN|11 / 50.456|45 / 46.297|22 / 29.939|
|paco_part|selector_ref_loo_auroc|11 / 40.878|45 / 45.173|22 / 53.584|
|paco_part|selector_ref_background_fg_mass|11 / 44.120|45 / 46.810|22 / 48.771|
|paco_part|selector_rho|11 / 44.999|45 / 45.580|22 / 51.938|
|paco_part|selector_ref_fg_area|11 / 43.850|45 / 45.119|22 / 53.584|
|paco_part|selector_query_e_below_median_fraction|11 / 40.878|45 / 45.173|22 / 53.584|
|paco_part|selector_indn_ind0_area_ratio|11 / 46.334|45 / 44.706|22 / 38.050|
|paco_part|selector_j_difference|11 / 40.878|45 / 46.646|22 / 53.584|
|suim|FoRIS|12 / 42.030|31 / 64.678|38 / 60.317|
|suim|IND0|12 / 33.665|31 / 53.431|38 / 60.811|
|suim|INDN|12 / 48.349|31 / 62.597|38 / 40.815|
|suim|selector_ref_loo_auroc|12 / 33.643|31 / 54.322|38 / 60.707|
|suim|selector_ref_background_fg_mass|12 / 43.122|31 / 58.668|38 / 53.593|
|suim|selector_rho|12 / 35.449|31 / 54.863|38 / 58.469|
|suim|selector_ref_fg_area|12 / 35.414|31 / 53.405|38 / 58.943|
|suim|selector_query_e_below_median_fraction|12 / 33.665|31 / 53.431|38 / 60.811|
|suim|selector_indn_ind0_area_ratio|12 / 46.655|31 / 57.159|38 / 56.230|
|suim|selector_j_difference|12 / 40.014|31 / 59.363|38 / 60.167|
|isaid|FoRIS|5 / 24.914|25 / 57.754|13 / 66.873|
|isaid|IND0|5 / 19.274|25 / 56.909|13 / 71.625|
|isaid|INDN|5 / 25.646|25 / 59.371|13 / 66.564|
|isaid|selector_ref_loo_auroc|5 / 19.274|25 / 56.951|13 / 71.044|
|isaid|selector_ref_background_fg_mass|5 / 19.274|25 / 56.133|13 / 68.176|
|isaid|selector_rho|5 / 21.729|25 / 57.696|13 / 69.909|
|isaid|selector_ref_fg_area|5 / 19.274|25 / 56.748|13 / 71.537|
|isaid|selector_query_e_below_median_fraction|5 / 19.274|25 / 56.909|13 / 71.625|
|isaid|selector_indn_ind0_area_ratio|5 / 21.249|25 / 57.144|13 / 71.625|
|isaid|selector_j_difference|5 / 21.249|25 / 57.951|13 / 75.446|
|isic|FoRIS|4 / 45.229|33 / 56.812|38 / 75.703|
|isic|IND0|4 / 17.618|33 / 48.822|38 / 77.151|
|isic|INDN|4 / 49.499|33 / 60.880|38 / 59.687|
|isic|selector_ref_loo_auroc|4 / 17.618|33 / 48.822|38 / 75.700|
|isic|selector_ref_background_fg_mass|4 / 31.795|33 / 51.593|38 / 71.704|
|isic|selector_rho|4 / 18.684|33 / 48.822|38 / 76.698|
|isic|selector_ref_fg_area|4 / 18.684|33 / 50.693|38 / 74.827|
|isic|selector_query_e_below_median_fraction|4 / 17.618|33 / 48.822|38 / 77.151|
|isic|selector_indn_ind0_area_ratio|4 / 44.469|33 / 52.858|38 / 69.347|
|isic|selector_j_difference|4 / 44.469|33 / 54.352|38 / 73.804|
|lung|FoRIS|0 / —|7 / 85.586|67 / 87.638|
|lung|IND0|0 / —|7 / 84.231|67 / 86.631|
|lung|INDN|0 / —|7 / 85.268|67 / 86.300|
|lung|selector_ref_loo_auroc|0 / —|7 / 84.231|67 / 86.631|
|lung|selector_ref_background_fg_mass|0 / —|7 / 84.231|67 / 86.631|
|lung|selector_rho|0 / —|7 / 84.231|67 / 86.631|
|lung|selector_ref_fg_area|0 / —|7 / 84.231|67 / 86.631|
|lung|selector_query_e_below_median_fraction|0 / —|7 / 84.231|67 / 86.631|
|lung|selector_indn_ind0_area_ratio|0 / —|7 / 84.231|67 / 86.631|
|lung|selector_j_difference|0 / —|7 / 84.231|67 / 86.631|
|fundus|FoRIS|0 / —|60 / 20.001|0 / —|
|fundus|IND0|0 / —|60 / 27.578|0 / —|
|fundus|INDN|0 / —|60 / 26.696|0 / —|
|fundus|selector_ref_loo_auroc|0 / —|60 / 27.578|0 / —|
|fundus|selector_ref_background_fg_mass|0 / —|60 / 27.578|0 / —|
|fundus|selector_rho|0 / —|60 / 26.696|0 / —|
|fundus|selector_ref_fg_area|0 / —|60 / 27.596|0 / —|
|fundus|selector_query_e_below_median_fraction|0 / —|60 / 27.578|0 / —|
|fundus|selector_indn_ind0_area_ratio|0 / —|60 / 27.533|0 / —|
|fundus|selector_j_difference|0 / —|60 / 27.578|0 / —|
|deepglobe_road|FoRIS|28 / 32.357|27 / 35.085|1 / 38.800|
|deepglobe_road|IND0|28 / 26.330|27 / 36.269|1 / 44.977|
|deepglobe_road|INDN|28 / 29.397|27 / 36.861|1 / 43.190|
|deepglobe_road|selector_ref_loo_auroc|28 / 26.370|27 / 35.616|1 / 44.977|
|deepglobe_road|selector_ref_background_fg_mass|28 / 26.393|27 / 35.696|1 / 43.190|
|deepglobe_road|selector_rho|28 / 29.397|27 / 36.861|1 / 43.190|
|deepglobe_road|selector_ref_fg_area|28 / 26.571|27 / 36.034|1 / 43.190|
|deepglobe_road|selector_query_e_below_median_fraction|28 / 26.330|27 / 36.269|1 / 44.977|
|deepglobe_road|selector_indn_ind0_area_ratio|28 / 27.920|27 / 36.807|1 / 44.977|
|deepglobe_road|selector_j_difference|28 / 27.260|27 / 37.071|1 / 44.977|

完整逐例整数、照片分组和诊断产物：`/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/x1012_A4_predictors/report.json`。

## 描述量与AUROC

|描述量|train拟合规则|留出AUROC|缺失n|不同值n|
|---|---|---:|---:|---:|
|ref_loo_auroc|INDN if -1*(x-0.997979)>0|0.399|38|490|
|ref_background_fg_mass|INDN if -1*(x-0.0121195)>0|0.655|0|678|
|rho|INDN if 1*(x-0.183556)>0|0.666|0|641|
|ref_fg_area|INDN if -1*(x-0.0211964)>0|0.661|0|677|
|query_e_below_median_fraction|constant IND0|0.497|0|24|
|indn_ind0_area_ratio|INDN if -1*(x-0.399748)>0|0.525|0|696|
|j_difference|INDN if 1*(x-0.0400137)>0|0.836|0|644|

## 判据

全部七量留出AUROC均低于.65才关闭参考统计切换；有量高于.75才具备后续连续调制资格。不可估计的AUROC不伪装成低于阈值。

{
  "state": "SIGNAL_ABOVE_075_CONTINUOUS_MODULATION_ELIGIBLE",
  "n": 1360,
  "train_n": 664,
  "report_n": 696,
  "per_half_domain_n": {
    "train": {
      "coco": 76,
      "lvis": 75,
      "pascal_part": 70,
      "paco_part": 72,
      "suim": 69,
      "isaid": 17,
      "isic": 75,
      "lung": 76,
      "fundus": 90,
      "deepglobe_road": 44
    },
    "report": {
      "coco": 74,
      "lvis": 75,
      "pascal_part": 80,
      "paco_part": 78,
      "suim": 81,
      "isaid": 43,
      "isic": 75,
      "lung": 74,
      "fundus": 60,
      "deepglobe_road": 56
    }
  },
  "heldout_train_oriented_weighted_auroc": {
    "ref_loo_auroc": 0.3994284925498188,
    "ref_background_fg_mass": 0.6546409780281766,
    "rho": 0.6663549420002176,
    "ref_fg_area": 0.6612206330609849,
    "query_e_below_median_fraction": 0.49670738407418497,
    "indn_ind0_area_ratio": 0.5246724795150686,
    "j_difference": 0.8355569581351057
  },
  "all_predictors_below_065": false,
  "reference_only_predictors_below_065": false,
  "above_075_predictors": [
    "j_difference"
  ],
  "continuous_modulation_run": false,
  "selector_chosen_on_report_half": false,
  "confirm_scored": false,
  "new_encoder_calls": 0,
  "additional_query_GT_reads": 0,
  "A4_metric_report_sha256": "b9322accd2d9d7aa408c17bdb0ee35bfc5f3c49871ccb2cfe1bcd7d9187119a9",
  "upper_report_preserved": "/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/x1012_A4/report.json"
}

原A4上界报告保持独立：/Users/misaya.yanghejazfs.com.au/paper_project/cv_data/a/x1012_A4/report.json。
