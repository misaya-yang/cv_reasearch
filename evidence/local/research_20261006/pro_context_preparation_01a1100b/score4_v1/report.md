# M1 首4真实四臂质量与成本检查

14文件seal、形状、原HW及两次renderer已先于GT核对。只计1024同packet GT，原尺寸未计分。
4例/4类别是活动检查，已暴露且非总体效果；bootstrap区间仅描述性。

| Arm | 1024 class-summed mIoU |
|---|---:|
| native | 36.316536 |
| ctx | 25.112759 |
| plain.control | 32.410145 |
| mean-unit.control | 24.975686 |
| uniform.control | 35.135329 |
| stored_mean.control | 35.684576 |
| rcg.control | 35.473585 |
| fine16.control | 34.911639 |
| fine64.control | 35.907872 |

| ctx versus | pp | paired descriptive 95% CI | up/down |
|---|---:|---|---|
| native | -11.203777 | [-16.30463694981299, -6.102917537118458] | 0/3 |
| plain.control | -7.297386 | [-12.382629196999133, -2.2121434619592364] | 0/3 |
| mean-unit.control | +0.137073 | [-7.443416672475546, 7.717563157212744] | 1/3 |
| uniform.control | -10.022571 | [-14.476247395774088, -5.568893767506024] | 0/4 |
| stored_mean.control | -10.571817 | [-14.321159187058317, -6.822475709896825] | 0/3 |
| rcg.control | -10.360826 | [-14.352973621211085, -6.368678324546661] | 0/3 |
| fine16.control | -9.798880 | [-13.242830894688602, -6.354929144032326] | 0/3 |
| fine64.control | -10.795113 | [-15.61246583908737, -5.977760542934396] | 0/3 |

推断4线程完整四臂共801.550s；peak RSS 3201520KiB。评分1线程、无新encoder前向。

完整逐例I/U及四种增删见episode_metrics.json；相对各简单/完整比较行的增删见pairwise_edits.json。
