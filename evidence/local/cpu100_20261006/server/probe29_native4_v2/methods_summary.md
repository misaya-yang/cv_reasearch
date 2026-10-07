# Actual native1024 four-case CPU probe

30 methods and all controls completed; shared DINO prototype 28.984250. Native FP32 final-LN 1024, no Part1. Four reused cases are descriptive, not population confirmation. Inference wall 124.25 s; scores are class-summed mean IoU.

| Method | Score | Δ prototype | Strongest declared control on these 4 | Δ control | Mean CPU time s |
|---|---:|---:|---|---:|---:|
| inv_huber_reference_readout | 40.4669 | +11.4826 | inv_huber_ridge | -0.0967 | 0.648 |
| DR08 | 37.5062 | +8.5220 | DR_control_average_logistic | -1.4076 | 0.304 |
| DR04 | 37.4078 | +8.4236 | DR_control_average_logistic | -1.5060 | 0.344 |
| DR03 | 36.6139 | +7.6296 | DR_control_average_logistic | -2.2999 | 0.328 |
| inv_adversarial_channel_support | 36.5016 | +7.5173 | inv_adversarial_constant | +0.5372 | 0.415 |
| QP02 | 35.9029 | +6.9186 | QP_center_prototype | +1.6987 | 0.641 |
| QP01 | 35.7922 | +6.8080 | QP_center_prototype | +1.5881 | 0.577 |
| cross_image_csls_hubness | 34.6458 | +5.6616 | dino_prototype.control | +5.6616 | 1.121 |
| RGB02 | 32.8750 | +3.8907 | RGB02.full_power | +0.1962 | 0.440 |
| DR01 | 30.4456 | +1.4613 | DR_control_average_logistic | -8.4682 | 0.366 |
| inv_local_affine_reconstruction | 30.2325 | +1.2482 | inv_affine_mean | -0.0718 | 1.094 |
| inv_multiscale_feature_consensus | 29.9770 | +0.9927 | inv_multiscale_score_pool | -0.6622 | 0.702 |
| inv_reference_mad_winsor | 29.7572 | +0.7730 | dino_prototype.control | +0.7730 | 1.515 |
| RGB04 | 29.3625 | +0.3783 | RGB04.ordinal | -2.6225 | 0.478 |
| inv_local_lowrank_reconstruction | 29.2787 | +0.2945 | inv_lowrank_mean | -0.1148 | 1.436 |
| inv_spatial_geomedian | 29.1834 | +0.1991 | inv_spatial_score_median | -0.3976 | 2.324 |
| DR05 | 29.1483 | +0.1640 | DR_control_average_logistic | -9.7655 | 0.487 |
| DR07 | 29.1053 | +0.1211 | DR_control_average_logistic | -9.8085 | 0.340 |
| local_002 | 29.0748 | +0.0906 | local_002.unary_control | -0.4036 | 3.469 |
| cross_image_background_anchor_shift | 28.9843 | +0.0000 | cross_image_whole_mean_shift_control | -8.0269 | 1.080 |
| RGB03 | 28.7049 | -0.2793 | RGB03.ordinal | -4.1700 | 0.529 |
| DR06 | 28.5806 | -0.4037 | DR_control_average_logistic | -10.3332 | 0.385 |
| cross_image_free_column_gram_matching | 26.6055 | -2.3788 | dino_prototype.control | -2.3788 | 0.713 |
| ref_ordinal_copula | 24.9959 | -3.9883 | dino_prototype.control | -3.9883 | 1.289 |
| RGB05 | 22.9314 | -6.0529 | dino_prototype.control | -6.0529 | 0.706 |
| DR02 | 18.9347 | -10.0495 | DR_control_average_logistic | -19.9791 | 0.296 |
| RGB01 | 17.5579 | -11.4263 | dino_prototype.control | -11.4263 | 0.378 |
| local_001 | 14.3401 | -14.6441 | dino_prototype.control | -14.6441 | 1.226 |
| ref_joint_channel_code | 6.1412 | -22.8430 | dino_prototype.control | -22.8430 | 0.342 |
| ref_local_support_radius | 0.9745 | -28.0097 | ref_support_nearest_control | -29.1460 | 0.407 |

The JSON includes every matched control, paired bootstrap intervals, four edit counts and exact per-class IoU changes. The strongest control above is a descriptive comparison, not another selected method. Global pixel purity is not class-macro gain. No method was silently excluded; zero failed/unavailable/missing arms.

## Per-class exact edit accounts

| Method | Class | Add TP | Add FP | Delete TP | Delete FP | Class Δ IoU |
|---|---:|---:|---:|---:|---:|---:|
| ref_ordinal_copula | 72 | 0 | 35347 | 0 | 0 | +0.00000 |
| ref_ordinal_copula | 73 | 0 | 0 | 3360 | 28267 | +29.97675 |
| ref_ordinal_copula | 74 | 0 | 0 | 10396 | 15224 | -36.14454 |
| ref_ordinal_copula | 75 | 0 | 0 | 15988 | 14063 | -9.78555 |
| ref_local_support_radius | 72 | 0 | 0 | 0 | 25824 | +0.00000 |
| ref_local_support_radius | 73 | 0 | 0 | 9303 | 30048 | -23.64108 |
| ref_local_support_radius | 74 | 0 | 0 | 11049 | 15224 | -42.05458 |
| ref_local_support_radius | 75 | 0 | 88 | 50618 | 47405 | -46.34333 |
| ref_joint_channel_code | 72 | 18 | 21268 | 0 | 17294 | +0.03076 |
| ref_joint_channel_code | 73 | 0 | 91157 | 1466 | 18494 | -16.64463 |
| ref_joint_channel_code | 74 | 0 | 30819 | 3772 | 14327 | -25.03833 |
| ref_joint_channel_code | 75 | 0 | 1652 | 52559 | 46112 | -49.71983 |
| local_001 | 72 | 18470 | 149447 | 0 | 15068 | +9.77616 |
| local_001 | 73 | 0 | 59266 | 4134 | 12453 | -17.64205 |
| local_001 | 74 | 0 | 54453 | 7468 | 11380 | -36.89062 |
| local_001 | 75 | 3319 | 38380 | 5663 | 4881 | -13.82005 |
| local_002 | 72 | 0 | 9033 | 0 | 11953 | +0.00000 |
| local_002 | 73 | 0 | 11 | 1807 | 17255 | +10.26674 |
| local_002 | 74 | 0 | 2524 | 412 | 4679 | +2.04941 |
| local_002 | 75 | 54 | 2929 | 16888 | 14038 | -11.95378 |
| QP01 | 72 | 0 | 132403 | 0 | 22 | +0.00000 |
| QP01 | 73 | 0 | 8 | 9 | 11973 | +10.29597 |
| QP01 | 74 | 0 | 1244 | 0 | 8205 | +15.15855 |
| QP01 | 75 | 1408 | 1316 | 841 | 3822 | +1.77738 |
| QP02 | 72 | 0 | 76022 | 0 | 9846 | +0.00000 |
| QP02 | 73 | 0 | 29 | 9 | 11741 | +9.98532 |
| QP02 | 74 | 0 | 1318 | 0 | 8195 | +14.91077 |
| QP02 | 75 | 2835 | 2392 | 324 | 3171 | +2.77833 |
| cross_image_csls_hubness | 72 | 0 | 35449 | 0 | 3886 | +0.00000 |
| cross_image_csls_hubness | 73 | 0 | 0 | 109 | 13456 | +11.86385 |
| cross_image_csls_hubness | 74 | 0 | 282 | 0 | 10198 | +25.49448 |
| cross_image_csls_hubness | 75 | 121 | 4014 | 16459 | 6419 | -14.71202 |
| cross_image_background_anchor_shift | 72 | 0 | 0 | 0 | 0 | +0.00000 |
| cross_image_background_anchor_shift | 73 | 0 | 0 | 0 | 0 | +0.00000 |
| cross_image_background_anchor_shift | 74 | 0 | 0 | 0 | 0 | +0.00000 |
| cross_image_background_anchor_shift | 75 | 0 | 0 | 0 | 0 | +0.00000 |
| cross_image_free_column_gram_matching | 72 | 0 | 14089 | 0 | 9893 | +0.00000 |
| cross_image_free_column_gram_matching | 73 | 0 | 87 | 0 | 11150 | +9.24566 |
| cross_image_free_column_gram_matching | 74 | 0 | 5997 | 0 | 1638 | -5.98446 |
| cross_image_free_column_gram_matching | 75 | 28 | 1522 | 14055 | 3072 | -12.77630 |
| RGB01 | 72 | 5934 | 84636 | 0 | 13115 | +4.70687 |
| RGB01 | 73 | 0 | 21399 | 0 | 540 | -8.19015 |
| RGB01 | 74 | 0 | 21886 | 91 | 19 | -19.29181 |
| RGB01 | 75 | 296 | 7055 | 28079 | 20421 | -22.93013 |
| RGB02 | 72 | 0 | 6722 | 0 | 10860 | +0.00000 |
| RGB02 | 73 | 0 | 4449 | 1676 | 24002 | +14.88302 |
| RGB02 | 74 | 0 | 0 | 0 | 0 | +0.00000 |
| RGB02 | 75 | 1423 | 1737 | 428 | 1188 | +0.67979 |
| RGB03 | 72 | 209 | 23707 | 0 | 3135 | +0.27821 |
| RGB03 | 73 | 0 | 10208 | 386 | 10030 | -1.08295 |
| RGB03 | 74 | 0 | 0 | 0 | 0 | +0.00000 |
| RGB03 | 75 | 263 | 484 | 565 | 430 | -0.31257 |
| RGB04 | 72 | 202 | 25501 | 0 | 1814 | +0.25819 |
| RGB04 | 73 | 0 | 16840 | 7 | 2655 | -6.27706 |
| RGB04 | 74 | 0 | 779 | 938 | 8313 | +11.90241 |
| RGB04 | 75 | 1572 | 11064 | 1591 | 1078 | -4.37048 |
| RGB05 | 72 | 2431 | 79124 | 0 | 5662 | +1.89904 |
| RGB05 | 73 | 0 | 24311 | 1950 | 15356 | -8.41936 |
| RGB05 | 74 | 0 | 18066 | 348 | 5129 | -14.76307 |
| RGB05 | 75 | 1254 | 3286 | 3905 | 2376 | -2.92801 |
| DR01 | 72 | 0 | 0 | 0 | 26 | +0.00000 |
| DR01 | 73 | 0 | 0 | 0 | 2363 | +1.51032 |
| DR01 | 74 | 0 | 0 | 0 | 2469 | +4.36199 |
| DR01 | 75 | 0 | 0 | 63 | 69 | -0.02694 |
| DR02 | 72 | 0 | 71739 | 0 | 3998 | +0.00000 |
| DR02 | 73 | 0 | 82664 | 29 | 3340 | -15.82646 |
| DR02 | 74 | 0 | 39166 | 0 | 0 | -25.17015 |
| DR02 | 75 | 3349 | 5849 | 73 | 1077 | +0.79850 |
| DR03 | 72 | 0 | 0 | 0 | 16643 | +0.00000 |
| DR03 | 73 | 0 | 0 | 56 | 14682 | +13.84322 |
| DR03 | 74 | 0 | 0 | 3528 | 14894 | +24.04086 |
| DR03 | 75 | 0 | 27 | 9345 | 3743 | -7.36549 |
| DR04 | 72 | 0 | 0 | 0 | 13057 | +0.00000 |
| DR04 | 73 | 0 | 0 | 5 | 6041 | +4.27246 |
| DR04 | 74 | 0 | 0 | 513 | 11837 | +30.92963 |
| DR04 | 75 | 18 | 307 | 3523 | 4243 | -1.50778 |
| DR05 | 72 | 0 | 1 | 0 | 119 | +0.00000 |
| DR05 | 73 | 0 | 16 | 0 | 36 | +0.01202 |
| DR05 | 74 | 0 | 0 | 0 | 362 | +0.58754 |
| DR05 | 75 | 3 | 12 | 23 | 170 | +0.05651 |
| DR06 | 72 | 326 | 82052 | 0 | 65 | +0.23876 |
| DR06 | 73 | 0 | 0 | 0 | 0 | +0.00000 |
| DR06 | 74 | 0 | 0 | 0 | 0 | +0.00000 |
| DR06 | 75 | 2744 | 9705 | 0 | 3 | -1.85341 |
| DR07 | 72 | 0 | 105 | 0 | 8 | +0.00000 |
| DR07 | 73 | 0 | 459 | 0 | 603 | +0.08683 |
| DR07 | 74 | 0 | 8 | 0 | 253 | +0.39586 |
| DR07 | 75 | 0 | 2 | 30 | 65 | +0.00157 |
| DR08 | 72 | 0 | 903 | 0 | 15864 | +0.00000 |
| DR08 | 73 | 0 | 0 | 299 | 17930 | +18.39244 |
| DR08 | 74 | 0 | 0 | 2852 | 14690 | +28.71292 |
| DR08 | 75 | 81 | 3026 | 15712 | 8213 | -13.01756 |
| inv_adversarial_channel_support | 72 | 0 | 0 | 0 | 25651 | +0.00000 |
| inv_adversarial_channel_support | 73 | 0 | 0 | 756 | 22328 | +26.56746 |
| inv_adversarial_channel_support | 74 | 0 | 0 | 3238 | 13963 | +21.39790 |
| inv_adversarial_channel_support | 75 | 0 | 0 | 30256 | 35311 | -17.89610 |
| inv_reference_mad_winsor | 72 | 0 | 0 | 0 | 1140 | +0.00000 |
| inv_reference_mad_winsor | 73 | 0 | 0 | 0 | 1012 | +0.62403 |
| inv_reference_mad_winsor | 74 | 0 | 7 | 0 | 1464 | +2.46911 |
| inv_reference_mad_winsor | 75 | 1 | 10 | 47 | 99 | -0.00122 |
| inv_multiscale_feature_consensus | 72 | 0 | 342 | 0 | 4628 | +0.00000 |
| inv_multiscale_feature_consensus | 73 | 0 | 985 | 0 | 1851 | +0.53198 |
| inv_multiscale_feature_consensus | 74 | 0 | 707 | 0 | 3879 | +5.77452 |
| inv_multiscale_feature_consensus | 75 | 522 | 2853 | 2330 | 1496 | -2.33560 |
| inv_spatial_geomedian | 72 | 0 | 974 | 0 | 2231 | +0.00000 |
| inv_spatial_geomedian | 73 | 0 | 410 | 0 | 1141 | +0.44748 |
| inv_spatial_geomedian | 74 | 0 | 820 | 0 | 1542 | +1.18835 |
| inv_spatial_geomedian | 75 | 428 | 1533 | 1096 | 1097 | -0.83938 |
| inv_local_affine_reconstruction | 72 | 0 | 938 | 0 | 6947 | +0.00000 |
| inv_local_affine_reconstruction | 73 | 0 | 1620 | 0 | 2879 | +0.78137 |
| inv_local_affine_reconstruction | 74 | 0 | 1213 | 64 | 5980 | +9.02419 |
| inv_local_affine_reconstruction | 75 | 618 | 4691 | 4937 | 3049 | -4.81257 |
| inv_local_lowrank_reconstruction | 72 | 0 | 305 | 0 | 1076 | +0.00000 |
| inv_local_lowrank_reconstruction | 73 | 0 | 261 | 0 | 609 | +0.21093 |
| inv_local_lowrank_reconstruction | 74 | 0 | 419 | 0 | 1029 | +0.99962 |
| inv_local_lowrank_reconstruction | 75 | 178 | 386 | 256 | 473 | -0.03261 |
| inv_huber_reference_readout | 72 | 0 | 0 | 0 | 17468 | +0.00000 |
| inv_huber_reference_readout | 73 | 0 | 0 | 981 | 23066 | +27.46116 |
| inv_huber_reference_readout | 74 | 0 | 0 | 3741 | 15206 | +23.97957 |
| inv_huber_reference_readout | 75 | 4 | 120 | 7227 | 3303 | -5.51025 |
