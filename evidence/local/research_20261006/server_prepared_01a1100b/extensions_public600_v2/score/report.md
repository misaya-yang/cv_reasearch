# Cached-feature candidate results

| Arm | Class-summed mIoU |
|---|---:|
| native | 61.335314 |
| mean.control | 62.931546 |
| clipped_mean.control | 62.931392 |
| pro_reference_relations | 60.632462 |
| pro_relations_zero.control | 60.773632 |
| pro_relations_positive.control | 60.851557 |
| pro_relations_absolute.control | 60.757900 |
| pro_relations_pair_independent.control | 60.600817 |
| pro_relations_block.control | 60.677671 |
| reference_hull | 48.915345 |
| hull_nearest.control | 43.023069 |
| hull_centroid.control | 29.931126 |
| hull_subspace.control | 45.655276 |
| hull_affine.control | 47.955173 |
| constellation_local | 62.308618 |
| constellation_local_global_v1.control | 61.712587 |
| constellation_local_bag.control | 39.185986 |
| constellation_local_prior.control | 1.885569 |
| reference_triplet_relations | 60.778988 |
| triplet_zero.control | 60.773632 |
| triplet_absolute.control | 60.768773 |
| triplet_no_third.control | 60.773632 |
| triplet_pro_m2.control | 60.632462 |
| stored_mean.control | 62.931530 |
| rcg.control | 63.088942 |
| fine16.control | 63.464527 |
| fine64.control | 63.681184 |

Primary: pro_reference_relations
Exposure: reused public600 development data; no independent confirmation; ProM2 cached seed variant, local constellation correction version

| Primary versus | Gain, pp | Paired 95% CI |
|---|---:|---|
| native | -0.702852 | [-0.974832659442488, -0.5554851715836928] |
| mean.control | -2.299084 | [-3.005595780984459, -1.7605466376451906] |
| clipped_mean.control | -2.298930 | [-3.005660500904623, -1.760670465680333] |
| pro_relations_zero.control | -0.141169 | [-0.27333949108543043, 0.055183370169554324] |
| pro_relations_positive.control | -0.219094 | [-0.2956773210940312, -0.13355107720488904] |
| pro_relations_absolute.control | -0.125438 | [-0.2882454007389891, 0.11160968902081114] |
| pro_relations_pair_independent.control | +0.031645 | [-0.027763289043069326, 0.12670485555633423] |
| pro_relations_block.control | -0.045209 | [-0.10749661825504671, 0.010419353220563995] |
| hull_nearest.control | +17.609393 | [15.084193950311693, 19.571638332283406] |
| hull_centroid.control | +30.701336 | [28.289158077061376, 33.127814118869985] |
| hull_subspace.control | +14.977186 | [12.299244491568768, 16.936353209260172] |
| hull_affine.control | +12.677290 | [9.483634918819055, 14.845369628301167] |
| constellation_local_global_v1.control | -1.080124 | [-1.9850551680979582, -0.22131336345452302] |
| constellation_local_bag.control | +21.446476 | [19.880181433541548, 23.809382781148617] |
| constellation_local_prior.control | +58.746893 | [56.28617739170469, 61.65162389690457] |
| triplet_zero.control | -0.141169 | [-0.27333949108543043, 0.055183370169554324] |
| triplet_absolute.control | -0.136311 | [-0.2735315305950268, 0.07160602947849362] |
| triplet_no_third.control | -0.141169 | [-0.27333949108543043, 0.055183370169554324] |
| triplet_pro_m2.control | +0.000000 | [0.0, 0.0] |
| stored_mean.control | -2.299068 | [-3.00559284547705, -1.7604962647770621] |
| rcg.control | -2.456479 | [-3.1274367108005783, -1.8387835501895606] |
| fine16.control | -2.832065 | [-3.598926559842794, -2.241209048485636] |
| fine64.control | -3.048722 | [-4.171447300626037, -1.5139132647371993] |

| Predeclared candidate | Comparison | Gain, pp | Paired 95% CI |
|---|---|---:|---|
| pro_reference_relations | native | -0.702852 | [-0.974832659442488, -0.5554851715836928] |
| pro_reference_relations | mean.control | -2.299084 | [-3.005595780984459, -1.7605466376451906] |
| pro_reference_relations | clipped_mean.control | -2.298930 | [-3.005660500904623, -1.760670465680333] |
| pro_reference_relations | pro_relations_zero.control | -0.141169 | [-0.27333949108543043, 0.055183370169554324] |
| pro_reference_relations | pro_relations_positive.control | -0.219094 | [-0.2956773210940312, -0.13355107720488904] |
| pro_reference_relations | pro_relations_absolute.control | -0.125438 | [-0.2882454007389891, 0.11160968902081114] |
| pro_reference_relations | pro_relations_pair_independent.control | +0.031645 | [-0.027763289043069326, 0.12670485555633423] |
| pro_reference_relations | pro_relations_block.control | -0.045209 | [-0.10749661825504671, 0.010419353220563995] |
| pro_reference_relations | hull_nearest.control | +17.609393 | [15.084193950311693, 19.571638332283406] |
| pro_reference_relations | hull_centroid.control | +30.701336 | [28.289158077061376, 33.127814118869985] |
| pro_reference_relations | hull_subspace.control | +14.977186 | [12.299244491568768, 16.936353209260172] |
| pro_reference_relations | hull_affine.control | +12.677290 | [9.483634918819055, 14.845369628301167] |
| pro_reference_relations | constellation_local_global_v1.control | -1.080124 | [-1.9850551680979582, -0.22131336345452302] |
| pro_reference_relations | constellation_local_bag.control | +21.446476 | [19.880181433541548, 23.809382781148617] |
| pro_reference_relations | constellation_local_prior.control | +58.746893 | [56.28617739170469, 61.65162389690457] |
| pro_reference_relations | triplet_zero.control | -0.141169 | [-0.27333949108543043, 0.055183370169554324] |
| pro_reference_relations | triplet_absolute.control | -0.136311 | [-0.2735315305950268, 0.07160602947849362] |
| pro_reference_relations | triplet_no_third.control | -0.141169 | [-0.27333949108543043, 0.055183370169554324] |
| pro_reference_relations | triplet_pro_m2.control | +0.000000 | [0.0, 0.0] |
| pro_reference_relations | stored_mean.control | -2.299068 | [-3.00559284547705, -1.7604962647770621] |
| pro_reference_relations | rcg.control | -2.456479 | [-3.1274367108005783, -1.8387835501895606] |
| pro_reference_relations | fine16.control | -2.832065 | [-3.598926559842794, -2.241209048485636] |
| pro_reference_relations | fine64.control | -3.048722 | [-4.171447300626037, -1.5139132647371993] |
| reference_hull | native | -12.419969 | [-14.756926141480516, -9.433635650220877] |
| reference_hull | mean.control | -14.016201 | [-16.477373476785463, -10.940802323874637] |
| reference_hull | clipped_mean.control | -14.016048 | [-16.476832335780646, -10.941031232995748] |
| reference_hull | pro_relations_zero.control | -11.858287 | [-14.080010645304785, -8.783263900593777] |
| reference_hull | pro_relations_positive.control | -11.936212 | [-14.176397594338884, -8.872623198119712] |
| reference_hull | pro_relations_absolute.control | -11.842555 | [-14.06081066929083, -8.745499135860417] |
| reference_hull | pro_relations_pair_independent.control | -11.685472 | [-13.924304248253119, -8.629677584254724] |
| reference_hull | pro_relations_block.control | -11.762327 | [-14.024073358834599, -8.708054045272789] |
| reference_hull | hull_nearest.control | +5.892276 | [3.6470913263286917, 8.481050546650229] |
| reference_hull | hull_centroid.control | +18.984219 | [16.30463777196778, 22.24538269482666] |
| reference_hull | hull_subspace.control | +3.260069 | [1.3689752052636346, 5.3740001662327375] |
| reference_hull | hull_affine.control | +0.960172 | [0.19592195393695383, 1.555419085897743] |
| reference_hull | constellation_local_global_v1.control | -12.797242 | [-15.16860570770991, -9.73166694158208] |
| reference_hull | constellation_local_bag.control | +9.729359 | [7.698634777922875, 13.346509078902809] |
| reference_hull | constellation_local_prior.control | +47.029776 | [44.54512538324479, 50.698841225899415] |
| reference_hull | triplet_zero.control | -11.858287 | [-14.080010645304785, -8.783263900593777] |
| reference_hull | triplet_absolute.control | -11.853428 | [-14.07009805540461, -8.779689309390744] |
| reference_hull | triplet_no_third.control | -11.858287 | [-14.080010645304785, -8.783263900593777] |
| reference_hull | triplet_pro_m2.control | -11.717117 | [-13.994833081382943, -8.666035241845611] |
| reference_hull | stored_mean.control | -14.016185 | [-16.477369038134288, -10.940790105240902] |
| reference_hull | rcg.control | -14.173597 | [-16.542978625956586, -11.128500203052441] |
| reference_hull | fine16.control | -14.549182 | [-17.028461431317595, -11.536344306529815] |
| reference_hull | fine64.control | -14.765840 | [-17.04616484352579, -11.411497568724492] |
| constellation_local | native | +0.973304 | [0.31771734919645367, 1.660649824593348] |
| constellation_local | mean.control | -0.622928 | [-1.0057404554426834, -0.2904401152336891] |
| constellation_local | clipped_mean.control | -0.622774 | [-1.005232648845459, -0.2889439554352544] |
| constellation_local | pro_relations_zero.control | +1.534986 | [0.9533982744485021, 2.359136949182466] |
| constellation_local | pro_relations_positive.control | +1.457062 | [0.8607871078362652, 2.2426967627851444] |
| constellation_local | pro_relations_absolute.control | +1.550718 | [0.9724457449649512, 2.3658014887502086] |
| constellation_local | pro_relations_pair_independent.control | +1.707801 | [1.1162319667494605, 2.507849749692294] |
| constellation_local | pro_relations_block.control | +1.630947 | [1.0319888678858218, 2.4154493935942045] |
| constellation_local | hull_nearest.control | +19.285549 | [16.76009498071767, 21.537485495958304] |
| constellation_local | hull_centroid.control | +32.377492 | [29.921259941999722, 35.02895065985609] |
| constellation_local | hull_subspace.control | +16.653342 | [13.935878427483008, 18.884985567837745] |
| constellation_local | hull_affine.control | +14.353445 | [11.209827797517717, 16.70231613168491] |
| constellation_local | constellation_local_global_v1.control | +0.596031 | [0.21936977658692366, 1.162669809446082] |
| constellation_local | constellation_local_bag.control | +23.122632 | [21.372847408463905, 25.84785753522412] |
| constellation_local | constellation_local_prior.control | +60.423049 | [57.94775229994449, 63.623125988892795] |
| constellation_local | triplet_zero.control | +1.534986 | [0.9533982744485021, 2.359136949182466] |
| constellation_local | triplet_absolute.control | +1.539845 | [0.9584346766670138, 2.370095287516682] |
| constellation_local | triplet_no_third.control | +1.534986 | [0.9533982744485021, 2.359136949182466] |
| constellation_local | triplet_pro_m2.control | +1.676156 | [1.0852024646178098, 2.45124198611441] |
| constellation_local | stored_mean.control | -0.622912 | [-1.0057217725175698, -0.2904182291799344] |
| constellation_local | rcg.control | -0.780324 | [-1.2501066175756905, -0.23037264045026254] |
| constellation_local | fine16.control | -1.155909 | [-1.7088826240083201, -0.645410051898023] |
| constellation_local | fine64.control | -1.372566 | [-2.222542289248511, 0.09242382894717346] |
| reference_triplet_relations | native | -0.556326 | [-0.8386047417106873, -0.4403979612387794] |
| reference_triplet_relations | mean.control | -2.152558 | [-2.9030404592849437, -1.6215150029760301] |
| reference_triplet_relations | clipped_mean.control | -2.152404 | [-2.9021850023440185, -1.6218494166942061] |
| reference_triplet_relations | pro_relations_zero.control | +0.005357 | [-0.0014744260387869802, 0.015886648873016707] |
| reference_triplet_relations | pro_relations_positive.control | -0.072568 | [-0.21856885190257422, 0.007486298552126105] |
| reference_triplet_relations | pro_relations_absolute.control | +0.021089 | [-0.1245108801785996, 0.18563699658868363] |
| reference_triplet_relations | pro_relations_pair_independent.control | +0.178171 | [0.049552165670275275, 0.29231275504297277] |
| reference_triplet_relations | pro_relations_block.control | +0.101317 | [-0.060254122935095286, 0.20255331676937458] |
| reference_triplet_relations | hull_nearest.control | +17.755919 | [15.185636422636296, 19.712427707412793] |
| reference_triplet_relations | hull_centroid.control | +30.847862 | [28.42724165706682, 33.254568055864254] |
| reference_triplet_relations | hull_subspace.control | +15.123712 | [12.408224375258836, 17.063618115832423] |
| reference_triplet_relations | hull_affine.control | +12.823816 | [9.628357114539366, 14.991145179185958] |
| reference_triplet_relations | constellation_local_global_v1.control | -0.933598 | [-1.8953051662196603, -0.07957565899994949] |
| reference_triplet_relations | constellation_local_bag.control | +21.593002 | [19.956647693539313, 23.922024664629998] |
| reference_triplet_relations | constellation_local_prior.control | +58.893419 | [56.403150370549646, 61.80108181307496] |
| reference_triplet_relations | triplet_zero.control | +0.005357 | [-0.0014744260387869802, 0.015886648873016707] |
| reference_triplet_relations | triplet_absolute.control | +0.010215 | [0.0018097743471431117, 0.03119210041283029] |
| reference_triplet_relations | triplet_no_third.control | +0.005357 | [-0.0014744260387869802, 0.015886648873016707] |
| reference_triplet_relations | triplet_pro_m2.control | +0.146526 | [-0.043018984400183945, 0.27777055950036034] |
| reference_triplet_relations | stored_mean.control | -2.152542 | [-2.9030098837820786, -1.6215311209718204] |
| reference_triplet_relations | rcg.control | -2.309953 | [-3.0032062521279017, -1.7198903478500263] |
| reference_triplet_relations | fine16.control | -2.685539 | [-3.4960040998067354, -2.118619901155412] |
| reference_triplet_relations | fine64.control | -2.902196 | [-4.0367312297904325, -1.4760781570033648] |
