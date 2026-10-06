# Cached-feature candidate results

| Arm | Class-summed mIoU |
|---|---:|
| native | 61.335314 |
| mean.control | 62.931546 |
| clipped_mean.control | 62.931392 |
| adjacency | 18.877422 |
| adjacency_bilinear.control | 18.669314 |
| mean_nearest.control | 61.771241 |
| huber | 62.527423 |
| boxed_quadratic.control | 62.931796 |
| pregraph.control | 61.208340 |
| original_quadratic.control | 62.931546 |
| color_bottleneck | 62.834849 |
| color_same_resize.control | 62.973914 |
| color_clipped_resize.control | 62.973914 |
| color_only.control | 58.960390 |
| constellation | 61.712587 |
| constellation_bag.control | 39.185986 |
| constellation_prior.control | 1.885569 |
| reference_shape | 57.131038 |
| shape_generic_square.control | 43.017743 |
| shape_bilinear.control | 57.490940 |
| reference_covariance | 12.343239 |
| covariance_trace.control | 45.996041 |
| covariance_bilinear.control | 12.352802 |
| covariance_mode_margin.control | 9.908336 |
| query_recurrence | 62.893978 |
| recurrence_all_seed.control | 62.045578 |
| recurrence_single_seed.control | 62.144735 |
| reference_prior_shift | 37.883317 |
| prior_balanced.control | 47.694545 |
| prior_reference_fraction.control | 37.116835 |
| prior_margin.control | 43.082269 |
| reference_quadratic | 51.607030 |
| quadratic_linear.control | 49.133401 |
| quadratic_homogeneous.control | 49.652330 |
| quadratic_kernel_mean.control | 34.766526 |
| quadratic_nearest.control | 43.023069 |
| quadratic_subspace.control | 45.655002 |
| stored_mean.control | 62.931530 |
| rcg.control | 63.088942 |
| fine16.control | 63.464527 |
| fine64.control | 63.681184 |

Primary: reference_covariance
Exposure: reused public600 development data; no independent confirmation

| Primary versus | Gain, pp | Paired 95% CI |
|---|---:|---|
| native | -48.992076 | [-52.804684000515344, -46.10913486788468] |
| mean.control | -50.588307 | [-54.478679594346154, -47.68155632761105] |
| clipped_mean.control | -50.588154 | [-54.47890623239734, -47.681266907415015] |
| adjacency_bilinear.control | -6.326076 | [-12.143439563346337, -5.0964947312594155] |
| mean_nearest.control | -49.428003 | [-53.23578881918317, -46.4522665620709] |
| boxed_quadratic.control | -50.588557 | [-54.47915604807211, -47.68118234499545] |
| pregraph.control | -48.865101 | [-52.60400501480082, -46.04578201113236] |
| original_quadratic.control | -50.588307 | [-54.478679594346154, -47.68155632761105] |
| color_same_resize.control | -50.630676 | [-54.51950095232618, -47.717673293569426] |
| color_clipped_resize.control | -50.630676 | [-54.51950095232618, -47.717673293569426] |
| color_only.control | -46.617152 | [-50.269669698278584, -43.71222828729468] |
| constellation_bag.control | -26.842748 | [-30.15981882824218, -23.36435555688305] |
| constellation_prior.control | +10.457669 | [7.88051973536413, 12.816029619495922] |
| shape_generic_square.control | -30.674504 | [-34.153222500721256, -27.59867678069229] |
| shape_bilinear.control | -45.147702 | [-48.76674785996369, -42.362385433833424] |
| covariance_trace.control | -33.652802 | [-36.34123875658804, -30.275251509843947] |
| covariance_bilinear.control | -0.009563 | [-0.03523515049780164, 0.021767470808171558] |
| covariance_mode_margin.control | +2.434903 | [-0.4765993314565613, 4.735180822392083] |
| recurrence_all_seed.control | -49.702339 | [-53.613194761590606, -46.759101328109296] |
| recurrence_single_seed.control | -49.801497 | [-53.69513839730186, -46.835103225617324] |
| prior_balanced.control | -35.351306 | [-40.02242341086153, -33.248028569776345] |
| prior_reference_fraction.control | -24.773596 | [-28.792623634686052, -21.652417047032838] |
| prior_margin.control | -30.739030 | [-34.887405718496275, -28.10313777710023] |
| quadratic_linear.control | -36.790162 | [-41.18637600698056, -34.2125408747551] |
| quadratic_homogeneous.control | -37.309092 | [-41.635216554908126, -34.685823585882304] |
| quadratic_kernel_mean.control | -22.423287 | [-26.41495144927546, -19.563299365528266] |
| quadratic_nearest.control | -30.679831 | [-34.8623341311925, -27.971891904342904] |
| quadratic_subspace.control | -33.311764 | [-37.617129662236984, -30.766835473766307] |
| stored_mean.control | -50.588292 | [-54.478674896196054, -47.68154406502011] |
| rcg.control | -50.745703 | [-54.670017153062496, -47.76062471074618] |
| fine16.control | -51.121288 | [-55.11048523157314, -48.15679455632576] |
| fine64.control | -51.337946 | [-55.23036744866404, -48.14526278710828] |

| Predeclared candidate | Comparison | Gain, pp | Paired 95% CI |
|---|---|---:|---|
| adjacency | native | -42.457893 | [-43.686623879577766, -37.79655392445192] |
| adjacency | mean.control | -44.054124 | [-45.39863818841575, -39.164109781733664] |
| adjacency | clipped_mean.control | -44.053971 | [-45.39905221047975, -39.16366522362991] |
| adjacency | adjacency_bilinear.control | +0.208108 | [0.04925825883296797, 0.4056634100035207] |
| adjacency | mean_nearest.control | -42.893819 | [-44.04482219408323, -37.94029589576243] |
| adjacency | boxed_quadratic.control | -44.054374 | [-45.399597086640256, -39.163928450094765] |
| adjacency | pregraph.control | -42.330918 | [-43.47816395059729, -37.589806328630516] |
| adjacency | original_quadratic.control | -44.054124 | [-45.39863818841575, -39.164109781733664] |
| adjacency | color_same_resize.control | -44.096492 | [-45.464076136077246, -39.18397838233796] |
| adjacency | color_clipped_resize.control | -44.096492 | [-45.464076136077246, -39.18397838233796] |
| adjacency | color_only.control | -40.082969 | [-41.41061326565963, -34.84530644360233] |
| adjacency | constellation_bag.control | -20.308565 | [-20.309688988302742, -15.736761249164712] |
| adjacency | constellation_prior.control | +16.991852 | [16.452681547060223, 21.869497086461937] |
| adjacency | shape_generic_square.control | -24.140321 | [-25.309971624157345, -19.097133276592277] |
| adjacency | shape_bilinear.control | -38.613519 | [-39.760033398221985, -33.68071419810894] |
| adjacency | covariance_trace.control | -27.118619 | [-28.011503310071394, -21.348164429587598] |
| adjacency | covariance_bilinear.control | +6.524620 | [5.337499260480452, 12.392131404668858] |
| adjacency | covariance_mode_margin.control | +8.969086 | [8.570707903144092, 13.481241921314782] |
| adjacency | recurrence_all_seed.control | -43.168156 | [-44.372878306474384, -38.475789254863294] |
| adjacency | recurrence_single_seed.control | -43.267314 | [-44.460502857258525, -38.53205015679197] |
| adjacency | prior_balanced.control | -28.817123 | [-30.848755399943542, -24.527630029815313] |
| adjacency | prior_reference_fraction.control | -18.239413 | [-20.105319751051272, -12.588161102783555] |
| adjacency | prior_margin.control | -24.204847 | [-25.3282481526062, -20.089462839096058] |
| adjacency | quadratic_linear.control | -30.255979 | [-31.730180174715784, -25.94180423822882] |
| adjacency | quadratic_homogeneous.control | -30.774909 | [-32.24458464954651, -26.288267509142464] |
| adjacency | quadratic_kernel_mean.control | -15.889104 | [-16.583754501823634, -11.739528818589298] |
| adjacency | quadratic_nearest.control | -24.145648 | [-25.14815820744789, -19.874098412727843] |
| adjacency | quadratic_subspace.control | -26.777580 | [-28.010619206582565, -22.4049681550203] |
| adjacency | stored_mean.control | -44.054108 | [-45.39863727095721, -39.16409422399866] |
| adjacency | rcg.control | -44.211520 | [-45.435429482983906, -39.29160698817587] |
| adjacency | fine16.control | -44.587105 | [-45.9056855182926, -39.703794289449746] |
| adjacency | fine64.control | -44.803763 | [-46.07057199487363, -39.48096025484746] |
| huber | native | +1.192109 | [0.646686130127746, 1.7102824277971145] |
| huber | mean.control | -0.404123 | [-0.7267529536491883, -0.18900066024496856] |
| huber | clipped_mean.control | -0.403969 | [-0.7259049928356367, -0.18908379003773135] |
| huber | adjacency_bilinear.control | +43.858109 | [39.07238231788213, 45.12928460816774] |
| huber | mean_nearest.control | +0.756182 | [0.5821402492515435, 1.0557991653070884] |
| huber | boxed_quadratic.control | -0.404372 | [-0.7257365625722405, -0.18898960994639605] |
| huber | pregraph.control | +1.319084 | [0.8726077774819098, 1.8581145531369598] |
| huber | original_quadratic.control | -0.404123 | [-0.7267529536491883, -0.18900066024496856] |
| huber | color_same_resize.control | -0.446491 | [-0.7733579917615845, -0.20634324537750007] |
| huber | color_clipped_resize.control | -0.446491 | [-0.7733579917615845, -0.20634324537750007] |
| huber | color_only.control | +3.567033 | [1.9698486295364568, 5.326796934534433] |
| huber | constellation_bag.control | +23.341437 | [21.647778045758134, 25.934652477529866] |
| huber | constellation_prior.control | +60.641854 | [58.122869550899225, 63.741883823670165] |
| huber | shape_generic_square.control | +19.509680 | [17.40557090488688, 21.933924651635806] |
| huber | shape_bilinear.control | +5.036483 | [3.7387196345497444, 6.4782945010716375] |
| huber | covariance_trace.control | +16.531383 | [14.511012859364211, 19.92946566220625] |
| huber | covariance_bilinear.control | +50.174622 | [47.26499477571132, 54.05062316015982] |
| huber | covariance_mode_margin.control | +52.619088 | [50.22114273189884, 55.25980637815386] |
| huber | recurrence_all_seed.control | +0.481845 | [-0.19243589976208533, 1.133327451211749] |
| huber | recurrence_single_seed.control | +0.382688 | [-0.2315343520648577, 1.0480831019119008] |
| huber | prior_balanced.control | +14.832878 | [11.493423982934079, 16.785519690045284] |
| huber | prior_reference_fraction.control | +25.410589 | [22.304124817609278, 28.793913868995663] |
| huber | prior_margin.control | +19.445155 | [16.820003393021913, 21.53140264194042] |
| huber | quadratic_linear.control | +13.394023 | [10.590174225694247, 15.300234510203502] |
| huber | quadratic_homogeneous.control | +12.875093 | [10.023570996459314, 14.866924345907172] |
| huber | quadratic_kernel_mean.control | +27.760898 | [25.2610403553614, 30.096691727851137] |
| huber | quadratic_nearest.control | +19.504354 | [16.83570435851473, 21.653434702864047] |
| huber | quadratic_subspace.control | +16.872421 | [14.043903457447488, 18.997647115023344] |
| huber | stored_mean.control | -0.404107 | [-0.726744199727737, -0.18898093562948842] |
| huber | rcg.control | -0.561518 | [-0.9854383792855298, -0.14254750689898812] |
| huber | fine16.control | -0.937104 | [-1.4834346925940363, -0.554352905425227] |
| huber | fine64.control | -1.153761 | [-2.0139779340988353, 0.1525915803084044] |
| color_bottleneck | native | +1.499535 | [0.5560041608365548, 2.418663322166404] |
| color_bottleneck | mean.control | -0.096697 | [-0.8857789274485878, 0.6537318258835356] |
| color_bottleneck | clipped_mean.control | -0.096543 | [-0.8857711532787125, 0.6527392701185398] |
| color_bottleneck | adjacency_bilinear.control | +44.165535 | [39.385958458751176, 45.46353245791396] |
| color_bottleneck | mean_nearest.control | +1.063608 | [0.2981480036011453, 1.9571558397807403] |
| color_bottleneck | boxed_quadratic.control | -0.096946 | [-0.8866143515334418, 0.6517165573254278] |
| color_bottleneck | pregraph.control | +1.626510 | [0.7906372576573724, 2.5940212329003596] |
| color_bottleneck | original_quadratic.control | -0.096697 | [-0.8857789274485878, 0.6537318258835356] |
| color_bottleneck | color_same_resize.control | -0.139065 | [-0.9188062783496553, 0.6049316041060948] |
| color_bottleneck | color_clipped_resize.control | -0.139065 | [-0.9188062783496553, 0.6049316041060948] |
| color_bottleneck | color_only.control | +3.874459 | [2.6885601510052535, 5.1782852814404565] |
| color_bottleneck | constellation_bag.control | +23.648863 | [21.766783163636514, 26.318595523299795] |
| color_bottleneck | constellation_prior.control | +60.949280 | [58.558221183060425, 63.97968033271865] |
| color_bottleneck | shape_generic_square.control | +19.817106 | [17.821561545046936, 22.106353309150574] |
| color_bottleneck | shape_bilinear.control | +5.343909 | [4.053764906996505, 6.828584478179496] |
| color_bottleneck | covariance_trace.control | +16.838809 | [14.911187605527294, 20.234660007154794] |
| color_bottleneck | covariance_bilinear.control | +50.482048 | [47.656478596358845, 54.36329437955921] |
| color_bottleneck | covariance_mode_margin.control | +52.926514 | [50.485632437581174, 55.58882845680291] |
| color_bottleneck | recurrence_all_seed.control | +0.789271 | [-0.20698215764237346, 1.728368169033783] |
| color_bottleneck | recurrence_single_seed.control | +0.690114 | [-0.23087028431922202, 1.631538655718132] |
| color_bottleneck | prior_balanced.control | +15.140304 | [11.877794815731352, 17.093030965406356] |
| color_bottleneck | prior_reference_fraction.control | +25.718015 | [22.713568081122542, 28.985741632376985] |
| color_bottleneck | prior_margin.control | +19.752581 | [17.109728470305168, 21.894393293827246] |
| color_bottleneck | quadratic_linear.control | +13.701449 | [10.880850371937486, 15.702248767742487] |
| color_bottleneck | quadratic_homogeneous.control | +13.182519 | [10.30807919647652, 15.185310544896181] |
| color_bottleneck | quadratic_kernel_mean.control | +28.068324 | [25.474338922777353, 30.49271160822884] |
| color_bottleneck | quadratic_nearest.control | +19.811780 | [17.217335258953053, 21.988440343052723] |
| color_bottleneck | quadratic_subspace.control | +17.179847 | [14.4442850758813, 19.311087810553683] |
| color_bottleneck | stored_mean.control | -0.096681 | [-0.8857709104347515, 0.6537446805568184] |
| color_bottleneck | rcg.control | -0.254092 | [-1.0324393762466941, 0.5688075731011787] |
| color_bottleneck | fine16.control | -0.629678 | [-1.466213986384788, 0.1300814440719953] |
| color_bottleneck | fine64.control | -0.846335 | [-1.6815712587636775, 0.6184279017963829] |
| constellation | native | +0.377272 | [-0.5412571466772211, 1.2361795348391593] |
| constellation | mean.control | -1.218959 | [-1.9938418707269863, -0.6099561741344645] |
| constellation | clipped_mean.control | -1.218806 | [-1.9933468996000143, -0.6076841299028654] |
| constellation | adjacency_bilinear.control | +43.043273 | [38.228760554393595, 44.387855281191705] |
| constellation | mean_nearest.control | -0.058654 | [-0.7336006140395135, 0.6674268658739919] |
| constellation | boxed_quadratic.control | -1.219209 | [-1.9935554330657592, -0.6078590945207317] |
| constellation | pregraph.control | +0.504247 | [-0.29939239572480825, 1.3815155983145238] |
| constellation | original_quadratic.control | -1.218959 | [-1.9938418707269863, -0.6099561741344645] |
| constellation | color_same_resize.control | -1.261328 | [-2.022231090921004, -0.6488983518656891] |
| constellation | color_clipped_resize.control | -1.261328 | [-2.022231090921004, -0.6488983518656891] |
| constellation | color_only.control | +2.752196 | [1.0220700583656073, 4.600095934027451] |
| constellation | constellation_bag.control | +22.526600 | [20.723032524808442, 25.22823767238203] |
| constellation | constellation_prior.control | +59.827017 | [57.24736186485049, 63.12340707498682] |
| constellation | shape_generic_square.control | +18.694844 | [16.59568614477772, 21.181202515923573] |
| constellation | shape_bilinear.control | +4.221646 | [2.8042988700326825, 5.923285519972857] |
| constellation | covariance_trace.control | +15.716546 | [13.535895260426928, 19.21002487126001] |
| constellation | covariance_bilinear.control | +49.359785 | [46.35476622463769, 53.46619473763198] |
| constellation | covariance_mode_margin.control | +51.804251 | [49.363446492788356, 54.574544999735565] |
| constellation | recurrence_all_seed.control | -0.332991 | [-1.3119109857725793, 0.5474362902304317] |
| constellation | recurrence_single_seed.control | -0.432149 | [-1.345527676318318, 0.472436187704615] |
| constellation | prior_balanced.control | +14.018042 | [10.620986502405598, 16.033986718426736] |
| constellation | prior_reference_fraction.control | +24.595752 | [21.43698684677655, 28.086753673911414] |
| constellation | prior_margin.control | +18.630318 | [15.903854999347809, 20.85186412178307] |
| constellation | quadratic_linear.control | +12.579186 | [9.73323488918441, 14.655920738498796] |
| constellation | quadratic_homogeneous.control | +12.060256 | [9.193050067144657, 14.203658660655462] |
| constellation | quadratic_kernel_mean.control | +26.946061 | [24.355637022080696, 29.438597596457228] |
| constellation | quadratic_nearest.control | +18.689517 | [16.079293698826273, 20.991604060037233] |
| constellation | quadratic_subspace.control | +16.057584 | [13.27342788418866, 18.312412613310297] |
| constellation | stored_mean.control | -1.218943 | [-1.9937954557189999, -0.6099404022138076] |
| constellation | rcg.control | -1.376355 | [-2.190253043356088, -0.6482109527515212] |
| constellation | fine16.control | -1.751940 | [-2.647698071043977, -1.0729914153204616] |
| constellation | fine64.control | -1.968598 | [-3.0287987971249035, -0.42519922824038825] |
| reference_shape | native | -4.204276 | [-5.777799026374497, -2.856421720629697] |
| reference_shape | mean.control | -5.800508 | [-7.334941950106158, -4.548744805672099] |
| reference_shape | clipped_mean.control | -5.800354 | [-7.335152808406186, -4.5482758519251005] |
| reference_shape | adjacency_bilinear.control | +38.461724 | [33.557666029869395, 39.62369946740776] |
| reference_shape | mean_nearest.control | -4.640203 | [-6.025629549612014, -3.3087316634277757] |
| reference_shape | boxed_quadratic.control | -5.800758 | [-7.336064153353528, -4.549393321637328] |
| reference_shape | pregraph.control | -4.077302 | [-5.493635136162978, -2.7619094411710647] |
| reference_shape | original_quadratic.control | -5.800508 | [-7.334941950106158, -4.548744805672099] |
| reference_shape | color_same_resize.control | -5.842876 | [-7.378717036059013, -4.58683187203614] |
| reference_shape | color_clipped_resize.control | -5.842876 | [-7.378717036059013, -4.58683187203614] |
| reference_shape | color_only.control | -1.829352 | [-3.6614249727547357, 0.07135582663533492] |
| reference_shape | constellation_bag.control | +17.945052 | [15.943817370286512, 20.629927555229397] |
| reference_shape | constellation_prior.control | +55.245469 | [52.81382670206261, 58.090009464864025] |
| reference_shape | shape_generic_square.control | +14.113295 | [11.771416494194414, 16.550514749885654] |
| reference_shape | shape_bilinear.control | -0.359903 | [-0.4261437418229244, -0.3431636350220494] |
| reference_shape | covariance_trace.control | +11.134997 | [9.046244037861843, 14.357172341270473] |
| reference_shape | covariance_bilinear.control | +44.778236 | [41.98095015542561, 48.340299513444] |
| reference_shape | covariance_mode_margin.control | +47.222702 | [44.79619159489888, 49.71455988184613] |
| reference_shape | recurrence_all_seed.control | -4.914540 | [-6.584847173031179, -3.5121111128499143] |
| reference_shape | recurrence_single_seed.control | -5.013697 | [-6.586204843511271, -3.60607162625759] |
| reference_shape | prior_balanced.control | +9.436493 | [5.8441198827749785, 11.48574452732098] |
| reference_shape | prior_reference_fraction.control | +20.014203 | [16.904021348961965, 23.216198775445744] |
| reference_shape | prior_margin.control | +14.048769 | [11.104161624381717, 16.27144574910053] |
| reference_shape | quadratic_linear.control | +7.997637 | [4.946978969807316, 10.052362556350664] |
| reference_shape | quadratic_homogeneous.control | +7.478707 | [4.379048168294252, 9.659665191106969] |
| reference_shape | quadratic_kernel_mean.control | +22.364512 | [19.602864226082165, 24.808384360342153] |
| reference_shape | quadratic_nearest.control | +14.107968 | [11.19246033667391, 16.460492714297146] |
| reference_shape | quadratic_subspace.control | +11.476036 | [8.417419949385367, 13.785254381096573] |
| reference_shape | stored_mean.control | -5.800492 | [-7.334902630756561, -4.548723369911889] |
| reference_shape | rcg.control | -5.957904 | [-7.477935370291029, -4.661099837496468] |
| reference_shape | fine16.control | -6.333489 | [-7.966855480774665, -5.068896934646208] |
| reference_shape | fine64.control | -6.550146 | [-7.909680477146766, -4.925251786914808] |
| reference_covariance | native | -48.992076 | [-52.804684000515344, -46.10913486788468] |
| reference_covariance | mean.control | -50.588307 | [-54.478679594346154, -47.68155632761105] |
| reference_covariance | clipped_mean.control | -50.588154 | [-54.47890623239734, -47.681266907415015] |
| reference_covariance | adjacency_bilinear.control | -6.326076 | [-12.143439563346337, -5.0964947312594155] |
| reference_covariance | mean_nearest.control | -49.428003 | [-53.23578881918317, -46.4522665620709] |
| reference_covariance | boxed_quadratic.control | -50.588557 | [-54.47915604807211, -47.68118234499545] |
| reference_covariance | pregraph.control | -48.865101 | [-52.60400501480082, -46.04578201113236] |
| reference_covariance | original_quadratic.control | -50.588307 | [-54.478679594346154, -47.68155632761105] |
| reference_covariance | color_same_resize.control | -50.630676 | [-54.51950095232618, -47.717673293569426] |
| reference_covariance | color_clipped_resize.control | -50.630676 | [-54.51950095232618, -47.717673293569426] |
| reference_covariance | color_only.control | -46.617152 | [-50.269669698278584, -43.71222828729468] |
| reference_covariance | constellation_bag.control | -26.842748 | [-30.15981882824218, -23.36435555688305] |
| reference_covariance | constellation_prior.control | +10.457669 | [7.88051973536413, 12.816029619495922] |
| reference_covariance | shape_generic_square.control | -30.674504 | [-34.153222500721256, -27.59867678069229] |
| reference_covariance | shape_bilinear.control | -45.147702 | [-48.76674785996369, -42.362385433833424] |
| reference_covariance | covariance_trace.control | -33.652802 | [-36.34123875658804, -30.275251509843947] |
| reference_covariance | covariance_bilinear.control | -0.009563 | [-0.03523515049780164, 0.021767470808171558] |
| reference_covariance | covariance_mode_margin.control | +2.434903 | [-0.4765993314565613, 4.735180822392083] |
| reference_covariance | recurrence_all_seed.control | -49.702339 | [-53.613194761590606, -46.759101328109296] |
| reference_covariance | recurrence_single_seed.control | -49.801497 | [-53.69513839730186, -46.835103225617324] |
| reference_covariance | prior_balanced.control | -35.351306 | [-40.02242341086153, -33.248028569776345] |
| reference_covariance | prior_reference_fraction.control | -24.773596 | [-28.792623634686052, -21.652417047032838] |
| reference_covariance | prior_margin.control | -30.739030 | [-34.887405718496275, -28.10313777710023] |
| reference_covariance | quadratic_linear.control | -36.790162 | [-41.18637600698056, -34.2125408747551] |
| reference_covariance | quadratic_homogeneous.control | -37.309092 | [-41.635216554908126, -34.685823585882304] |
| reference_covariance | quadratic_kernel_mean.control | -22.423287 | [-26.41495144927546, -19.563299365528266] |
| reference_covariance | quadratic_nearest.control | -30.679831 | [-34.8623341311925, -27.971891904342904] |
| reference_covariance | quadratic_subspace.control | -33.311764 | [-37.617129662236984, -30.766835473766307] |
| reference_covariance | stored_mean.control | -50.588292 | [-54.478674896196054, -47.68154406502011] |
| reference_covariance | rcg.control | -50.745703 | [-54.670017153062496, -47.76062471074618] |
| reference_covariance | fine16.control | -51.121288 | [-55.11048523157314, -48.15679455632576] |
| reference_covariance | fine64.control | -51.337946 | [-55.23036744866404, -48.14526278710828] |
| query_recurrence | native | +1.558663 | [0.9568308243758474, 2.255029861443602] |
| query_recurrence | mean.control | -0.037568 | [-0.3461870001940456, 0.3391216586336874] |
| query_recurrence | clipped_mean.control | -0.037415 | [-0.34655734841084396, 0.3383423375764027] |
| query_recurrence | adjacency_bilinear.control | +44.224664 | [39.48462341238346, 45.579361717807984] |
| query_recurrence | mean_nearest.control | +1.122737 | [0.8657834751775869, 1.62220149647512] |
| query_recurrence | boxed_quadratic.control | -0.037818 | [-0.34701012232854345, 0.33648093943372165] |
| query_recurrence | pregraph.control | +1.685638 | [1.1971397155351613, 2.371901272326117] |
| query_recurrence | original_quadratic.control | -0.037568 | [-0.3461870001940456, 0.3391216586336874] |
| query_recurrence | color_same_resize.control | -0.079936 | [-0.3945577748926006, 0.31583631021456443] |
| query_recurrence | color_clipped_resize.control | -0.079936 | [-0.3945577748926006, 0.31583631021456443] |
| query_recurrence | color_only.control | +3.933587 | [2.386488474006252, 5.689827485427303] |
| query_recurrence | constellation_bag.control | +23.707991 | [21.98399294110864, 26.400914442474406] |
| query_recurrence | constellation_prior.control | +61.008408 | [58.55139528417462, 64.18790336683475] |
| query_recurrence | shape_generic_square.control | +19.876235 | [17.780137800882137, 22.36772707130517] |
| query_recurrence | shape_bilinear.control | +5.403037 | [4.111837503160384, 6.930493462861274] |
| query_recurrence | covariance_trace.control | +16.897937 | [14.921149480646427, 20.340394873534482] |
| query_recurrence | covariance_bilinear.control | +50.541176 | [47.704904861413716, 54.48605533355945] |
| query_recurrence | covariance_mode_margin.control | +52.985642 | [50.58550551411073, 55.70837848130385] |
| query_recurrence | recurrence_all_seed.control | +0.848400 | [0.37279550265106065, 1.4186935443447375] |
| query_recurrence | recurrence_single_seed.control | +0.749243 | [0.3125662628325861, 1.3521493531453956] |
| query_recurrence | prior_balanced.control | +15.199433 | [11.88479451208, 17.259458974564637] |
| query_recurrence | prior_reference_fraction.control | +25.777143 | [22.695663601286682, 29.2172072383154] |
| query_recurrence | prior_margin.control | +19.811709 | [17.19476457517002, 21.976920710342956] |
| query_recurrence | quadratic_linear.control | +13.760577 | [10.979640190045094, 15.700219130074712] |
| query_recurrence | quadratic_homogeneous.control | +13.241647 | [10.45740882999234, 15.295518462500414] |
| query_recurrence | quadratic_kernel_mean.control | +28.127452 | [25.62150399564585, 30.501941580058475] |
| query_recurrence | quadratic_nearest.control | +19.870908 | [17.307148817611345, 22.071704244853308] |
| query_recurrence | quadratic_subspace.control | +17.238976 | [14.491943364763152, 19.425975649135243] |
| query_recurrence | stored_mean.control | -0.037552 | [-0.3461877131994001, 0.33916939645187866] |
| query_recurrence | rcg.control | -0.194964 | [-0.6321344453286402, 0.4023374471397037] |
| query_recurrence | fine16.control | -0.570549 | [-1.1113677704070608, -0.011609689768932696] |
| query_recurrence | fine64.control | -0.787207 | [-1.6283287212087676, 0.7233409899419777] |
| reference_prior_shift | native | -23.451997 | [-28.04727608597538, -21.563793598360867] |
| reference_prior_shift | mean.control | -25.048229 | [-29.68333901753377, -23.08153400765984] |
| reference_prior_shift | clipped_mean.control | -25.048075 | [-29.682391008170747, -23.07982646161326] |
| reference_prior_shift | adjacency_bilinear.control | +19.214003 | [12.80336995316717, 19.645420875228545] |
| reference_prior_shift | mean_nearest.control | -23.887924 | [-28.41735720317125, -21.852911607124746] |
| reference_prior_shift | boxed_quadratic.control | -25.048479 | [-29.682786273961664, -23.080257254002902] |
| reference_prior_shift | pregraph.control | -23.325023 | [-27.77876325448674, -21.431855501065193] |
| reference_prior_shift | original_quadratic.control | -25.048229 | [-29.68333901753377, -23.08153400765984] |
| reference_prior_shift | color_same_resize.control | -25.090597 | [-29.691060083822116, -23.12781685765018] |
| reference_prior_shift | color_clipped_resize.control | -25.090597 | [-29.691060083822116, -23.12781685765018] |
| reference_prior_shift | color_only.control | -21.077073 | [-25.652366090028934, -18.957644199339615] |
| reference_prior_shift | constellation_bag.control | -1.302669 | [-5.1860246996085095, 1.0249356264576326] |
| reference_prior_shift | constellation_prior.control | +35.997748 | [31.650415239481816, 38.324583774060606] |
| reference_prior_shift | shape_generic_square.control | -5.134426 | [-9.679373751968162, -3.0211451089041046] |
| reference_prior_shift | shape_bilinear.control | -19.607624 | [-24.185673988473482, -17.515993894490393] |
| reference_prior_shift | covariance_trace.control | -8.112724 | [-12.48166853355238, -4.950496529909723] |
| reference_prior_shift | covariance_bilinear.control | +25.530515 | [21.107234799402946, 28.54344057443087] |
| reference_prior_shift | covariance_mode_margin.control | +27.974981 | [23.611061576336112, 29.98107611116197] |
| reference_prior_shift | recurrence_all_seed.control | -24.162261 | [-28.87380148043188, -22.119697317999524] |
| reference_prior_shift | recurrence_single_seed.control | -24.261418 | [-28.90928941343213, -22.19188536661837] |
| reference_prior_shift | prior_balanced.control | -9.811228 | [-14.48890513557769, -9.056893764205334] |
| reference_prior_shift | prior_reference_fraction.control | +0.766482 | [-2.521629514309696, 1.8635909575567136] |
| reference_prior_shift | prior_margin.control | -5.198952 | [-9.97450530030844, -3.5765035772959357] |
| reference_prior_shift | quadratic_linear.control | -11.250084 | [-15.978685322926525, -10.03561486075027] |
| reference_prior_shift | quadratic_homogeneous.control | -11.769014 | [-16.439638986663788, -10.481799193092327] |
| reference_prior_shift | quadratic_kernel_mean.control | +3.116791 | [-1.565753723777983, 5.126275287194227] |
| reference_prior_shift | quadratic_nearest.control | -5.139752 | [-9.737058205408706, -3.4741654652753535] |
| reference_prior_shift | quadratic_subspace.control | -7.771685 | [-12.376109338845229, -6.325666707741229] |
| reference_prior_shift | stored_mean.control | -25.048213 | [-29.683325757763477, -23.08153392620077] |
| reference_prior_shift | rcg.control | -25.205625 | [-29.773984686848404, -23.205138881445812] |
| reference_prior_shift | fine16.control | -25.581210 | [-30.26703402377494, -23.60700385759004] |
| reference_prior_shift | fine64.control | -25.797867 | [-30.356107103217262, -23.665812847608017] |
| reference_quadratic | native | -9.728284 | [-11.776083239789955, -6.89716260006317] |
| reference_quadratic | mean.control | -11.324516 | [-13.476967410949852, -8.389441453113122] |
| reference_quadratic | clipped_mean.control | -11.324362 | [-13.476701610913706, -8.387146187743358] |
| reference_quadratic | adjacency_bilinear.control | +32.937716 | [28.440871280337007, 34.763324249598725] |
| reference_quadratic | mean_nearest.control | -10.164211 | [-12.25436734995538, -7.124639227377505] |
| reference_quadratic | boxed_quadratic.control | -11.324766 | [-13.476763344235176, -8.387369053271355] |
| reference_quadratic | pregraph.control | -9.601309 | [-11.557800200170666, -6.734028869819409] |
| reference_quadratic | original_quadratic.control | -11.324516 | [-13.476967410949852, -8.389441453113122] |
| reference_quadratic | color_same_resize.control | -11.366884 | [-13.509397197617657, -8.418908584794814] |
| reference_quadratic | color_clipped_resize.control | -11.366884 | [-13.509397197617657, -8.418908584794814] |
| reference_quadratic | color_only.control | -7.353360 | [-9.47031956024753, -4.2468287871561365] |
| reference_quadratic | constellation_bag.control | +12.421044 | [10.658102222393286, 15.967812096356532] |
| reference_quadratic | constellation_prior.control | +49.721461 | [47.33943394355638, 53.4046156334539] |
| reference_quadratic | shape_generic_square.control | +8.589287 | [6.325189019971981, 11.940998245510437] |
| reference_quadratic | shape_bilinear.control | -5.883910 | [-8.050763392365186, -2.7001999565771224] |
| reference_quadratic | covariance_trace.control | +5.610989 | [3.4473025673267648, 10.10575410867586] |
| reference_quadratic | covariance_bilinear.control | +39.254228 | [36.795417591293386, 43.67102545015977] |
| reference_quadratic | covariance_mode_margin.control | +41.698694 | [39.40188371360307, 44.969372374625834] |
| reference_quadratic | recurrence_all_seed.control | -10.438548 | [-12.663656592125827, -7.412402693276798] |
| reference_quadratic | recurrence_single_seed.control | -10.537705 | [-12.70312154981439, -7.500683441519847] |
| reference_quadratic | prior_balanced.control | +3.912485 | [2.2778568234697296, 5.177631061173795] |
| reference_quadratic | prior_reference_fraction.control | +14.490195 | [12.33285589601144, 17.970409832362822] |
| reference_quadratic | prior_margin.control | +8.524762 | [6.5333471608907905, 11.047458112865604] |
| reference_quadratic | quadratic_linear.control | +2.473630 | [1.1235366069450863, 3.9697247974847607] |
| reference_quadratic | quadratic_homogeneous.control | +1.954700 | [0.7897289162252396, 3.38455410539937] |
| reference_quadratic | quadratic_kernel_mean.control | +16.840504 | [14.333797775366211, 19.98105008687068] |
| reference_quadratic | quadratic_nearest.control | +8.583961 | [6.6808300499984465, 11.102275323808135] |
| reference_quadratic | quadratic_subspace.control | +5.952028 | [4.139503219587635, 8.08077562250889] |
| reference_quadratic | stored_mean.control | -11.324500 | [-13.476975669364254, -8.389455419869439] |
| reference_quadratic | rcg.control | -11.481912 | [-13.610895094724345, -8.455038924608957] |
| reference_quadratic | fine16.control | -11.857497 | [-14.092989956250737, -8.877163960680347] |
| reference_quadratic | fine64.control | -12.074154 | [-14.109992128204407, -8.720533324906425] |
