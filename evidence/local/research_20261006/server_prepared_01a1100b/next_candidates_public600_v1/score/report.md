# Cached-feature candidate results

| Arm | Class-summed mIoU |
|---|---:|
| native | 61.335314 |
| mean.control | 62.931546 |
| clipped_mean.control | 62.931392 |
| reference_absorption | 37.821542 |
| absorption_nearest.control | 51.891297 |
| absorption_kernel.control | 49.783948 |
| absorption_one_hop.control | 25.642711 |
| absorption_one_step.control | 61.538043 |
| absorption_component.control | 18.161979 |
| absorption_full_harmonic.control | 37.779498 |
| reference_gaussian_density | 61.949185 |
| gaussian_density_polynomial.control | 62.873699 |
| gaussian_density_quadratic_ridge.control | 62.971322 |
| gaussian_density_uniform.control | 62.083492 |
| gaussian_density_nearest.control | 61.738769 |
| gaussian_density_centroid.control | 60.540400 |
| gaussian_density_standalone.control | 39.309061 |
| mean_rgb_potts | 62.903951 |
| mean_rgb_unary.control | 62.769442 |
| stored_mean.control | 62.931530 |
| rcg.control | 63.088942 |
| fine16.control | 63.464527 |
| fine64.control | 63.681184 |

Primary: reference_absorption
Exposure: reused public600 development data; no independent confirmation; absorption candidate, Gaussian density correction version, RGB Potts simple control

| Primary versus | Gain, pp | Paired 95% CI |
|---|---:|---|
| native | -23.513772 | [-25.707112954692395, -18.82187244951477] |
| mean.control | -25.110004 | [-27.246089185302562, -20.47940422793673] |
| clipped_mean.control | -25.109851 | [-27.246089663243385, -20.479316663859446] |
| absorption_nearest.control | -14.069755 | [-16.85616419618044, -10.28296177656383] |
| absorption_kernel.control | -11.962406 | [-14.867416933240396, -8.193336957544993] |
| absorption_one_hop.control | +12.178831 | [9.586745756403188, 15.026906646466562] |
| absorption_one_step.control | -23.716501 | [-25.55841537904127, -18.997158214485236] |
| absorption_component.control | +19.659563 | [16.43582543631572, 22.221595505210214] |
| absorption_full_harmonic.control | +0.042044 | [-1.954808949979315, 0.8669244090879072] |
| gaussian_density_polynomial.control | -25.052157 | [-27.197393983286382, -20.379893834822557] |
| gaussian_density_quadratic_ridge.control | -25.149780 | [-27.385229663644445, -20.514744090639745] |
| gaussian_density_uniform.control | -24.261950 | [-26.672485382983236, -19.523977920013422] |
| gaussian_density_nearest.control | -23.917227 | [-26.400116343863434, -19.14755844429822] |
| gaussian_density_centroid.control | -22.718859 | [-25.13853826228439, -17.89779069889787] |
| gaussian_density_standalone.control | -1.487519 | [-4.313320709527872, 3.408742910634417] |
| mean_rgb_unary.control | -24.947900 | [-27.049807296438004, -20.284795758418554] |
| stored_mean.control | -25.109988 | [-27.246076869484263, -20.47939131244378] |
| rcg.control | -25.267400 | [-27.358001567087115, -20.56107923335548] |
| fine16.control | -25.642985 | [-27.801562661998343, -20.962889003927785] |
| fine64.control | -25.859643 | [-27.820350758627008, -20.930901903986467] |

| Predeclared candidate | Comparison | Gain, pp | Paired 95% CI |
|---|---|---:|---|
| reference_absorption | native | -23.513772 | [-25.707112954692395, -18.82187244951477] |
| reference_absorption | mean.control | -25.110004 | [-27.246089185302562, -20.47940422793673] |
| reference_absorption | clipped_mean.control | -25.109851 | [-27.246089663243385, -20.479316663859446] |
| reference_absorption | absorption_nearest.control | -14.069755 | [-16.85616419618044, -10.28296177656383] |
| reference_absorption | absorption_kernel.control | -11.962406 | [-14.867416933240396, -8.193336957544993] |
| reference_absorption | absorption_one_hop.control | +12.178831 | [9.586745756403188, 15.026906646466562] |
| reference_absorption | absorption_one_step.control | -23.716501 | [-25.55841537904127, -18.997158214485236] |
| reference_absorption | absorption_component.control | +19.659563 | [16.43582543631572, 22.221595505210214] |
| reference_absorption | absorption_full_harmonic.control | +0.042044 | [-1.954808949979315, 0.8669244090879072] |
| reference_absorption | gaussian_density_polynomial.control | -25.052157 | [-27.197393983286382, -20.379893834822557] |
| reference_absorption | gaussian_density_quadratic_ridge.control | -25.149780 | [-27.385229663644445, -20.514744090639745] |
| reference_absorption | gaussian_density_uniform.control | -24.261950 | [-26.672485382983236, -19.523977920013422] |
| reference_absorption | gaussian_density_nearest.control | -23.917227 | [-26.400116343863434, -19.14755844429822] |
| reference_absorption | gaussian_density_centroid.control | -22.718859 | [-25.13853826228439, -17.89779069889787] |
| reference_absorption | gaussian_density_standalone.control | -1.487519 | [-4.313320709527872, 3.408742910634417] |
| reference_absorption | mean_rgb_unary.control | -24.947900 | [-27.049807296438004, -20.284795758418554] |
| reference_absorption | stored_mean.control | -25.109988 | [-27.246076869484263, -20.47939131244378] |
| reference_absorption | rcg.control | -25.267400 | [-27.358001567087115, -20.56107923335548] |
| reference_absorption | fine16.control | -25.642985 | [-27.801562661998343, -20.962889003927785] |
| reference_absorption | fine64.control | -25.859643 | [-27.820350758627008, -20.930901903986467] |
| reference_gaussian_density | native | +0.613870 | [-0.04423959342277701, 1.4740098753307174] |
| reference_gaussian_density | mean.control | -0.982361 | [-1.5942177484967506, -0.2023636906198157] |
| reference_gaussian_density | clipped_mean.control | -0.982208 | [-1.5931494686818803, -0.2017100110680627] |
| reference_gaussian_density | absorption_nearest.control | +10.057888 | [7.51297729598208, 11.361965236935951] |
| reference_gaussian_density | absorption_kernel.control | +12.165237 | [9.286440663368323, 13.513871355298585] |
| reference_gaussian_density | absorption_one_hop.control | +36.306474 | [32.000608163399185, 38.26697330310838] |
| reference_gaussian_density | absorption_one_step.control | +0.411142 | [-0.3586438307827235, 1.8193425875041669] |
| reference_gaussian_density | absorption_component.control | +43.787205 | [38.77755598008975, 45.67101485184568] |
| reference_gaussian_density | absorption_full_harmonic.control | +24.169686 | [18.654125812057103, 25.870063866837835] |
| reference_gaussian_density | gaussian_density_polynomial.control | -0.924514 | [-1.450622023936174, -0.22903191379140664] |
| reference_gaussian_density | gaussian_density_quadratic_ridge.control | -1.022137 | [-1.612876205334058, -0.32795580694235266] |
| reference_gaussian_density | gaussian_density_uniform.control | -0.134307 | [-0.30886007204757515, 0.04143056753582318] |
| reference_gaussian_density | gaussian_density_nearest.control | +0.210416 | [-0.2603832390804124, 0.6581187234934786] |
| reference_gaussian_density | gaussian_density_centroid.control | +1.408784 | [0.971658334953003, 1.9857192347716184] |
| reference_gaussian_density | gaussian_density_standalone.control | +22.640124 | [20.243120407062605, 24.869301882258632] |
| reference_gaussian_density | mean_rgb_unary.control | -0.820257 | [-1.4154345014636387, -0.002107211865549002] |
| reference_gaussian_density | stored_mean.control | -0.982346 | [-1.594170243545872, -0.20239266618009344] |
| reference_gaussian_density | rcg.control | -1.139757 | [-1.835028292523967, -0.1980192374553537] |
| reference_gaussian_density | fine16.control | -1.515342 | [-2.307410140849402, -0.6207057596174927] |
| reference_gaussian_density | fine64.control | -1.732000 | [-2.797181095296567, -0.02816121636782912] |
| mean_rgb_potts | native | +1.568637 | [0.9496815140715553, 2.1599077788021908] |
| mean_rgb_potts | mean.control | -0.027595 | [-0.16043517305101815, 0.03241963506002781] |
| mean_rgb_potts | clipped_mean.control | -0.027441 | [-0.1604347570451125, 0.032683963557064794] |
| mean_rgb_potts | absorption_nearest.control | +11.012654 | [8.330654373000014, 12.309878032270953] |
| mean_rgb_potts | absorption_kernel.control | +13.120003 | [10.121769099972063, 14.374439187932625] |
| mean_rgb_potts | absorption_one_hop.control | +37.261240 | [32.89826358980054, 39.04983345292991] |
| mean_rgb_potts | absorption_one_step.control | +1.365908 | [0.8009459468317767, 2.4719662504530575] |
| mean_rgb_potts | absorption_component.control | +44.741972 | [39.73916256486882, 46.39040492139824] |
| mean_rgb_potts | absorption_full_harmonic.control | +25.124453 | [19.62366409866312, 26.608200702361007] |
| mean_rgb_potts | gaussian_density_polynomial.control | +0.030252 | [-0.16609109983387835, 0.19258672302705906] |
| mean_rgb_potts | gaussian_density_quadratic_ridge.control | -0.067371 | [-0.4104845317681036, 0.1803598041332631] |
| mean_rgb_potts | gaussian_density_uniform.control | +0.820459 | [0.06506868273102669, 1.3973270752747584] |
| mean_rgb_potts | gaussian_density_nearest.control | +1.165182 | [0.28602039212729713, 1.8563638659163417] |
| mean_rgb_potts | gaussian_density_centroid.control | +2.363551 | [1.2954007406013521, 3.297013174524016] |
| mean_rgb_potts | gaussian_density_standalone.control | +23.594890 | [20.960268276054865, 25.80575214668227] |
| mean_rgb_potts | mean_rgb_unary.control | +0.134509 | [0.05226807157858957, 0.21193656744729183] |
| mean_rgb_potts | stored_mean.control | -0.027579 | [-0.1604470297767385, 0.03242479942805117] |
| mean_rgb_potts | rcg.control | -0.184991 | [-0.5541959454864113, 0.23679057258643396] |
| mean_rgb_potts | fine16.control | -0.560576 | [-1.025518961255165, -0.1980459115286112] |
| mean_rgb_potts | fine64.control | -0.777233 | [-1.6054184475063236, 0.586464732122459] |
