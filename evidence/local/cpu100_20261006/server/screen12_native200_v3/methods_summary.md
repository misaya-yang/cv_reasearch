# Native1024 CPU development screen: 12 methods, 200 cases

All 32 requested method/control arms completed on all 200 cases, with 74 observed classes. Native FP32 final-LN DINOv3 at 1024, no Part1; direct prototype 42.580632. Inference wall 551.16s. Methods were chosen posthoc on four cases, and this is a reused public development cohort; it is not independent confirmation.

| Method | Score | Δ prototype (95% interval) | Best declared control | Δ control (95% interval) | Mean seconds |
|---|---:|---|---|---|---:|
| inv_huber_reference_readout | 52.3734 | +9.7928 [+3.3868, +10.9957] | inv_huber_ridge | +0.7109 [+0.1353, +1.0014] | 0.606 |
| DR08 | 49.0343 | +6.4537 [+1.3337, +8.5010] | DR_control_average_logistic | -1.3124 [-4.1985, +1.2503] | 0.282 |
| DR03 | 48.9408 | +6.3602 [+1.4103, +8.3148] | DR_control_average_logistic | -1.4059 [-4.1245, +0.8737] | 0.307 |
| DR04 | 48.4795 | +5.8988 [+2.5583, +7.3365] | DR_control_average_logistic | -1.8672 [-2.5244, -0.4178] | 0.313 |
| cross_image_csls_hubness | 45.2626 | +2.6820 [-0.3956, +4.6278] | cross_image_cosine_dictionary_control | -0.3644 [-1.9189, +1.8824] | 0.841 |
| inv_adversarial_channel_support | 43.9545 | +1.3739 [-4.5677, +4.1440] | inv_adversarial_constant | -6.7634 [-9.1110, -4.2504] | 0.364 |
| inv_reference_mad_winsor | 42.8000 | +0.2193 [-0.5930, +0.7226] | dino_prototype.control | +0.2193 [-0.5930, +0.7226] | 1.451 |
| QP01 | 42.6975 | +0.1168 [-3.1102, +3.4088] | QP_center_prototype | -0.3444 [-1.9342, +3.5864] | 0.527 |
| QP02 | 42.4507 | -0.1299 [-3.2119, +3.4751] | QP_center_prototype | -0.5912 [-1.9394, +3.7173] | 0.564 |
| RGB02 | 38.0095 | -4.5711 [-6.7604, -2.8068] | dino_prototype.control | -4.5711 [-6.7604, -2.8068] | 0.432 |
| ref_ordinal_copula | 36.2029 | -6.3777 [-11.4528, -2.4672] | dino_prototype.control | -6.3777 [-11.4528, -2.4672] | 1.293 |
| local_001 | 12.1145 | -30.4661 [-34.0133, -27.6119] | dino_prototype.control | -30.4661 [-34.0133, -27.6119] | 1.266 |

## Global edit counts (descriptive; not a substitute for class-IoU accounting)

| Method | Add target | Add background | Delete target | Delete background |
|---|---:|---:|---:|---:|
| inv_adversarial_channel_support | 0 | 0 | 2260698 | 4952742 |
| inv_reference_mad_winsor | 5149 | 145943 | 129034 | 420373 |
| QP01 | 367765 | 3581735 | 533138 | 2915838 |
| QP02 | 451907 | 3347807 | 481950 | 2709926 |
| cross_image_csls_hubness | 238044 | 1479502 | 475948 | 2556437 |
| RGB02 | 221397 | 2144417 | 454775 | 1039609 |
| inv_huber_reference_readout | 25300 | 264781 | 1137142 | 3753092 |
| DR03 | 39528 | 180085 | 1117228 | 3717422 |
| DR04 | 5210 | 428732 | 617606 | 2230176 |
| DR08 | 97926 | 642470 | 1165096 | 3619936 |
| local_001 | 523466 | 12868426 | 2493406 | 2881310 |
| ref_ordinal_copula | 30143 | 1109403 | 2683477 | 4089263 |

Full per-class four-action counts and exact intersection/union gains are in methods_summary.json. All scores are class-summed mean IoU, not episode-average IoU or pooled pixel accuracy. Strongest-control comparisons are descriptive selection among fixed controls. No new method variant, parameter sweep or remote job was started from these scores.
