# Fixed object-crop CLS diagnostic: exposed DEV241

query GT defines diagnostic region boxes before inference; this is not a deployable proposal generator or complete segmentation method

All197 fixed region descriptors retained; primary comparison is18 paired episodes/71 regions. No fitted weights, sign reversal or sweep.

| cue | paired episodes | episode region AUROC [95% CI] |
|---|---:|---:|
| object_cls | 18 | 0.5920 [0.3889,0.7878] |
| raw_mean_prototype | 18 | 0.3944 [0.1680,0.6072] |
| projected_mean_prototype | 18 | 0.2556 [0.0733,0.4556] |

Paired contrasts:

{
  "object_cls-minus-raw_mean_prototype": {
    "mean": 0.19753086419753085,
    "ci95": [
      -0.07142857142857142,
      0.4703703703703704
    ],
    "eligible_episodes": 18,
    "finite_bootstrap_draws": 2000
  },
  "object_cls-minus-projected_mean_prototype": {
    "mean": 0.33641975308641975,
    "ci95": [
      0.12497957516339869,
      0.5625
    ],
    "eligible_episodes": 18,
    "finite_bootstrap_draws": 2000
  }
}

Folds:

{
  "0": {
    "eligible_episodes": 4,
    "episode_auc": {
      "object_cls": 0.65,
      "raw_mean_prototype": 0.4,
      "projected_mean_prototype": 0.4
    },
    "CLS_minus_raw": 0.25
  },
  "1": {
    "eligible_episodes": 5,
    "episode_auc": {
      "object_cls": 0.6,
      "raw_mean_prototype": 0.4,
      "projected_mean_prototype": 0.4
    },
    "CLS_minus_raw": 0.2
  },
  "2": {
    "eligible_episodes": 4,
    "episode_auc": {
      "object_cls": 0.5138888888888888,
      "raw_mean_prototype": 0.375,
      "projected_mean_prototype": 0.0
    },
    "CLS_minus_raw": 0.1388888888888889
  },
  "3": {
    "eligible_episodes": 5,
    "episode_auc": {
      "object_cls": 0.6,
      "raw_mean_prototype": 0.4,
      "projected_mean_prototype": 0.2
    },
    "CLS_minus_raw": 0.2
  }
}

Supplement only (all197 regions across85 episodes):

{
  "object_cls": {
    "region_pooled_auc": 0.4490593954766434,
    "pixel_mass_weighted_auc": 0.6944081765577361,
    "fixed_zero_margin_TP": 77,
    "fixed_zero_margin_FP": 59,
    "positive_regions": 114,
    "negative_regions": 83
  },
  "raw_mean_prototype": {
    "region_pooled_auc": 0.4476854787571338,
    "pixel_mass_weighted_auc": 0.34342385251212887,
    "fixed_zero_margin_TP": 80,
    "fixed_zero_margin_FP": 62,
    "positive_regions": 114,
    "negative_regions": 83
  },
  "projected_mean_prototype": {
    "region_pooled_auc": 0.43119847812301837,
    "pixel_mass_weighted_auc": 0.3605534006963067,
    "fixed_zero_margin_TP": 90,
    "fixed_zero_margin_FP": 73,
    "positive_regions": 114,
    "negative_regions": 83
  }
}

only justifies seeking a legal proposal producer; does not establish SOTA or complete-mask improvement

stop this fixed crop/global-CLS construction; not all global representations
