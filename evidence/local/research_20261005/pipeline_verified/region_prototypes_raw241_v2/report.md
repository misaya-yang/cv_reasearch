# Region-mean semantic reference diagnostic: full exposed DEV241

GT geometry is privileged. No legal proposal generator, complete recovery method, encoder call, fitting or sign inversion.

| Fixed cue | Eligible episodes | Episode region AUROC [95% CI] | Region-pooled AUROC | Pixel-mass-weighted AUROC |
|---|---:|---:|---:|---:|
| raw_mean_prototype | 18 | 0.3944 [0.1680, 0.6072] | 0.4477 | 0.3434 |
| projected_mean_prototype | 18 | 0.2556 [0.0733, 0.4556] | 0.4312 | 0.3606 |
| stored_nn_mean | 18 | 0.3272 [0.1176, 0.5556] | 0.3989 | 0.2090 |
| fine64_field_mean | 18 | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 |

## Paired contrasts

{
  "raw_mean_prototype-minus-projected_mean_prototype": {
    "mean": 0.1388888888888889,
    "ci95": [
      0.027777777777777776,
      0.28125
    ],
    "eligible_episodes": 18,
    "eligible_photo_groups": 18,
    "finite_bootstrap_draws": 2000
  },
  "raw_mean_prototype-minus-stored_nn_mean": {
    "mean": 0.06728395061728396,
    "ci95": [
      -0.06262626262626263,
      0.225017361111111
    ],
    "eligible_episodes": 18,
    "eligible_photo_groups": 18,
    "finite_bootstrap_draws": 2000
  },
  "projected_mean_prototype-minus-stored_nn_mean": {
    "mean": -0.07160493827160494,
    "ci95": [
      -0.15449206349206349,
      0.0
    ],
    "eligible_episodes": 18,
    "eligible_photo_groups": 18,
    "finite_bootstrap_draws": 2000
  },
  "raw_mean_prototype-minus-fine64_field_mean": {
    "mean": 0.39444444444444443,
    "ci95": [
      0.16796666666666668,
      0.607227443609022
    ],
    "eligible_episodes": 18,
    "eligible_photo_groups": 18,
    "finite_bootstrap_draws": 2000
  }
}

## Region availability and pixel mass

{
  "whole_missed_GT": {
    "all_regions": 170,
    "eligible_regions": 114,
    "regions_without_pure_tokens": 56,
    "total_pixels": 1333939,
    "eligible_pixels": 1305139,
    "pixels_without_pure_tokens": 28800,
    "pure_tokens": 3828,
    "episodes_with_regions": 74,
    "episodes_with_eligible_regions": 59,
    "small_regions_area_lt256": 29,
    "small_regions_without_pure_tokens": 29,
    "no_pure_token_area_percentiles": [
      2.0,
      240.0,
      1421.0,
      2573.0
    ]
  },
  "stray_fine64": {
    "all_regions": 111,
    "eligible_regions": 83,
    "regions_without_pure_tokens": 28,
    "total_pixels": 1527467,
    "eligible_pixels": 1526749,
    "pixels_without_pure_tokens": 718,
    "pure_tokens": 5961,
    "episodes_with_regions": 51,
    "episodes_with_eligible_regions": 44,
    "small_regions_area_lt256": 33,
    "small_regions_without_pure_tokens": 28,
    "no_pure_token_area_percentiles": [
      1.0,
      8.5,
      76.70000000000003,
      155.0
    ]
  }
}

## Reference purity and fallback

{
  "fallback_episodes": 0,
  "fallback_max_coverage": [],
  "pure_FG_tokens_percentiles": [
    5.0,
    53.0,
    171.0,
    647.0,
    3947.0
  ],
  "exact_BG_tokens_percentiles": [
    53.0,
    3326.0,
    3841.0,
    3990.0,
    4057.0
  ],
  "selected_FG_min_coverage_percentiles": [
    0.90234375,
    0.90625,
    0.90625,
    0.921875,
    1.0
  ],
  "selected_FG_mean_coverage_percentiles": [
    0.965624988079071,
    0.994140625,
    0.9964817762374878,
    0.998335599899292,
    1.0
  ]
}

## Four folds

{
  "0": {
    "n": 61,
    "eligible_episodes": 4,
    "episode_auc": {
      "raw_mean_prototype": 0.4,
      "projected_mean_prototype": 0.4,
      "stored_nn_mean": 0.5,
      "fine64_field_mean": 0.0
    },
    "raw_minus_projected": 0.0,
    "raw_minus_stored_nn": -0.1
  },
  "1": {
    "n": 60,
    "eligible_episodes": 5,
    "episode_auc": {
      "raw_mean_prototype": 0.4,
      "projected_mean_prototype": 0.4,
      "stored_nn_mean": 0.4,
      "fine64_field_mean": 0.0
    },
    "raw_minus_projected": 0.0,
    "raw_minus_stored_nn": 0.0
  },
  "2": {
    "n": 60,
    "eligible_episodes": 4,
    "episode_auc": {
      "raw_mean_prototype": 0.375,
      "projected_mean_prototype": 0.0,
      "stored_nn_mean": 0.09722222222222222,
      "fine64_field_mean": 0.0
    },
    "raw_minus_projected": 0.375,
    "raw_minus_stored_nn": 0.2777777777777778
  },
  "3": {
    "n": 60,
    "eligible_episodes": 5,
    "episode_auc": {
      "raw_mean_prototype": 0.4,
      "projected_mean_prototype": 0.2,
      "stored_nn_mean": 0.3,
      "fine64_field_mean": 0.0
    },
    "raw_minus_projected": 0.2,
    "raw_minus_stored_nn": 0.1
  }
}

## Actual conditional projection subsets

{
  "projection_applied": {
    "n": 239,
    "episode_auc": {
      "raw_mean_prototype": {
        "mean": 0.39444444444444443,
        "ci95": [
          0.16796666666666668,
          0.607227443609022
        ],
        "eligible_episodes": 18,
        "eligible_photo_groups": 18,
        "finite_bootstrap_draws": 2000
      },
      "projected_mean_prototype": {
        "mean": 0.25555555555555554,
        "ci95": [
          0.07328571428571429,
          0.4555555555555555
        ],
        "eligible_episodes": 18,
        "eligible_photo_groups": 18,
        "finite_bootstrap_draws": 2000
      },
      "stored_nn_mean": {
        "mean": 0.3271604938271605,
        "ci95": [
          0.11764705882352941,
          0.5556186868686865
        ],
        "eligible_episodes": 18,
        "eligible_photo_groups": 18,
        "finite_bootstrap_draws": 2000
      },
      "fine64_field_mean": {
        "mean": 0.0,
        "ci95": [
          0.0,
          0.0
        ],
        "eligible_episodes": 18,
        "eligible_photo_groups": 18,
        "finite_bootstrap_draws": 2000
      }
    },
    "raw_minus_projected": {
      "mean": 0.1388888888888889,
      "ci95": [
        0.027777777777777776,
        0.28125
      ],
      "eligible_episodes": 18,
      "eligible_photo_groups": 18,
      "finite_bootstrap_draws": 2000
    },
    "raw_minus_stored_nn": {
      "mean": 0.06728395061728396,
      "ci95": [
        -0.06262626262626263,
        0.225017361111111
      ],
      "eligible_episodes": 18,
      "eligible_photo_groups": 18,
      "finite_bootstrap_draws": 2000
    }
  },
  "projection_not_applied": {
    "n": 2,
    "episode_auc": {
      "raw_mean_prototype": {
        "mean": null,
        "ci95": null,
        "eligible_episodes": 0,
        "eligible_photo_groups": 0,
        "finite_bootstrap_draws": 0
      },
      "projected_mean_prototype": {
        "mean": null,
        "ci95": null,
        "eligible_episodes": 0,
        "eligible_photo_groups": 0,
        "finite_bootstrap_draws": 0
      },
      "stored_nn_mean": {
        "mean": null,
        "ci95": null,
        "eligible_episodes": 0,
        "eligible_photo_groups": 0,
        "finite_bootstrap_draws": 0
      },
      "fine64_field_mean": {
        "mean": null,
        "ci95": null,
        "eligible_episodes": 0,
        "eligible_photo_groups": 0,
        "finite_bootstrap_draws": 0
      }
    },
    "raw_minus_projected": {
      "mean": null,
      "ci95": null,
      "eligible_episodes": 0,
      "eligible_photo_groups": 0,
      "finite_bootstrap_draws": 0
    },
    "raw_minus_stored_nn": {
      "mean": null,
      "ci95": null,
      "eligible_episodes": 0,
      "eligible_photo_groups": 0,
      "finite_bootstrap_draws": 0
    }
  }
}

fine64 field scores are conditioned on errors of its own rendered mask; poor separation can be tautological

raw FP32 versus cached FP16 includes quantization difference; only cache-debiased=True episodes applied projection, reported separately

positive cue does not supply legal object proposals or complete-mask gains; negative result limits this fixed mean-prototype construction
