# Existing cues on RCG remaining errors:4000 reused draws

GT diagnostics only; semantic connected regions are not instances. No encoder, fitting or threshold search.

| cohort | cue | eligible episodes | mean episode AUROC [95% CI] | fixed-rule balanced accuracy | pooled recall / FPR |
|---|---|---:|---|---:|---|
| whole_regions | signed_nn | 223 | 0.2532 [0.2157, 0.2935] | 0.3423 | 0.2330 / 0.7244 |
| whole_regions | foris_score | 223 | 0.0870 [0.0606, 0.1155] | 0.1818 | 0.1077 / 0.8734 |
| whole_regions | rcg_field | 223 | 0.0000 [0.0000, 0.0000] | 0.0013 | 0.0002 / 0.9974 |
| deep_errors | signed_nn | 1088 | 0.2325 [0.2156, 0.2498] | 0.3374 | 0.3260 / 0.7343 |
| deep_errors | foris_score | 1088 | 0.0955 [0.0842, 0.1073] | 0.1795 | 0.1118 / 0.8730 |
| deep_errors | rcg_field | 1088 | 0.0003 [0.0001, 0.0006] | 0.0095 | 0.0025 / 0.9972 |

RCG field is conditioned on errors of its rendered mask; poor field AUROC can be partly tautological and does not prove weak features

## Region availability

{
  "whole_missed_GT": {
    "regions": 1720,
    "pixels": 9361369,
    "regions_with_pure_tokens": 690,
    "regions_without_pure_tokens": 1030,
    "pixels_without_pure_tokens": 299488,
    "regions_with_positive_nn": 421,
    "eligible_regions_no_positive_nn": 269,
    "pure_tokens": 28063,
    "positive_nn_tokens": 6538,
    "small_regions_area_lt256": 679,
    "small_regions_without_pure_tokens": 679,
    "positive_nn_fraction_all_regions": {
      "ratio": 0.24476744186046512,
      "ci95": [
        0.21278712797315835,
        0.27974728432705886
      ]
    },
    "positive_nn_fraction_eligible_regions": {
      "ratio": 0.6101449275362318,
      "ci95": [
        0.5570170691881218,
        0.6592607885180033
      ]
    }
  },
  "stray_RCG": {
    "regions": 3894,
    "pixels": 33572981,
    "regions_with_pure_tokens": 3680,
    "regions_without_pure_tokens": 214,
    "pixels_without_pure_tokens": 4269,
    "regions_with_positive_nn": 2931,
    "eligible_regions_no_positive_nn": 749,
    "pure_tokens": 137055,
    "positive_nn_tokens": 99288,
    "small_regions_area_lt256": 1699,
    "small_regions_without_pure_tokens": 213,
    "positive_nn_fraction_all_regions": {
      "ratio": 0.7526964560862865,
      "ci95": [
        0.7260694675236844,
        0.7774770957027027
      ]
    },
    "positive_nn_fraction_eligible_regions": {
      "ratio": 0.7964673913043478,
      "ci95": [
        0.7695227715064227,
        0.8219478051540693
      ]
    }
  }
}

## Paired comparisons

{
  "whole_regions": {
    "signed_nn-minus-foris_score": {
      "mean": 0.16618846296050874,
      "ci95": [
        0.12802281558727827,
        0.20592532806286443
      ],
      "eligible_episodes": 223,
      "eligible_photo_groups": 209,
      "finite_bootstrap_draws": 2000
    },
    "signed_nn-minus-rcg_field": {
      "mean": 0.2531971644907306,
      "ci95": [
        0.21571188708745379,
        0.2934655232916272
      ],
      "eligible_episodes": 223,
      "eligible_photo_groups": 209,
      "finite_bootstrap_draws": 2000
    }
  },
  "deep_errors": {
    "signed_nn-minus-foris_score": {
      "mean": 0.13705290474052176,
      "ci95": [
        0.1202658206865056,
        0.15446508573094705
      ],
      "eligible_episodes": 1088,
      "eligible_photo_groups": 947,
      "finite_bootstrap_draws": 2000
    },
    "signed_nn-minus-rcg_field": {
      "mean": 0.23222128386830868,
      "ci95": [
        0.21522786336623992,
        0.24948213526210317
      ],
      "eligible_episodes": 1088,
      "eligible_photo_groups": 947,
      "finite_bootstrap_draws": 2000
    }
  }
}

Positive NN in a GT-selected region does not supply a deployable detector or complete-mask recovery result.
