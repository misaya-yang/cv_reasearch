# Frozen object-crop CLS diagnostic: exposed1200

GT crop geometry is privileged; no legal proposal generator or complete segmentation gain is established.

| cue | paired episodes | episode AUROC [95% CI] |
|---|---:|---:|
| object_cls | 77 | 0.4340 [0.3401,0.5272] |
| stored_nn_mean | 77 | 0.2433 [0.1629,0.3329] |

Paired contrasts:

{
  "CLS_minus_NN": {
    "mean": 0.19067408781694498,
    "ci95": [
      0.09020250582750582,
      0.30078477075761806
    ],
    "eligible_episodes": 77
  },
  "CLS_minus_chance": {
    "mean": -0.06604102246959391,
    "ci95": [
      -0.15990058042736613,
      0.02720033453734453
    ],
    "eligible_episodes": 77
  }
}

Folds:

{
  "0": {
    "eligible_episodes": 19,
    "episode_auc": {
      "object_cls": 0.40313283208020045,
      "stored_nn_mean": 0.17293233082706766
    },
    "CLS_minus_NN": 0.23020050125313282
  },
  "1": {
    "eligible_episodes": 19,
    "episode_auc": {
      "object_cls": 0.5833333333333333,
      "stored_nn_mean": 0.19035087719298244
    },
    "CLS_minus_NN": 0.3929824561403509
  },
  "2": {
    "eligible_episodes": 21,
    "episode_auc": {
      "object_cls": 0.40955026455026455,
      "stored_nn_mean": 0.36494708994708996
    },
    "CLS_minus_NN": 0.044603174603174596
  },
  "3": {
    "eligible_episodes": 18,
    "episode_auc": {
      "object_cls": 0.33730158730158727,
      "stored_nn_mean": 0.23148148148148145
    },
    "CLS_minus_NN": 0.10582010582010581
  }
}

Secondary pooled-region summaries:

{
  "object_cls": {
    "region_pooled_auc": 0.46615530135083205,
    "pixel_mass_weighted_auc": 0.5458148444351932,
    "positive_regions": 378,
    "negative_regions": 358,
    "fixed_zero_margin_TP": 252,
    "fixed_zero_margin_FP": 257
  },
  "stored_nn_mean": {
    "region_pooled_auc": 0.3590420028967515,
    "pixel_mass_weighted_auc": 0.1889570790884121,
    "positive_regions": 378,
    "negative_regions": 358,
    "fixed_zero_margin_TP": 191,
    "fixed_zero_margin_FP": 256
  }
}

Full1200 raw activations are missing. The raw/projection controls remain explicitly limited to original241; region-averaged NN is a different full1200 control. All197 original CLS margins reproduce exactly.
