# Fixed reference-positive RGB-erasure ablation

GT boxes remain privileged; one causal intervention changes only reference-FG erasure.

{
  "primary": {
    "naturalFG": {
      "mean": 0.4949051741908885,
      "ci95": [
        0.4008913502109705,
        0.5854733013088276
      ],
      "eligible_episodes": 77
    },
    "maskedFG": {
      "mean": 0.4339589775304062,
      "ci95": [
        0.34009941957263384,
        0.5272003345373445
      ],
      "eligible_episodes": 77
    },
    "NN": {
      "mean": 0.24328488971346113,
      "ci95": [
        0.1628736808706688,
        0.33292770245114167
      ],
      "eligible_episodes": 77
    }
  },
  "natural_minus_masked": {
    "mean": 0.06094619666048238,
    "ci95": [
      -0.03251262315270936,
      0.15030946913133322
    ],
    "eligible_episodes": 77
  },
  "natural_vs_NN_and_chance": {
    "CLS_minus_NN": {
      "mean": 0.25162028447742735,
      "ci95": [
        0.1476864111498258,
        0.3557145165945164
      ],
      "eligible_episodes": 77
    },
    "CLS_minus_chance": {
      "mean": -0.005094825809111522,
      "ci95": [
        -0.09910864978902953,
        0.08547330130882762
      ],
      "eligible_episodes": 77
    }
  },
  "natural_folds": {
    "0": {
      "eligible_episodes": 19,
      "episode_auc": {
        "object_cls": 0.4307017543859649,
        "stored_nn_mean": 0.17293233082706766
      },
      "CLS_minus_NN": 0.2577694235588972
    },
    "1": {
      "eligible_episodes": 19,
      "episode_auc": {
        "object_cls": 0.5684210526315789,
        "stored_nn_mean": 0.19035087719298244
      },
      "CLS_minus_NN": 0.3780701754385965
    },
    "2": {
      "eligible_episodes": 21,
      "episode_auc": {
        "object_cls": 0.515899470899471,
        "stored_nn_mean": 0.36494708994708996
      },
      "CLS_minus_NN": 0.15095238095238095
    },
    "3": {
      "eligible_episodes": 18,
      "episode_auc": {
        "object_cls": 0.46058201058201054,
        "stored_nn_mean": 0.23148148148148145
      },
      "CLS_minus_NN": 0.22910052910052908
    }
  }
}
