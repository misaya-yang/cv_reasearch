1. Assumption: identity ordering may be limited by final/source-debiased representations, independently of stopping.
2. Prediction: all12 source masks match saved native bits; first2 observed/unobserved exact; hooks10CPU cases exact; full tensors native-dtype<=4.5GiB,5GiB reserve. No task-gain prediction.
3. Match: one frozen cache enables paired CPU FN/FP and score-conditioned identity audits of raw/pre-debias/actual-debias/middle/native-QKV, without more GPU encoding.
4. Mismatch: preserve ERROR and partial compact evidence; repair the exact interface defect, never substitute recomputed QKV, averaged/low-rank features, old metric/decoder or precision rounding. No automatic GPU launch or shutdown.
