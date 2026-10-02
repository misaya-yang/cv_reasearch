"""Synthetic contract checks, not model quality measurements."""
import json
from pathlib import Path

import numpy as np

from diagnose_oracle_gap import diagnose


gt = np.zeros((2, 12, 12), dtype=bool)
gt[:, 3:9, 3:9] = True
masks = np.repeat(gt[:, None], 4, axis=1)
masks[:, 1] = ~gt
prediction = np.array([[0.1, 0.9, 0.5, 0.3], [0.1, 0.9, 0.5, 0.3]])
low = np.where(masks, 2.0, -2.0)
arrays = {"masks": masks, "iou_prediction": prediction, "gt": gt, "low_resolution_logits": low}
results = diagnose(arrays)
assert results["policies"]["multimask_iou_head_1to3"]["mean_iou"] == 0.0
assert results["oracle_diagnostic_extra_ground_truth_not_deployable"]["mean_multimask_iou_gap"] == 1.0
assert results["policies"]["single_token0"]["mean_boundary_iou"] == 1.0
assert results["policies"]["sam2_snapshot_dynamic_rule_on_supplied_outputs"]["mean_iou"] == 1.0
float_results = diagnose({**arrays, "logits": low, "masks": masks})
assert float_results == results
logit_only = dict(arrays)
del logit_only["masks"]
logit_only["logits"] = low
assert diagnose(logit_only) == results
status = {"status": "SYNTHETIC_CONTRACT_ONLY", "all_masks": 4,
          "oracle_gap_selection_check": "PASS", "full_mask_vs_logit_input_check": "PASS",
          "boundary_identical_mask_check": "PASS", "local_sam2_stability_rule_check": "PASS"}
Path(__file__).with_name("diagnostic_contract_results.json").write_text(json.dumps(status, indent=2) + "\n")
print(json.dumps(status, indent=2))
