"""Small exact-arithmetic checks of the proposed decision rule, not an image run.

No images, features, model calls or new segmentation experiments are involved.
The finite Boolean check supplements the general proof in design.md.
"""

import hashlib
import json
import time
from fractions import Fraction as F
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def iou(mask, truth):
    assert truth
    return F(bin(mask & truth).count("1"), bin(mask | truth).count("1"))


def propagate(v):
    """Exact inverse of I + L for a 1/100 edge between vertices 0 and 1."""
    return [(101 * v[0] + v[1]) / 102,
            (v[0] + 101 * v[1]) / 102, v[2]]


def mask(v):
    return sum(1 << i for i, x in enumerate(v) if x > F(1, 2))


def main():
    start = time.process_time()
    output = HERE / "theory_check.json"
    assert not output.exists(), "Do not overwrite a completed check."

    # Exhaust all four-pixel intervals, all baselines and every nonempty truth
    # they cover. This checks the actual IoU consequence, not just the formula.
    checked = 0
    full = 15
    for lower in range(16):
        for upper in range(16):
            if lower & (full ^ upper):
                continue
            for baseline in range(16):
                chosen = (baseline | lower) & upper
                for truth in range(1, 16):
                    if lower & (full ^ truth) or truth & (full ^ upper):
                        continue
                    added = chosen & (full ^ baseline)
                    deleted = baseline & (full ^ chosen)
                    assert not (added & (full ^ truth))
                    assert not (deleted & truth)
                    assert iou(chosen, truth) >= iou(baseline, truth)
                    checked += 1

    # Two seed hypotheses, with one shared background correction. Averaging
    # changes an ambiguous pixel; preserving the interval retains base evidence.
    y = [F(55, 100), F(49, 100), F(60, 100)]
    priors = [[F(1), F(2, 5), F(0)], [F(2, 5), F(1), F(0)]]
    beta = F(1, 4)
    base = propagate(y)

    def field(p):
        return propagate([a + beta * (b - F(1, 2)) for a, b in zip(y, p)])

    branches = [field(p) for p in priors]
    lower = field([min(p[i] for p in priors) for i in range(3)])
    upper = field([max(p[i] for p in priors) for i in range(3)])
    assert all(lower[i] <= z[i] <= upper[i] for z in branches for i in range(3))
    chosen = [min(max(x, lo), hi) for x, lo, hi in zip(base, lower, upper)]
    averaged = field([sum(p[i] for p in priors) / 2 for i in range(3)])
    truth = 1  # The first of three synthetic pixels only.
    assert mask(chosen) == ((mask(base) | mask(lower)) & mask(upper))
    assert mask(chosen) == truth
    assert mask(averaged) != truth
    assert mask(base) != truth
    singleton = [min(max(x, z), z) for x, z in zip(base, branches[0])]
    assert singleton == branches[0]

    # Re-read an existing stage record; these are historical counts, not new
    # masks or a measurement of the proposed ambiguity mechanism.
    source = ROOT / "evidence/local/research_20261005/stage241_v1/report.json"
    report = json.loads(source.read_text())
    step = next(s for s in report["chains"]["foris"]
                if s["step"] == "foris.s3_vote -> foris.s3")
    stats = dict(step)
    stats["correct_edits"] = step["add_TP"] + step["delete_FP"]
    stats["wrong_edits"] = step["add_FP"] + step["delete_TP"]

    result = {
        "scope": "exact arithmetic on a synthetic three-pixel example and finite Boolean proof checks",
        "real_segmentation_runs": 0,
        "four_pixel_coverage_cases_checked": checked,
        "rational_example": {
            "baseline_field": list(map(str, base)),
            "branch_fields": [list(map(str, z)) for z in branches],
            "lower_field": list(map(str, lower)),
            "upper_field": list(map(str, upper)),
            "chosen_field": list(map(str, chosen)),
            "masks_as_bitsets": {
                "baseline": mask(base), "hard_second_seed": mask(branches[1]),
                "uniform_seed_average": mask(averaged), "delayed_decision": mask(chosen),
                "truth": truth},
            "IoU": {name: str(iou(mask(z), truth)) for name, z in {
                "baseline": base, "hard_second_seed": branches[1],
                "uniform_seed_average": averaged, "delayed_decision": chosen}.items()},
        },
        "existing_stage_record": stats,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "local_process_cpu_seconds": time.process_time() - start,
        "limitations": [
            "synthetic arithmetic is not evidence of segmentation accuracy",
            "coverage by the new seed envelope has not been measured",
            "the old record has no alternative-seed outputs or top-two seed margins",
            "old stage counts do not establish that near ties caused the wrong edits",
        ],
    }
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"coverage_cases": checked, "toy_IoU": result["rational_example"]["IoU"],
                      "old_correct_edits": stats["correct_edits"],
                      "old_wrong_edits": stats["wrong_edits"],
                      "local_cpu_seconds": result["local_process_cpu_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
