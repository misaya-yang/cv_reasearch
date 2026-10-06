"""Small brute-force mathematical checks; no dataset, model or GPU calls."""

from __future__ import annotations

import itertools
import json
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ics.methods.hierarchy_oracle import (
    Choice, OracleTree, RegionCounts, solve_class_miou_oracle, solve_class_oracle,
)


def exhaustive_choices(tree: OracleTree, budget: int | None) -> list[Choice]:
    parents = {child: i for i, node in enumerate(tree.nodes) for child in node.children}

    def ancestor(a: int, b: int) -> bool:
        while b in parents:
            b = parents[b]
            if a == b:
                return True
        return False

    eligible = [i for i, node in enumerate(tree.nodes) if node.eligible]
    result = []
    for size in range((len(eligible) if budget is None else min(budget, len(eligible))) + 1):
        for ids in itertools.combinations(eligible, size):
            if any(ancestor(a, b) or ancestor(b, a) for a, b in itertools.combinations(ids, 2)):
                continue
            result.append(Choice(sum(tree.nodes[i].tp for i in ids), sum(tree.nodes[i].fp for i in ids), ids))
    return result


def main() -> None:
    rng = random.Random(0)

    def make_tree() -> OracleTree:
        leaves = tuple(RegionCounts(rng.randint(0, 7), rng.randint(0, 11), eligible=rng.random() > .2) for _ in range(4))
        left = RegionCounts(sum(n.tp for n in leaves[:2]), sum(n.fp for n in leaves[:2]), (3, 4))
        right = RegionCounts(sum(n.tp for n in leaves[2:]), sum(n.fp for n in leaves[2:]), (5, 6))
        root = RegionCounts(left.tp + right.tp, left.fp + right.fp, (1, 2), rng.random() > .2)
        return OracleTree((root, left, right, *leaves), 0, root.tp)

    started = time.monotonic()
    checked = 0
    for _ in range(80):
        trees = (make_tree(), make_tree())
        gt = sum(t.gt_area for t in trees)
        if not gt:
            continue
        previous = Fraction(-1)
        for budget in (0, 1, 2, 3, None):
            result = solve_class_oracle(trees, budget)
            brute = max(Fraction(a.tp + b.tp, gt + a.fp + b.fp) for a, b in itertools.product(*(exhaustive_choices(t, budget) for t in trees)))
            assert result.iou == brute
            assert result.iou >= previous
            previous = result.iou
            for tree, selection in zip(trees, result.selections):
                canonical = Choice(selection.tp, selection.fp, tuple(sorted(selection.nodes)))
                assert canonical in exhaustive_choices(tree, budget)
            checked += 1

    first = OracleTree((RegionCounts(10, 100, (1, 2)), RegionCounts(1, 0), RegionCounts(9, 100)), 0, 10, "first")
    second = OracleTree((RegionCounts(10, 100, (1, 2)), RegionCounts(7, 5, (3, 4)), RegionCounts(3, 95), RegionCounts(5, 0), RegionCounts(2, 5)), 0, 10, "second")
    local = [max(exhaustive_choices(t, 1), key=lambda c: Fraction(c.tp, t.gt_area + c.fp)) for t in (first, second)]
    local_iou = Fraction(sum(c.tp for c in local), 20 + sum(c.fp for c in local))
    joint = solve_class_oracle((first, second), 1)
    assert local_iou == Fraction(3, 10) and joint.iou == Fraction(8, 25)
    macro, _ = solve_class_miou_oracle({"a": (first, second), "b": (first,)}, 1)
    assert macro == (joint.iou + solve_class_oracle((first,), 1).iou) / 2

    malformed = OracleTree((RegionCounts(1, 0, (1, 1)), RegionCounts(0, 0)), 0, 1)
    try:
        solve_class_oracle((malformed,), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("Repeated children accepted")
    output = {
        "state": "EXACT_SOLVER_SYNTHETIC_CHECKS_PASSED",
        "bruteforce_comparisons": checked,
        "budgets": [0, 1, 2, 3, "unlimited"],
        "seed": 0,
        "elapsed_seconds": time.monotonic() - started,
        "counterexample": {"per_episode_selection_class_iou": float(local_iou), "class_joint_oracle_iou": float(joint.iou)},
        "dataset_cases_used": 0,
        "gpu_calls": 0,
        "limitation": "Synthetic checks only; no real hierarchy capacity or deployable selector validated",
    }
    target = Path(__file__).resolve().parents[1] / "evidence/local/research_20261006/hierarchy_joint/solver_check.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
