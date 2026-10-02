"""CPU algebra diagnostics, not pretrained SAM quality or a GPU implementation.

Dense inverses are deliberately limited to tiny grids. This is an inspectable
reference for a shared image-conditioned operator with independent constrained
right-hand sides, plus permutation-safe within-prompt hypothesis matching.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def image_laplacian(features, sigma=0.2):
    """Four-neighbor graph; similarity is image-only, never other prompts."""
    h, w, _ = features.shape
    lap = np.zeros((h * w, h * w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            for dy, dx in ((0, 1), (1, 0)):
                yy, xx = y + dy, x + dx
                if yy >= h or xx >= w:
                    continue
                i, j = y * w + x, yy * w + xx
                dist = np.sum((features[y, x] - features[yy, xx]) ** 2)
                weight = np.exp(-dist / (2 * sigma * sigma))
                lap[i, i] += weight
                lap[j, j] += weight
                lap[i, j] -= weight
                lap[j, i] -= weight
    return lap


class SharedImageOperator:
    def __init__(self, features, strength=2.0, sigma=0.2):
        if strength < 0 or sigma <= 0:
            raise ValueError("strength must be nonnegative and sigma positive")
        self.shape = features.shape[:2]
        self.a = np.eye(np.prod(self.shape)) + strength * image_laplacian(features, sigma)
        # Reference only: dense N^2 storage is not suitable for production.
        self.inverse = np.linalg.solve(self.a, np.eye(self.a.shape[0]))

    def refine(self, logits, points=(), labels=(), target_margin=2.0):
        """Independent K-mask refinement with exact point-value constraints.

        logits: [K,H,W]. All K masks retained. points: integer [J,2] (y,x).
        labels: binary positive/negative. No box-to-positive-area shortcut.
        """
        if logits.shape[1:] != self.shape or target_margin <= 0:
            raise ValueError("grid mismatch or invalid point margin")
        points, labels = list(points), list(labels)
        if len(points) != len(labels):
            raise ValueError("one binary label is needed per point")
        h, w = self.shape
        unique = {}
        for (y, x), label in zip(points, labels):
            if label not in (0, 1) or not (0 <= y < h and 0 <= x < w):
                raise ValueError("invalid point or label")
            index = y * w + x
            if index in unique and unique[index] != label:
                raise ValueError("conflicting labels at the same raster location")
            unique[index] = label
        flat = logits.reshape(logits.shape[0], -1)
        unconstrained = np.einsum("kn,mn->km", flat, self.inverse, optimize=False)
        if not unique:
            return unconstrained.reshape(logits.shape)
        indices = np.array(list(unique))
        targets = target_margin * (2 * np.array(list(unique.values())) - 1)
        columns = self.inverse[:, indices]
        schur = self.inverse[np.ix_(indices, indices)]
        residual = targets[None] - unconstrained[:, indices]
        multipliers = np.linalg.solve(schur, residual.T).T
        return (unconstrained + np.einsum("kj,nj->kn", multipliers, columns, optimize=False)).reshape(logits.shape)


def pairwise_mask_iou(first, second):
    a, b = first.reshape(len(first), -1), second.reshape(len(second), -1)
    intersections = np.logical_and(a[:, None], b[None]).sum(-1)
    unions = np.logical_or(a[:, None], b[None]).sum(-1)
    return np.divide(intersections, unions, out=np.ones_like(intersections, dtype=float), where=unions > 0)


def match_hypotheses(reference_logits, perturbed_logits):
    """Use masks alone to align the K outputs of one prompt and one perturbation.

    For K=4 exhaustive assignment is an exact 24-permutation CPU reference.
    Stable token IDs are not assumed. This does not establish semantic quality.
    """
    if len(reference_logits) != len(perturbed_logits):
        raise ValueError("candidate counts must match")
    similarity = pairwise_mask_iou(reference_logits > 0, perturbed_logits > 0)
    permutations = itertools.permutations(range(len(reference_logits)))
    assignment = max(permutations, key=lambda p: sum(similarity[i, j] for i, j in enumerate(p)))
    return np.array(assignment), similarity[np.arange(len(assignment)), assignment]


def threshold_stability(logits, delta=0.05):
    flat = logits.reshape(logits.shape[0], -1)
    intersection = (flat > delta).sum(-1)
    union = (flat > -delta).sum(-1)
    return np.divide(intersection, union, out=np.ones_like(intersection, dtype=float), where=union > 0)


def diagnostics():
    rng = np.random.default_rng(3071)
    h, w, k = 8, 10, 4
    features = np.zeros((h, w, 1))
    features[:, w // 2:] = 1.0
    operator = SharedImageOperator(features)
    points, labels = [(1, 1), (6, 8)], [1, 0]
    logits = rng.normal(size=(k, h, w))
    refined = operator.refine(logits, points, labels)
    indices = np.array([y * w + x for y, x in points])
    constraint = np.eye(h * w)[indices]
    targets = np.array([2.0, -2.0])
    block = np.block([[operator.a, constraint.T], [constraint, np.zeros((2, 2))]])
    direct = np.linalg.solve(block, np.concatenate((logits.reshape(k, -1), np.tile(targets, (k, 1))), axis=1).T).T[:, :h * w]
    kkt_error = float(np.max(np.abs(direct - refined.reshape(k, -1))))
    constraint_error = float(np.max(np.abs(refined.reshape(k, -1)[:, indices] - targets)))

    # Other independent prompt batches cannot alter a prompt's operator/RHS.
    prompt_rhs = rng.normal(size=(3, k, h, w))
    batch = np.stack([operator.refine(x, points, labels) for x in prompt_rhs])
    permutation = [2, 0, 1]
    permuted = np.stack([operator.refine(prompt_rhs[i], points, labels) for i in permutation])
    order_error = float(np.max(np.abs(permuted - batch[permutation])))

    # Matching avoids artificial inconsistency due only to multimask permutation.
    shift = np.array([2, 0, 3, 1])
    assignment, scores = match_hypotheses(logits, logits[shift])
    permutation_matching_error = float(np.max(np.abs(scores - 1.0)))

    # Deliberate counterexample: threshold-stable confident outputs can be wrong.
    ground_truth = np.zeros((h, w), dtype=bool)
    ground_truth[:, :w // 2] = True
    confident_wrong = np.where(~ground_truth, 5.0, -5.0)[None]
    stable_wrong = float(threshold_stability(confident_wrong)[0])
    wrong_iou = float(pairwise_mask_iou(confident_wrong > 0, ground_truth[None])[0, 0])

    # Reject inconsistent constraints rather than pretending there is a solution.
    conflicting_rejected = False
    try:
        operator.refine(logits, [(1, 1), (1, 1)], [0, 1])
    except ValueError:
        conflicting_rejected = True
    min_eigenvalue = float(np.linalg.eigvalsh(operator.a).min())
    assert kkt_error < 1e-10 and constraint_error < 1e-10
    assert order_error == 0 and permutation_matching_error == 0
    assert stable_wrong == 1.0 and wrong_iou == 0.0
    assert conflicting_rejected and min_eigenvalue > 0
    return {
        "status": "CPU_ALGEBRA_ONLY_NO_SAM_QUALITY_OR_SPEED_CLAIM",
        "seed": 3071, "grid": [h, w], "all_masks": k,
        "kkt_max_abs_error": kkt_error,
        "point_constraint_max_abs_error": constraint_error,
        "independent_prompt_permutation_max_abs_error": order_error,
        "hypothesis_permutation_matching_max_abs_error": permutation_matching_error,
        "minimum_operator_eigenvalue": min_eigenvalue,
        "conflicting_point_labels_rejected": conflicting_rejected,
        "stable_but_wrong_counterexample": {"threshold_stability": stable_wrong, "synthetic_ground_truth_iou": wrong_iou},
        "limitations": ["Dense inverse is a tiny CPU reference only", "No pretrained weights or real annotated data used", "Point constraints apply to rasterized coordinates", "No implication that image smoothing improves real masks"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("cpu_algebra_results.json"))
    args = parser.parse_args()
    results = diagnostics()
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
