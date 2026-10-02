# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Candidate-mask decisions from shared, additive foreground-mass evidence.

The bounds and stopping certificate are deterministic, conditional on valid
simultaneous mass intervals. This module does not manufacture such intervals
from model scores and does not claim that a vision verifier is calibrated.
The acquisition rule is a decision-gap heuristic, not an optimal policy theorem.
"""

from dataclasses import dataclass
import numpy as np


def common_atoms(masks):
    """Return candidate-by-atom membership, atom areas, and pixel-to-atom ids."""
    masks = np.asarray(masks, dtype=bool)
    signatures, labels, areas = np.unique(
        masks.T, axis=0, return_inverse=True, return_counts=True
    )
    return signatures.T, areas.astype(float), labels


def iou_from_masses(membership, areas, masses):
    membership = np.asarray(membership, dtype=float)
    numerator = membership @ masses
    denominator = membership @ areas + (1 - membership) @ masses
    # Standard present-target episodes: an empty candidate has IoU zero.
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def iou_bounds(membership, areas, lower, upper):
    membership = np.asarray(membership, dtype=float)
    candidate_area = membership @ areas
    lower_denom = candidate_area + (1 - membership) @ upper
    upper_denom = candidate_area + (1 - membership) @ lower
    lo = np.divide(membership @ lower, lower_denom, out=np.zeros(len(membership)), where=lower_denom > 0)
    hi = np.divide(membership @ upper, upper_denom, out=np.zeros(len(membership)), where=upper_denom > 0)
    return lo, hi


def fractional_subset(numerator, denominator_increment, denominator_base):
    """Maximize a nonnegative linear-fractional objective by sorted prefixes."""
    numerator = np.asarray(numerator, dtype=float)
    increment = np.asarray(denominator_increment, dtype=float)
    assert np.all(numerator >= 0) and np.all(increment >= 0) and denominator_base >= 0
    ratio = np.divide(numerator, increment, out=np.zeros_like(numerator), where=increment > 0)
    ratio[(increment == 0) & (numerator > 0)] = np.inf
    order = np.argsort(-ratio, kind="stable")
    top = np.cumsum(numerator[order])
    bottom = denominator_base + np.cumsum(increment[order])
    objective = np.divide(top, bottom, out=np.zeros_like(top), where=bottom > 0)
    mask = np.zeros(len(numerator), dtype=bool)
    if objective.max(initial=0) > 0:
        end = int(np.argmax(objective))
        mask[order[:end+1]] = True
        return mask, float(objective[end])
    return mask, 0.


def robust_union_decision(areas, lower, upper):
    """Optimal IoU lower bound and a regret certificate over ALL atom unions.

    No shape prior is imposed. Applications needing one must restrict the
    candidate family, e.g. use the finite-candidate MassDecision class.
    """
    areas, lower, upper = map(lambda x: np.asarray(x, dtype=float), (areas, lower, upper))
    assert np.all(0 <= lower) and np.all(lower <= upper) and np.all(upper <= areas)
    selected, best_lower = fractional_subset(lower, areas - upper, float(upper.sum()))
    _, best_upper = fractional_subset(upper, areas - lower, float(lower.sum()))
    return selected, best_lower, max(0., best_upper - best_lower)


@dataclass
class MassDecision:
    membership: np.ndarray
    areas: np.ndarray
    estimate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self):
        self.membership = np.asarray(self.membership, dtype=bool)
        for key in ("areas", "estimate", "lower", "upper"):
            setattr(self, key, np.asarray(getattr(self, key), dtype=float).copy())
        assert np.all(self.lower >= 0) and np.all(self.upper <= self.areas)
        assert np.all(self.lower <= self.upper)
        self.estimate = np.clip(self.estimate, self.lower, self.upper)

    def recommendation(self):
        return int(np.argmax(iou_from_masses(self.membership, self.areas, self.estimate)))

    def certificate(self):
        """Return a robust incumbent and an upper bound on candidate-set regret."""
        lo, hi = iou_bounds(self.membership, self.areas, self.lower, self.upper)
        incumbent = int(np.argmax(lo))
        return incumbent, max(0., float(hi.max() - lo[incumbent]))

    def next_atom(self, costs=None):
        """Spend the next observation on evidence affecting the winner/challenger gap."""
        if costs is None:
            costs = np.ones_like(self.areas)
        costs = np.asarray(costs, dtype=float)
        assert np.all(costs > 0)
        uncertain = self.upper - self.lower > 1e-10
        if not uncertain.any():
            return None
        incumbent = self.recommendation()
        _, hi = iou_bounds(self.membership, self.areas, self.lower, self.upper)
        hi[incumbent] = -np.inf
        challenger = int(np.argmax(hi))
        pair = self.membership[[incumbent, challenger]]
        scores = np.full(len(self.areas), -np.inf)
        for atom in np.flatnonzero(uncertain):
            low = self.estimate.copy()
            high = self.estimate.copy()
            low[atom] = self.lower[atom]
            high[atom] = self.upper[atom]
            jl = iou_from_masses(pair, self.areas, low)
            ju = iou_from_masses(pair, self.areas, high)
            scores[atom] = abs((ju[1] - ju[0]) - (jl[1] - jl[0])) / costs[atom]
        if scores.max() < 1e-12:
            scores = np.where(uncertain, (self.upper - self.lower) / costs, -np.inf)
        return int(np.argmax(scores))

    def observe(self, atom, lower, upper, estimate):
        """Intersect an additional observation with existing evidence."""
        lo = max(self.lower[atom], float(lower))
        hi = min(self.upper[atom], float(upper))
        if lo > hi + 1e-10:
            raise ValueError("Inconsistent mass intervals; no certificate may be issued.")
        self.lower[atom], self.upper[atom] = lo, hi
        self.estimate[atom] = np.clip(estimate, lo, hi)


def self_check():
    """Check the algebra against exhaustive binary masks and random intervals."""
    rng = np.random.default_rng(20261002)
    interval_checks = stopping_checks = increment_checks = 0
    for _ in range(2000):
        pixels = 12
        truth = rng.integers(0, 2, pixels).astype(bool)
        if not truth.any():
            truth[0] = True
        masks = rng.integers(0, 2, (8, pixels)).astype(bool)
        membership, areas, labels = common_atoms(masks)
        mass = np.bincount(labels, weights=truth, minlength=len(areas))
        lower = np.maximum(0, mass - rng.random(len(areas)) * areas)
        upper = np.minimum(areas, mass + rng.random(len(areas)) * areas)
        actual = np.array([(m & truth).sum() / max((m | truth).sum(), 1) for m in masks])
        inferred = iou_from_masses(membership, areas, mass)
        assert np.allclose(actual, inferred)
        lo, hi = iou_bounds(membership, areas, lower, upper)
        assert np.all(lo <= actual + 1e-12) and np.all(actual <= hi + 1e-12)
        interval_checks += 1
        engine = MassDecision(membership, areas, mass, lower, upper)
        incumbent, regret_bound = engine.certificate()
        assert actual.max() - actual[incumbent] <= regret_bound + 1e-12
        stopping_checks += 1
        candidate = masks[0]
        addition = (~candidate) & rng.integers(0, 2, pixels).astype(bool)
        if addition.any():
            j = actual[0]
            p = (addition & truth).sum() / addition.sum()
            expanded = candidate | addition
            gain = (expanded & truth).sum() / max((expanded | truth).sum(), 1) - j
            sign = p - j / (1 + j)
            assert np.sign(gain) == np.sign(sign) or abs(gain) < 1e-12 or abs(sign) < 1e-12
            increment_checks += 1
    exhaustive_checks = 0
    for _ in range(100):
        atoms = 6
        areas = rng.uniform(.1, 5, atoms)
        truth = areas * rng.random(atoms)
        lower = truth * rng.random(atoms)
        upper = truth + (areas - truth) * rng.random(atoms)
        family = ((np.arange(2 ** atoms)[:, None] >> np.arange(atoms)) & 1).astype(bool)
        lo, hi = iou_bounds(family, areas, lower, upper)
        chosen, bound, gap = robust_union_decision(areas, lower, upper)
        assert abs(bound - lo.max()) < 1e-10
        actual = iou_from_masses(family, areas, truth)
        chosen_quality = iou_from_masses(chosen[None], areas, truth)[0]
        assert actual.max() - chosen_quality <= gap + 1e-10
        # The uniform L1 mass-error bound holds even for correlated labels.
        estimate = areas * rng.random(atoms)
        estimated_quality = iou_from_masses(family, areas, estimate)
        error = np.abs(estimate - truth).sum() / truth.sum()
        assert np.max(np.abs(estimated_quality - actual)) <= error + 1e-10
        exhaustive_checks += 1
    return {"interval_checks": interval_checks, "stopping_checks": stopping_checks,
            "increment_checks": increment_checks, "exhaustive_union_and_error_checks": exhaustive_checks,
            "passed": True}


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(), indent=2))
