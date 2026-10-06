"""Exact GT capacity oracle for a laminar query-region tree.

This module measures candidate capacity, not reference-based deployable selection.
Node TP/FP counts require query ground truth. Sibling regions must be disjoint;
descendants must lie inside ancestors. Counts alone cannot verify those mask facts.
The metric sums intersections/unions within each class before averaging classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RegionCounts:
    tp: int
    fp: int
    children: tuple[int, ...] = ()
    eligible: bool = True


@dataclass(frozen=True)
class OracleTree:
    nodes: tuple[RegionCounts, ...]
    root: int
    gt_area: int
    episode_id: str = ""

    def postorder(self) -> tuple[int, ...]:
        """Validate the count/topology contract and return child-before-parent order."""
        if not self.nodes or not 0 <= self.root < len(self.nodes):
            raise ValueError("A nonempty tree with a valid root is required")
        if type(self.gt_area) is not int or self.gt_area < 0:
            raise ValueError("GT area must be a nonnegative integer")
        parents = [0] * len(self.nodes)
        for node in self.nodes:
            if any(type(x) is not int or x < 0 for x in (node.tp, node.fp)):
                raise ValueError("TP and FP must be nonnegative integers")
            if type(node.eligible) is not bool:
                raise ValueError("Node eligibility must be boolean")
            for child in node.children:
                if type(child) is not int or not 0 <= child < len(self.nodes):
                    raise ValueError("Invalid child index")
                parents[child] += 1
            if node.tp < sum(self.nodes[c].tp for c in node.children):
                raise ValueError("Disjoint children have more TP than their parent")
            if node.fp < sum(self.nodes[c].fp for c in node.children):
                raise ValueError("Disjoint children have more FP than their parent")
        if parents[self.root] or any(p != 1 for i, p in enumerate(parents) if i != self.root):
            raise ValueError("Each nonroot node must have exactly one parent")
        if self.nodes[self.root].tp > self.gt_area:
            raise ValueError("A candidate contains more TP than total GT area")
        seen: set[int] = set()
        order: list[int] = []
        stack = [(self.root, False)]
        while stack:
            index, done = stack.pop()
            if done:
                order.append(index)
                continue
            if index in seen:
                raise ValueError("Cycle or repeated child in tree")
            seen.add(index)
            stack.append((index, True))
            stack.extend((c, False) for c in reversed(self.nodes[index].children))
        if len(seen) != len(self.nodes):
            raise ValueError("All nodes must be reachable from the root")
        return tuple(order)


@dataclass(frozen=True)
class Choice:
    tp: int
    fp: int
    nodes: tuple[int, ...]


@dataclass(frozen=True)
class ClassOracle:
    intersection: int
    union: int
    selections: tuple[Choice, ...]
    iterations: int

    @property
    def iou(self) -> Fraction:
        return Fraction(self.intersection, self.union)


def _better(candidate: Choice, current: Choice | None, numerator: int, denominator: int) -> bool:
    """Integer arithmetic avoids float ties in the fractional optimization."""
    if current is None:
        return True
    score = candidate.tp * denominator - candidate.fp * numerator
    old = current.tp * denominator - current.fp * numerator
    if score != old:
        return score > old
    return (len(candidate.nodes), candidate.fp, candidate.nodes) < (
        len(current.nodes), current.fp, current.nodes
    )


def _linear_choice(
    tree: OracleTree, order: Sequence[int], max_nodes: int, numerator: int, denominator: int
) -> Choice:
    states: dict[int, list[Choice | None]] = {}
    empty = Choice(0, 0, ())
    for index in order:
        node = tree.nodes[index]
        combined: list[Choice | None] = [empty] + [None] * max_nodes
        for child in node.children:
            merged: list[Choice | None] = [None] * (max_nodes + 1)
            child_states = states.pop(child)
            for used, left in enumerate(combined):
                if left is None:
                    continue
                for extra, right in enumerate(child_states[: max_nodes - used + 1]):
                    if right is None:
                        continue
                    choice = Choice(left.tp + right.tp, left.fp + right.fp, left.nodes + right.nodes)
                    if _better(choice, merged[used + extra], numerator, denominator):
                        merged[used + extra] = choice
            combined = merged
        whole = Choice(node.tp, node.fp, (index,))
        if max_nodes and node.eligible and _better(whole, combined[1], numerator, denominator):
            combined[1] = whole
        states[index] = combined
    best = empty
    for choice in states[tree.root]:
        if choice is not None and _better(choice, best, numerator, denominator):
            best = choice
    return best


def _unbudgeted_choice(
    tree: OracleTree, order: Sequence[int], numerator: int, denominator: int
) -> Choice:
    states: dict[int, Choice] = {}
    for index in order:
        node = tree.nodes[index]
        parts = [states.pop(child) for child in node.children]
        best = Choice(
            sum(part.tp for part in parts),
            sum(part.fp for part in parts),
            tuple(i for part in parts for i in part.nodes),
        )
        whole = Choice(node.tp, node.fp, (index,))
        if node.eligible and _better(whole, best, numerator, denominator):
            best = whole
        states[index] = best
    return states[tree.root]


def solve_class_oracle(
    trees: Sequence[OracleTree], max_nodes: int | None, *, max_iterations: int = 256
) -> ClassOracle:
    """Maximize class-summed IoU with at most K regions per query, exactly.

    At ratio a/b, each node has integer weight b*TP-a*FP. Each tree is solved
    by antichain DP. Update a/b to the selected class intersection/union until
    the exact fractional residual is zero. A runtime guard raises on exhaustion;
    it never reports an approximate optimum. Empty selections are permitted.
    None removes the node budget while retaining candidate-node eligibility.
    """
    if not trees or (max_nodes is not None and (type(max_nodes) is not int or max_nodes < 0)):
        raise ValueError("Provide trees and a nonnegative integer node budget")
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("Iteration guard must be a positive integer")
    orders = tuple(tree.postorder() for tree in trees)
    gt_area = sum(tree.gt_area for tree in trees)
    if gt_area == 0:
        raise ValueError("A class must have positive total GT area; absent-class convention is unspecified")
    ratio = Fraction(0)
    for iteration in range(1, max_iterations + 1):
        choices = tuple(
            (
                _unbudgeted_choice(tree, order, ratio.numerator, ratio.denominator)
                if max_nodes is None
                else _linear_choice(tree, order, max_nodes, ratio.numerator, ratio.denominator)
            )
            for tree, order in zip(trees, orders)
        )
        intersection = sum(choice.tp for choice in choices)
        union = gt_area + sum(choice.fp for choice in choices)
        residual = intersection * ratio.denominator - ratio.numerator * union
        if residual < 0:
            raise RuntimeError("Fractional residual decreased; the oracle contract is violated")
        if residual == 0:
            return ClassOracle(intersection, union, choices, iteration)
        ratio = Fraction(intersection, union)
    raise RuntimeError("Iteration guard reached before proving exact optimality")


def solve_class_miou_oracle(
    classes: Mapping[str, Sequence[OracleTree]], max_nodes: int | None
) -> tuple[Fraction, dict[str, ClassOracle]]:
    """Return exact macro class IoU and independently optimal per-class masks."""
    if not classes:
        raise ValueError("At least one evaluated class is required")
    results = {label: solve_class_oracle(trees, max_nodes) for label, trees in classes.items()}
    macro = sum((result.iou for result in results.values()), Fraction(0)) / len(results)
    return macro, results
