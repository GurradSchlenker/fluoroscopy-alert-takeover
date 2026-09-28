"""The honest-split causal forest.

Ref: Algorithm 4, p. 18, steps 1-11::

    split O* into a structural half and an estimation half by patient
    for t = 1 to T: grow a causal tree on the structural half, splitting on the covariate
                    history only; within each leaf, estimate the mediated component on the
                    estimation half
    for each pre-specified stratum g: aggregate the leaf estimates weighted by leaf mass
                                      to obtain theta_g; form the interval from the
                                      aggregated influence function

The splitting criterion is the difference in the within-child exposure-outcome contrast,
which uses the covariate history and the outcome on the structural half only; the
mediated component is then read off the estimation half, which is what makes the split
honest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..estimators.influence import InfluenceEstimate, linear_contrast
from ..utils.types import BoolArray, FloatArray, IntArray
from .honest_split import HonestSplit, honest_split

_EPS = 1e-9


@dataclass(frozen=True)
class ForestConfig:
    """Tree count, leaf floor and depth of the causal forest.

    The manuscript prints the tree count and the minimum leaf size as inputs of Algorithm 4
    but not their values; both settings here are release engineering defaults.
    """

    trees: int = 200
    min_leaf: int = 50
    max_depth: int = 4
    thresholds_per_feature: int = 8
    seed: int = 20260101

    def __post_init__(self) -> None:
        if self.trees < 1 or self.min_leaf < 2 or self.max_depth < 1:
            raise ValueError("tree count, leaf floor and depth must be positive")


@dataclass
class TreeNode:
    """One node of a causal tree; a leaf has ``feature < 0``."""

    feature: int = -1
    threshold: float = 0.0
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature < 0


@dataclass
class HonestSplitCausalForest:
    """An ensemble of causal trees grown on the structural half."""

    config: ForestConfig
    split: HonestSplit = field(
        default_factory=lambda: HonestSplit(np.empty(0, np.bool_), np.empty(0, np.bool_))
    )
    trees: list[TreeNode] = field(default_factory=list)

    def fit(
        self,
        design: FloatArray,
        alert: FloatArray,
        outcome: FloatArray,
        region: BoolArray,
        patient_id: IntArray,
    ) -> HonestSplitCausalForest:
        """Grow the ensemble and record the honest split it used."""
        split = honest_split(patient_id, region, self.config.seed)
        generator = np.random.default_rng(self.config.seed + 1)
        structural_rows = np.nonzero(split.structural)[0]
        self.trees = []
        for _tree_index in range(self.config.trees):
            drawn = generator.choice(structural_rows, size=structural_rows.shape[0], replace=True)
            self.trees.append(
                _grow(
                    design,
                    alert,
                    outcome,
                    np.asarray(np.unique(drawn), dtype=np.int64),
                    depth=0,
                    config=self.config,
                )
            )
        self.split = split
        return self


def _descend(tree: TreeNode, row: FloatArray) -> int:
    """Leaf index of one row, counted left to right, returning -1 outside any leaf path."""
    node = tree
    index = 0
    while not node.is_leaf:
        left = node.left
        right = node.right
        if left is None or right is None:
            raise ValueError("the tree has an internal node without two children")
        if row[node.feature] <= node.threshold:
            node = left
        else:
            index += _leaf_count(left)
            node = right
    return index


def _leaf_count(tree: TreeNode) -> int:
    if tree.is_leaf:
        return 1
    assert tree.left is not None
    assert tree.right is not None
    return _leaf_count(tree.left) + _leaf_count(tree.right)


def _arm_contrast(alert: FloatArray, outcome: FloatArray, rows: IntArray) -> float:
    treated = rows[alert[rows] > 0.5]
    control = rows[alert[rows] <= 0.5]
    if treated.size == 0 or control.size == 0:
        return float("nan")
    return float(np.mean(outcome[treated]) - np.mean(outcome[control]))


def _best_split(
    design: FloatArray,
    alert: FloatArray,
    outcome: FloatArray,
    rows: IntArray,
    config: ForestConfig,
) -> tuple[int, float, float]:
    """Best split of a node by the between-child contrast difference, or ``(-1, 0, -inf)``."""
    node_contrast = _arm_contrast(alert, outcome, rows)
    if not np.isfinite(node_contrast):
        return -1, 0.0, float("-inf")
    best = (-1, 0.0, 0.0)
    features = np.arange(design.shape[1], dtype=np.int64)
    for feature in features:
        values = design[rows, feature]
        candidates = np.unique(
            np.quantile(values, np.linspace(0.1, 0.9, config.thresholds_per_feature))
        )
        for threshold in candidates:
            left = rows[values <= threshold]
            right = rows[values > threshold]
            if left.size < config.min_leaf or right.size < config.min_leaf:
                continue
            left_contrast = _arm_contrast(alert, outcome, left)
            right_contrast = _arm_contrast(alert, outcome, right)
            if not (np.isfinite(left_contrast) and np.isfinite(right_contrast)):
                continue
            weight = (left.size * right.size) / float(rows.size * rows.size)
            score = weight * (left_contrast - right_contrast) ** 2
            if score > best[2]:
                best = (int(feature), float(threshold), float(score))
    return best


def _grow(
    design: FloatArray,
    alert: FloatArray,
    outcome: FloatArray,
    rows: IntArray,
    depth: int,
    config: ForestConfig,
) -> TreeNode:
    """Recursively grow one causal tree; splitting never sees the estimation half."""
    if depth >= config.max_depth or rows.size < 2 * config.min_leaf:
        return TreeNode()
    feature, threshold, score = _best_split(design, alert, outcome, rows, config)
    if feature < 0 or score <= 0.0:
        return TreeNode()
    left_rows = rows[design[rows, feature] <= threshold]
    right_rows = rows[design[rows, feature] > threshold]
    return TreeNode(
        feature=feature,
        threshold=threshold,
        left=_grow(design, alert, outcome, left_rows, depth + 1, config),
        right=_grow(design, alert, outcome, right_rows, depth + 1, config),
    )


def tree_stratum_estimate(
    tree: TreeNode,
    design: FloatArray,
    terms_total: FloatArray,
    terms_mediated: FloatArray,
    estimation: BoolArray,
    stratum: BoolArray,
) -> InfluenceEstimate | None:
    """The mediated fraction of one stratum under one tree's partition.

    Leaf estimates carry the mass of the estimation-half rows they hold, so a leaf that
    fragments a stratum contributes in proportion to how much of the stratum it covers.
    """
    rows = np.nonzero(estimation & stratum)[0].astype(np.int64)
    if rows.size < 2:
        return None
    leaves = np.asarray([_descend(tree, design[row]) for row in rows], dtype=np.int64)
    total_influence = np.zeros(rows.size, dtype=np.float64)
    mediated_influence = np.zeros(rows.size, dtype=np.float64)
    total_point = 0.0
    mediated_point = 0.0
    for leaf in np.unique(leaves):
        position = np.nonzero(leaves == leaf)[0]
        if position.size < 2:
            continue
        weight = position.size / float(rows.size)
        leaf_total = float(np.mean(terms_total[rows[position]]))
        leaf_mediated = float(np.mean(terms_mediated[rows[position]]))
        total_point += weight * leaf_total
        mediated_point += weight * leaf_mediated
        total_influence[position] = terms_total[rows[position]] - leaf_total
        mediated_influence[position] = terms_mediated[rows[position]] - leaf_mediated
    if abs(total_point) <= _EPS:
        return None
    theta = mediated_point / total_point
    influence = (mediated_influence - theta * total_influence) / total_point
    return InfluenceEstimate(
        name="forest_mediated_fraction",
        point=float(theta),
        influence=np.asarray(influence, dtype=np.float64),
    )


def forest_stratum_estimate(
    forest: HonestSplitCausalForest,
    design: FloatArray,
    terms_total: FloatArray,
    terms_mediated: FloatArray,
    stratum: BoolArray,
) -> InfluenceEstimate | None:
    """Average the per-tree stratum estimates into one forest estimate."""
    if not forest.trees:
        raise ValueError("the forest has not been fitted")
    estimates = [
        tree_stratum_estimate(
            tree, design, terms_total, terms_mediated, forest.split.estimation, stratum
        )
        for tree in forest.trees
    ]
    usable = [estimate for estimate in estimates if estimate is not None]
    if len(usable) < 2:
        return None
    weight = 1.0 / len(usable)
    return linear_contrast("forest_stratum", [(weight, estimate) for estimate in usable])


def forest_strata(
    forest: HonestSplitCausalForest,
    design: FloatArray,
    terms_total: FloatArray,
    terms_mediated: FloatArray,
    strata: list[tuple[str, BoolArray]],
) -> dict[str, InfluenceEstimate]:
    """Per-stratum mediated fractions for every pre-specified stratum."""
    results: dict[str, InfluenceEstimate] = {}
    for label, mask in strata:
        estimate = forest_stratum_estimate(forest, design, terms_total, terms_mediated, mask)
        if estimate is not None:
            results[label] = estimate
    return results


__all__ = [
    "ForestConfig",
    "HonestSplitCausalForest",
    "TreeNode",
    "forest_strata",
    "forest_stratum_estimate",
    "tree_stratum_estimate",
]
