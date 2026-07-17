"""Seed discipline helpers for reproducible fits."""

import random
import statistics
from collections.abc import Callable, Sequence

import numpy as np
import torch

__all__ = ["fit_seeds", "seed_everything"]


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch global RNGs in one call.

    Args:
        seed: The seed applied to all three generators.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)  # pyright: ignore[reportUnknownMemberType]


def fit_seeds[EstimatorT](
    factory: Callable[[int], EstimatorT],
    x: object,
    y: object | None = None,
    *,
    seeds: Sequence[int],
) -> dict[str, object]:
    """Fit one estimator per seed and aggregate the headline read-outs.

    Args:
        factory: Builds a fresh estimator for a given seed (typically
            ``lambda seed: IGLRegressor(..., random_state=seed)``).
        x: Training inputs, forwarded to ``fit``.
        y: Optional training targets, forwarded to ``fit``.
        seeds: The seed list — e.g. a project's canonical seed protocol.

    Returns:
        A dict with ``"seeds"``, ``"estimators"`` (the fitted instances),
        and ``"aggregate"``: for each numeric read-out found on the fitted
        estimators (``effective_dimension_``, the final validation metric),
        its mean, std, median, and IQR across seeds.
    """
    estimators: list[EstimatorT] = []
    collected: dict[str, list[float]] = {}
    for seed in seeds:
        estimator = factory(seed)
        fit = estimator.fit  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
        fit(x) if y is None else fit(x, y)
        estimators.append(estimator)
        effective = getattr(estimator, "effective_dimension_", None)
        if effective is not None:
            collected.setdefault("effective_dimension", []).append(float(effective))
        history = getattr(estimator, "history_", None)
        if history is not None and history.val_metric:
            collected.setdefault("final_val_metric", []).append(float(history.val_metric[-1]))

    aggregate: dict[str, dict[str, float]] = {}
    for name, values in collected.items():
        quartiles = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
        aggregate[name] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "median": statistics.median(values),
            "iqr": quartiles[2] - quartiles[0],
        }
    return {"seeds": list(seeds), "estimators": estimators, "aggregate": aggregate}
