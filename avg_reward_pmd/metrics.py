"""Diagnostics derived from a :class:`RunResult`."""

from __future__ import annotations

import numpy as np

from avg_reward_pmd.pmd import RunResult


def empirical_geometric_rate(delta: np.ndarray, floor: float = 1e-16) -> np.ndarray:
    """gamma_t = (delta_t / delta_0)^{1/t}, clipped for numerical stability.

    Returns NaN for ``t = 0`` and wherever ``delta_t`` drops below ``floor``.
    """
    t = np.arange(len(delta))
    ratio = np.clip(delta / max(delta[0], floor), floor, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma_t = np.exp(np.log(ratio) / np.where(t == 0, 1, t))
    gamma_t[0] = np.nan
    gamma_t[delta < floor] = np.nan
    return gamma_t


def tail_chi(chi: np.ndarray, tail_fraction: float = 0.2) -> float:
    """Median of the last ``tail_fraction`` of the chi_t trajectory."""
    n_tail = max(1, int(tail_fraction * len(chi)))
    return float(np.median(chi[-n_tail:]))


def summary(result: RunResult) -> dict[str, float]:
    """Scalar summary of a run."""
    return {
        "delta_final": float(result.delta[-1]),
        "chi_bar": float(np.max(result.chi)),
        "chi_tail": tail_chi(result.chi),
        "horizon": result.horizon,
    }
