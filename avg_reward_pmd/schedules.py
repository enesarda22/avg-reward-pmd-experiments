"""Step-size schedules for PMD.

The master lemma's sharp asymptotic rate (Lemma 3.7(ii), Corollary 4.3)
requires the step-size ratio to diverge: ``eta_t / eta_{t-1} -> infinity``.
A plain geometric schedule ``eta_0 * gamma^t`` has constant ratio ``gamma``
and does *not* satisfy this hypothesis. The ``super_geometric`` schedule
below does: its ratio is ``beta^{t+1}``, which diverges.
"""

from __future__ import annotations

import numpy as np


def geometric(eta0: float, gamma: float, horizon: int) -> np.ndarray:
    """eta_t = eta0 * gamma^t (constant ratio, satisfies Lemma 3.7(i) only)."""
    return eta0 * np.power(gamma, np.arange(horizon))


def super_geometric(eta0: float, beta: float, horizon: int) -> np.ndarray:
    """eta_t = eta0 * beta^{t(t+1)/2}; ratio beta^{t+1} diverges.

    Satisfies the hypothesis of Lemma 3.7(ii) but has ratio 1 at t = 0,
    so Lemma 3.7(i) may fail at the start. Use :func:`super_geometric_with_floor`
    for a schedule satisfying both.
    """
    t = np.arange(horizon)
    log_eta = np.log(eta0) + np.log(beta) * t * (t + 1) / 2.0
    return np.exp(log_eta)


def super_geometric_with_floor(
    eta0: float, c: float, alpha: float, horizon: int
) -> np.ndarray:
    """eta_t = eta0 * c^t * alpha^{t^2}; ratio c * alpha^{2t+1}.

    With ``c >= chi_bar / (chi_bar - 1)`` and ``alpha > 1``, satisfies both
    the non-asymptotic floor of Lemma 3.7(i) and the divergence condition
    of Lemma 3.7(ii).
    """
    t = np.arange(horizon)
    log_eta = np.log(eta0) + t * np.log(c) + t**2 * np.log(alpha)
    return np.exp(log_eta)
