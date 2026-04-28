"""Reproduce the exact-PMD experiment reported in the paper's appendix.

Two cases on the same three-state MDP:

- Case I (multiple minimizers): ``chi_t`` stabilizes above 1 and the
  empirical suboptimality decays strictly faster than the non-asymptotic
  upper bound from Lemma 3.7(i), illustrating the tail regime of
  Corollary 4.3 where the effective rate is governed by
  ``chi = limsup chi_t`` rather than ``chi_bar = sup chi_t``.
- Case II (unique minimizer): ``chi_t -> 1`` and the rate collapses toward 0,
  illustrating the superlinear regime of Corollary 4.3.

Both cases use the mixed super-geometric schedule
``eta_t = eta_0 * c^t * alpha^{t^2}`` with ``c`` above the floor
``chi_bar / (chi_bar - 1)`` required by Lemma 3.7(i) and ``alpha > 1`` so
the ratio diverges as required by Lemma 3.7(ii).

Run (from the repo root):
    uv run python scripts/run_exact_pmd.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from avg_reward_pmd.metrics import summary
from avg_reward_pmd.mdp import three_state_mdp
from avg_reward_pmd.pmd import (
    _policy_kernel,
    differential_q,
    run_pmd,
    stationary_distribution,
    weighted_kl,
)
from avg_reward_pmd.plotting import rate_and_chi_figure
from avg_reward_pmd.schedules import super_geometric_with_floor

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

HORIZON = 80
ETA0 = 1e-1
C = 1.05  # >= chi_bar / (chi_bar - 1) for chi_bar up to ~20
ALPHA = 1.005  # > 1 so the ratio diverges
ETA_MINUS_1 = ETA0 / C  # largest eta_{-1} consistent with the floor hypothesis at t=0

PI_INIT = np.array(
    [
        [0.999, 0.001],
        [0.001, 0.999],
        [0.010, 0.990],
    ]
)


def reference_optimal_policy(mdp, unique_optimum: bool) -> tuple[np.ndarray, float]:
    """Optimal policy used to define chi_t, and its average cost rho_star."""
    if unique_optimum:
        pi_star = np.zeros((mdp.n_states, mdp.n_actions))
        pi_star[0, 1] = 1.0
        pi_star[1, 0] = 1.0
        pi_star[2, 0] = 1.0
    else:
        # Multi-minimizer: any policy choosing action 0 in state 2 is optimal.
        pi_star = np.full((mdp.n_states, mdp.n_actions), 0.5)
        pi_star[2] = np.array([1.0, 0.0])

    _, rho_star = differential_q(mdp, pi_star)
    return pi_star, rho_star


def initial_lyapunov(mdp, pi0, pi_star, rho_star, eta_minus_1, chi_bar) -> float:
    """Psi_0 = Delta_0 + D_0^* / (eta_{-1} * chi_bar).

    With D^* the state-weighted KL from pi_star to pi0 under d^{pi_star}.
    """
    d_star = stationary_distribution(_policy_kernel(mdp, pi_star))
    delta_0 = float(np.sum(d_star * (pi0 * mdp.r).sum(axis=-1)) - rho_star)
    # D_0^* = state-weighted KL from pi_star (target) to pi0 (current).
    d_0_star = weighted_kl(pi_star, pi0, d_star)
    return delta_0 + d_0_star / (eta_minus_1 * chi_bar)


def run_case(label: str, unique_optimum: bool) -> None:
    mdp = three_state_mdp(p=0.05, unique_optimum=unique_optimum)
    pi_star, rho_star = reference_optimal_policy(mdp, unique_optimum)

    schedule = super_geometric_with_floor(
        eta0=ETA0, c=C, alpha=ALPHA, horizon=HORIZON
    )
    result = run_pmd(mdp, PI_INIT, schedule, pi_star=pi_star, rho_star=rho_star)

    chi_bar = float(np.max(result.chi))
    psi_0 = initial_lyapunov(mdp, PI_INIT, pi_star, rho_star, ETA_MINUS_1, chi_bar)

    rate_and_chi_figure(result, FIG_DIR / f"pmd_{label}.pdf", psi_0=psi_0)

    s = summary(result)
    print(
        f"{label:>7s}  delta_final={s['delta_final']:.3e}  "
        f"chi_bar={s['chi_bar']:.2f}  chi_tail={s['chi_tail']:.2f}  "
        f"Psi_0={psi_0:.3f}"
    )


def main() -> None:
    run_case("multi", unique_optimum=False)
    run_case("unique", unique_optimum=True)


if __name__ == "__main__":
    main()
