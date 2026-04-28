"""Inexact PMD experiment: sweep over critic error and step size.

Illustrates the floor structure predicted by Theorem 4.2:

- (a) Fixing the critic error ``epsilon_q`` and sweeping the step size
  ``eta_0``, all trajectories converge at different rates to essentially the
  same floor. This realizes the decoupling described in Section 3.1: step
  size controls the rate, not the floor.

- (b) Fixing the step size and sweeping ``epsilon_q``, the floors are
  monotonically ordered by the noise magnitude. The theoretical upper bound
  ``2(chi_bar + 1) * epsilon_q`` from Theorem 4.2 is loose for this small
  MDP, but the qualitative dependence of the floor on ``epsilon_q`` is
  visible.

Uses the unique-minimizer tie-broken instance of the three-state MDP. The
step-size schedule is geometric ``eta_t = eta_0 * gamma^t`` with ratio
``gamma`` chosen above the floor ``chi_bar / (chi_bar - 1)`` required by
Lemma 3.7(i). Each configuration is averaged over several random seeds.

Run (from the repo root):
    uv run python scripts/run_inexact_pmd.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from avg_reward_pmd.mdp import three_state_mdp
from avg_reward_pmd.pmd import differential_q, run_pmd
from avg_reward_pmd.plotting import inexact_sweep_figure
from avg_reward_pmd.schedules import geometric

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"

HORIZON = 500
GAMMA = 1.05  # above chi_bar / (chi_bar - 1) ~ 1.003 for the unique case
N_SEEDS = 100

ETA0_VALUES = (1e-2, 1e-1, 1.0)
EPSQ_VALUES = (1e-3, 1e-2, 1e-1)
ETA0_FIXED = 1e-1
EPSQ_FIXED = 1e-2
TIE_BREAKER_EPS = 1e-4  # below smallest noise scale so all epsilon_q regimes are active

PI_INIT = np.array(
    [
        [0.999, 0.001],
        [0.001, 0.999],
        [0.010, 0.990],
    ]
)


def reference_optimal_policy(mdp) -> tuple[np.ndarray, float]:
    """Unique-minimizer reference under tie-broken costs."""
    pi_star = np.zeros((mdp.n_states, mdp.n_actions))
    pi_star[0, 1] = 1.0
    pi_star[1, 0] = 1.0
    pi_star[2, 0] = 1.0
    _, rho_star = differential_q(mdp, pi_star)
    return pi_star, rho_star


def mean_trajectory(mdp, pi_star, rho_star, eta0: float, epsq: float) -> np.ndarray:
    """Mean Delta_t over N_SEEDS independent runs at (eta0, epsq)."""
    schedule = geometric(eta0=eta0, gamma=GAMMA, horizon=HORIZON)
    trajs = np.empty((N_SEEDS, HORIZON + 1))
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        result = run_pmd(
            mdp,
            PI_INIT,
            schedule,
            pi_star=pi_star,
            rho_star=rho_star,
            epsilon_q=epsq,
            rng=rng,
        )
        # Average-cost objective can dip below rho_star when noise flips the
        # policy's action preference; we report the absolute gap, which is the
        # quantity controlled by Thm 4.2's floor.
        trajs[seed] = np.abs(result.delta)
    return trajs.mean(axis=0)


def main() -> None:
    mdp = three_state_mdp(p=0.05, unique_optimum=True, eps=TIE_BREAKER_EPS)
    pi_star, rho_star = reference_optimal_policy(mdp)

    by_eta0 = {
        eta0: mean_trajectory(mdp, pi_star, rho_star, eta0, EPSQ_FIXED)
        for eta0 in ETA0_VALUES
    }
    by_epsq = {
        eps: mean_trajectory(mdp, pi_star, rho_star, ETA0_FIXED, eps)
        for eps in EPSQ_VALUES
    }

    time_axis = np.arange(HORIZON + 1)
    inexact_sweep_figure(
        by_eta0=by_eta0,
        by_epsq=by_epsq,
        time_axis=time_axis,
        fixed_eta0=ETA0_FIXED,
        fixed_epsq=EPSQ_FIXED,
        out_path=FIG_DIR / "pmd_inexact_sweep.pdf",
    )

    print("(a) eta_0 sweep at eps_Q =", EPSQ_FIXED)
    for eta0, traj in sorted(by_eta0.items()):
        print(f"   eta_0={eta0:<6g}  floor (mean of last 10%) = {traj[-8:].mean():.3e}")
    print("(b) eps_Q sweep at eta_0 =", ETA0_FIXED)
    for eps, traj in sorted(by_epsq.items()):
        print(f"   eps_Q={eps:<6g}  floor (mean of last 10%) = {traj[-8:].mean():.3e}")


if __name__ == "__main__":
    main()
