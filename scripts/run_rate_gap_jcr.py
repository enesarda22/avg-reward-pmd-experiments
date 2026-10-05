"""Jack's Car Rental rate-gap experiment (paper, Appendix F.4), with the same
mixed super-geometric schedule as ``run_rate_gap.py``. Uses the gym-classics
``JacksCarRental-v0`` env (S&B Example 4.2): 441 states, 11 actions, daily
Poisson rentals/returns make the chain naturally ergodic.

Output: ``figures/rate_gap_jcr_data.json``.

Run:
    uv run python scripts/run_rate_gap_jcr.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from avg_reward_pmd.jacks_car_rental import jacks_car_rental_mdp
from avg_reward_pmd.metrics import tail_chi
from avg_reward_pmd.pmd import run_pmd, stationary_distribution, weighted_kl
from avg_reward_pmd.random_mdp import solve_optimal_policy
from avg_reward_pmd.schedules import super_geometric_with_floor

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "figures" / "rate_gap_jcr_data.json"
)

DEFAULT_N_INITS: int = 80
DEFAULT_HORIZON: int = 60
DIRICHLET_ALPHA: float = 0.5

ETA0: float = 10.0
C: float = 1.1
ALPHA: float = 1.001
ETA_MINUS_1: float = ETA0 / C


def _schedule(horizon: int) -> np.ndarray:
    return super_geometric_with_floor(eta0=ETA0, c=C, alpha=ALPHA, horizon=horizon)


def random_initial_policy(rng, n_states, n_actions):
    return rng.dirichlet(np.ones(n_actions) * DIRICHLET_ALPHA, size=n_states)


def _run_one(mdp, pi_star, rho_star, pi_init, horizon, seed):
    schedule = _schedule(horizon)
    result = run_pmd(mdp=mdp, pi0=pi_init, eta_schedule=schedule,
                    pi_star=pi_star, rho_star=rho_star)
    chi_bar = float(np.max(result.chi))
    chi_hat = tail_chi(result.chi, tail_fraction=0.2)
    delta_0 = float(result.delta[0])
    d_star = stationary_distribution(np.einsum('sa,san->sn', pi_star, mdp.P))
    d_0_star = float(weighted_kl(pi_star, pi_init, d_star))
    psi_0 = delta_0 + d_0_star / (ETA_MINUS_1 * chi_bar)
    return {
        "tag": "random", "seed": seed, "horizon": horizon,
        "eta0": ETA0, "c": C, "alpha": ALPHA,
        "chi_bar": chi_bar, "chi_tail": chi_hat,
        "ratio_bar_over_tail": chi_bar / max(chi_hat, 1e-300),
        "delta_0": delta_0,
        "delta_final": float(result.delta[-1]),
        "d_0_star": d_0_star,
        "psi_0": float(psi_0),
        "chi_traj": result.chi.tolist(),
        "delta_traj": result.delta.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-inits", type=int, default=DEFAULT_N_INITS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    args = parser.parse_args()

    print("Building Jack's Car Rental MDP (~2s)...")
    mdp = jacks_car_rental_mdp()
    print(f"  n_states={mdp.n_states}, n_actions={mdp.n_actions}")
    pi_star, rho_star = solve_optimal_policy(mdp)
    print(f"  rho_star={rho_star:.4f}")

    records: list[dict] = []
    t_start = time.time()

    for seed in range(args.n_inits):
        rng = np.random.default_rng(4000 + seed)  # disjoint seed range
        pi0 = random_initial_policy(rng, mdp.n_states, mdp.n_actions)
        records.append(_run_one(mdp, pi_star, rho_star, pi0,
                                horizon=args.horizon, seed=seed))
        if (seed + 1) % 10 == 0:
            r = records[-1]
            print(f"[{seed+1}] elapsed {time.time()-t_start:.1f}s "
                  f"chi_bar={r['chi_bar']:.2f} "
                  f"chi_tail={r['chi_tail']:.2f} "
                  f"ratio={r['ratio_bar_over_tail']:.1f}")

    payload = {
        "config": {
            "env": "JacksCarRental-v0 (gym-classics; S&B Example 4.2)",
            "n_inits": args.n_inits,
            "horizon": args.horizon,
            "eta0": ETA0, "c": C, "alpha": ALPHA,
            "schedule": "super_geometric_with_floor",
            "dirichlet_alpha_init": DIRICHLET_ALPHA,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {len(records)} records to {args.output} "
          f"in {time.time()-t_start:.1f}s.")


if __name__ == "__main__":
    main()
