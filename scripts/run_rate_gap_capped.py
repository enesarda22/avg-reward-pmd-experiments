"""Random-MDP rate-gap experiment with a *capped super-geometric* schedule.

Sibling of ``run_rate_gap.py``. Same MDP ensembles and adversarial init, but
the step size follows ``eta_t = min(eta_0 * c^t * alpha^{t^2}, eta_cap)``. The
super-geometric portion (until the cap binds) satisfies the floor hypothesis
of Lemma 3.7(i) (with ``c >= chi_bar / (chi_bar - 1)`` for ``chi_bar`` not too
small) and the divergent-ratio hypothesis of Lemma 3.7(ii); the cap keeps the
schedule numerically sensible for the plotted horizon.

Output: ``figures/rate_gap_data_capped.json`` (does not overwrite the
constant-eta baseline at ``figures/rate_gap_data.json``).

Run:
    uv run python scripts/run_rate_gap_capped.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from avg_reward_pmd.metrics import tail_chi
from avg_reward_pmd.pmd import run_pmd
from avg_reward_pmd.random_mdp import (
    sample_random_mdp,
    solve_optimal_policy,
    worst_cost_deterministic_policy,
)
from avg_reward_pmd.schedules import super_geometric_with_floor

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent.parent / "figures" / "rate_gap_data_capped.json"
)

DEFAULT_CONFIGS: list[tuple[int, int]] = [
    (5, 2),
    (10, 3),
    (20, 4),
]

DEFAULT_N_MDPS: int = 40
DEFAULT_HORIZON: int = 60
DEFAULT_DIRICHLET_ALPHA: float = 0.1

# Capped super-geometric schedule.
ETA0: float = 10.0
C: float = 1.1
ALPHA: float = 1.001
ETA_CAP: float = 100.0


def _capped_schedule(horizon: int) -> np.ndarray:
    return np.minimum(
        super_geometric_with_floor(eta0=ETA0, c=C, alpha=ALPHA, horizon=horizon),
        ETA_CAP,
    )


def _run_one(n_states, n_actions, seed, horizon, dirichlet_alpha):
    rng = np.random.default_rng(seed)
    mdp = sample_random_mdp(rng, n_states=n_states, n_actions=n_actions,
                            dirichlet_alpha=dirichlet_alpha)
    pi_star, rho_star = solve_optimal_policy(mdp)
    pi_init = worst_cost_deterministic_policy(mdp)
    schedule = _capped_schedule(horizon)
    result = run_pmd(mdp=mdp, pi0=pi_init, eta_schedule=schedule,
                    pi_star=pi_star, rho_star=rho_star)
    chi_bar = float(np.max(result.chi))
    chi_hat = tail_chi(result.chi, tail_fraction=0.2)
    return {
        "seed": seed,
        "n_states": n_states,
        "n_actions": n_actions,
        "horizon": horizon,
        "eta0": ETA0, "c": C, "alpha": ALPHA, "eta_cap": ETA_CAP,
        "chi_bar": chi_bar,
        "chi_tail": chi_hat,
        "ratio_bar_over_tail": chi_bar / max(chi_hat, 1e-300),
        "delta_0": float(result.delta[0]),
        "delta_final": float(result.delta[-1]),
        "chi_traj": result.chi.tolist(),
        "delta_traj": result.delta.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-mdps", type=int, default=DEFAULT_N_MDPS)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--dirichlet-alpha", type=float, default=DEFAULT_DIRICHLET_ALPHA)
    args = parser.parse_args()

    records: list[dict] = []
    t_start = time.time()
    seed = 0
    for n_states, n_actions in DEFAULT_CONFIGS:
        for _ in range(args.n_mdps):
            records.append(_run_one(n_states, n_actions, seed, args.horizon,
                                    args.dirichlet_alpha))
            seed += 1
            if seed % 20 == 0:
                r = records[-1]
                print(f"[{seed}] elapsed {time.time()-t_start:.1f}s "
                      f"S={n_states} A={n_actions} chi_bar={r['chi_bar']:.1f} "
                      f"chi_tail={r['chi_tail']:.2f} "
                      f"ratio={r['ratio_bar_over_tail']:.1f}")

    payload = {
        "config": {
            "configs": DEFAULT_CONFIGS,
            "n_mdps_per_config": args.n_mdps,
            "horizon": args.horizon,
            "eta0": ETA0, "c": C, "alpha": ALPHA, "eta_cap": ETA_CAP,
            "schedule": "super_geometric_with_floor capped at eta_cap",
            "dirichlet_alpha": args.dirichlet_alpha,
            "init_policy": "worst-cost deterministic",
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
