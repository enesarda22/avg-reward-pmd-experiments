# Numerical experiments — Average-Reward PMD

Reproduces the figures in the paper *"Sharp Convergence and Sample Complexity
of Policy Mirror Descent for Average-Reward MDPs"*.

## Quick start

```bash
uv sync                                    # install deps (creates .venv/)
uv run python scripts/run_exact_pmd.py     # F.1, F.2 (3-state exact PMD)
uv run python scripts/run_inexact_pmd.py   # F.3 (3-state inexact PMD sweep)

# Rate-gap experiment (Figure 1 in the body, Figure F.4 in the appendix):
uv run python scripts/run_rate_gap_capped.py            # random Dirichlet MDPs
uv run python scripts/run_rate_gap_queuing_capped.py    # access-control queuing
uv run python scripts/run_rate_gap_gridworld_capped.py  # 5x5 GridWorld
uv run python scripts/run_rate_gap_jcr_capped.py        # Jack's car rental

uv run python scripts/plot_rate_gap_combined_capped.py  # body Figure 1
uv run python scripts/plot_rate_gap_appendix_panels.py  # appendix Figure F.4
```

All outputs are written to `figures/` (auto-created; gitignored). The
`run_rate_gap_*_capped.py` scripts each emit a `rate_gap_*_data_capped.json`
file with the per-run `chi` and `delta` trajectories; the two
`plot_rate_gap_*` scripts read those JSONs and produce the body / appendix
figures.

## Figure ↔ script map

| Paper figure | Producing script(s) |
|---|---|
| Body Fig. 1     | `run_rate_gap_*_capped.py` (4 scripts) → `plot_rate_gap_combined_capped.py` |
| App. Fig. F.1, F.2 | `run_exact_pmd.py` |
| App. Fig. F.3   | `run_inexact_pmd.py` |
| App. Fig. F.4   | `run_rate_gap_*_capped.py` (4 scripts) → `plot_rate_gap_appendix_panels.py` |

## Step-size schedules

Two schedules are used across the experiments:

- **Mixed super-geometric** $\eta_t = \eta_0\, c^{\,t}\, \alpha^{\,t^2}$ with
  ratio $c\,\alpha^{2t-1}$. Bounded below by $c$ for all $t \ge 0$ (so the
  non-asymptotic floor of Lemma 3.7(i) holds with $\bar\chi/(\bar\chi-1) \le c$)
  and divergent as $t \to \infty$ (so the asymptotic hypothesis of
  Lemma 3.7(ii) holds). Used by the 3-state exact-PMD experiments.
- **Capped super-geometric** $\eta_t = \min(\eta_0\, c^{\,t}\, \alpha^{\,t^2},
  \eta_{\mathrm{cap}})$. Same structure, capped to prevent floating-point
  overflow once $\eta_t$ exceeds the inverse $Q$-range. Used by the rate-gap
  experiments with $(\eta_0, c, \alpha, \eta_{\mathrm{cap}}) = (10, 1.1, 1.001, 100)$.

Both are implemented in `avg_reward_pmd/schedules.py`.

## Package layout

```
avg_reward_pmd/
  mdp.py                 # MDP dataclass
  pmd.py                 # stationary distribution, differential Q, KL-PMD step, run loop
  schedules.py           # step-size schedules
  metrics.py             # tail-chi, empirical-rate helpers
  plotting.py            # shared plot styling and the F.1/F.2 (a)(b) panels
  random_mdp.py          # Dirichlet-sampled random MDPs and oracle solver
  gridworld.py           # 5x5 slip-prone GridWorld
  jacks_car_rental.py    # Jack's Car Rental (Sutton & Barto, Example 4.2) via gym-classics
  access_control.py      # Access-control queuing (Sutton & Barto, Example 10.2)
scripts/
  run_exact_pmd.py                  # F.1, F.2
  run_inexact_pmd.py                # F.3
  run_rate_gap_capped.py            # random Dirichlet rate-gap data
  run_rate_gap_queuing_capped.py    # queuing rate-gap data
  run_rate_gap_gridworld_capped.py  # gridworld rate-gap data
  run_rate_gap_jcr_capped.py        # JCR rate-gap data
  plot_rate_gap_combined_capped.py  # body Figure 1
  plot_rate_gap_appendix_panels.py  # appendix Figure F.4
```

## Dependencies

- Python `>=3.11`
- `numpy`, `scipy`, `matplotlib`
- `gym`, `gymnasium`, `gym-classics` (the latter provides the
  `JacksCarRental-v0` environment)

Locked in `uv.lock`; `uv sync` is the canonical install path.

## Reproducibility notes

- All experiments are deterministic given the random seeds used internally
  by the run scripts (`np.random.default_rng(seed + offset)`). Re-running any
  script overwrites the corresponding output in `figures/`; no other state
  is shared.
- The matplotlib plots use `text.usetex=True`, so a working LaTeX install is
  required to regenerate the figures (the data scripts do not).
