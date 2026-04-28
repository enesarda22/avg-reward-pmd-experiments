"""Numerical experiments for average-reward PMD."""

from avg_reward_pmd.mdp import MDP, three_state_mdp
from avg_reward_pmd.pmd import (
    differential_q,
    kl_pmd_step,
    run_pmd,
    stationary_distribution,
    weighted_kl,
)
from avg_reward_pmd.random_mdp import (
    sample_random_mdp,
    solve_optimal_policy,
    worst_cost_deterministic_policy,
)

__all__ = [
    "MDP",
    "three_state_mdp",
    "differential_q",
    "kl_pmd_step",
    "run_pmd",
    "stationary_distribution",
    "weighted_kl",
    "sample_random_mdp",
    "solve_optimal_policy",
    "worst_cost_deterministic_policy",
]
