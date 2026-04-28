"""Random MDP sampler and exact avg-reward policy-iteration solver.

Used by the rate-gap ensemble experiment to sample many MDPs, solve for the
optimal policy, and feed the result to KL-PMD.
"""

from __future__ import annotations

import numpy as np

from avg_reward_pmd.mdp import MDP
from avg_reward_pmd.pmd import _policy_kernel, differential_q


def sample_random_mdp(
    rng: np.random.Generator,
    n_states: int,
    n_actions: int,
    dirichlet_alpha: float = 1.0,
) -> MDP:
    """Sample an ergodic random MDP.

    Transitions: each ``P(s, a, :)`` is drawn from Dirichlet(``alpha``), giving
    full-support rows (so every induced Markov kernel is irreducible and
    aperiodic --- Assumption 3.1 holds by construction). Costs: i.i.d.
    uniform on ``[0, 1]``.
    """
    P = rng.dirichlet(np.ones(n_states) * dirichlet_alpha, size=(n_states, n_actions))
    r = rng.uniform(0.0, 1.0, size=(n_states, n_actions))
    return MDP(P=P, r=r)


def solve_optimal_policy(
    mdp: MDP, max_iter: int = 500
) -> tuple[np.ndarray, float]:
    """Policy iteration for the average-cost MDP.

    Returns a deterministic optimal policy ``pi_star`` (one-hot per state) and
    its average cost ``rho_star``.

    Uses greedy-on-differential-Q improvement until the greedy action pattern
    stabilises. Converges in finitely many steps for a finite MDP; ``max_iter``
    is a safety cap.
    """
    pi = np.full((mdp.n_states, mdp.n_actions), 1.0 / mdp.n_actions)
    best_actions = np.full(mdp.n_states, -1)

    for _ in range(max_iter):
        q, _ = differential_q(mdp, pi)
        new_actions = np.argmin(q, axis=-1)
        if np.array_equal(new_actions, best_actions):
            break
        best_actions = new_actions
        pi = np.zeros_like(pi)
        pi[np.arange(mdp.n_states), best_actions] = 1.0

    _, rho_star = differential_q(mdp, pi)
    return pi, float(rho_star)


def worst_cost_deterministic_policy(mdp: MDP) -> np.ndarray:
    """Deterministic policy that picks the most-costly action at each state.

    Used as an adversarial initialisation: forces a large transient distance
    to the optimum, which is what produces a sizeable ``chi_bar`` in the
    ensemble experiment.
    """
    pi = np.zeros((mdp.n_states, mdp.n_actions))
    worst = np.argmax(mdp.r, axis=-1)
    pi[np.arange(mdp.n_states), worst] = 1.0
    return pi


def stationary_distribution_of_policy(mdp: MDP, pi: np.ndarray) -> np.ndarray:
    """Convenience wrapper: stationary distribution under policy ``pi``."""
    from avg_reward_pmd.pmd import stationary_distribution

    return stationary_distribution(_policy_kernel(mdp, pi))
