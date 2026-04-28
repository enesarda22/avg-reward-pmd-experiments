"""Jack's Car Rental (Sutton & Barto Example 4.2) as an avg-reward MDP.

Two locations, up to 20 cars each (441 states), and 11 overnight-move actions
(-5 to +5 cars). Daily Poisson rentals (lambda=3, 4) and returns
(lambda=3, 2) make every transition full-support, so every policy --
including the deterministic optimum -- induces an ergodic chain. This is the
key reason this benchmark works for our chi_t experiment without any
ergodicity smoothing.

We assemble the explicit (P, r) matrices from gym-classics' ``model()``
interface and convert to the cost-minimization convention used elsewhere in
this package.
"""

from __future__ import annotations

import numpy as np

from avg_reward_pmd.mdp import MDP


def jacks_car_rental_mdp(env_id: str = "JacksCarRental-v0") -> MDP:
    """Build the avg-cost MDP from gym-classics' Jack's Car Rental env."""
    import gym_classics
    import gym
    gym_classics.register("gym")
    env = gym.make(env_id).unwrapped

    states = list(env.states())
    actions = list(env.actions())
    n_states = len(states)
    n_actions = len(actions)

    P = np.zeros((n_states, n_actions, n_states))
    reward = np.zeros((n_states, n_actions))

    for s_idx, s in enumerate(states):
        for a_idx, a in enumerate(actions):
            ns_arr, r_arr, _done, p_arr = env.model(s, a)
            # Aggregate per-(s, a) transition kernel and expected reward.
            for ns, r, p in zip(ns_arr, r_arr, p_arr):
                P[s_idx, a_idx, ns] += p
                reward[s_idx, a_idx] += p * r

    # Sanity: rows sum to 1.
    P = P / P.sum(axis=-1, keepdims=True)

    # Cost form: negate, shift to nonnegative.
    cost = -reward
    cost = cost - cost.min()
    return MDP(P=P, r=cost)
