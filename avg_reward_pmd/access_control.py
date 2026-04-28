"""Access-control queuing MDP (Sutton & Barto, Example 10.2).

A canonical average-reward benchmark. ``n_servers`` identical servers handle
customers arriving one per step with priority drawn uniformly from
``priorities``. The agent observes the joint state ``(n_free, priority)`` and
chooses to accept (serve) or reject the customer. Each busy server completes
(becomes free) independently with probability ``p_free`` per step. Accepting
pays a reward equal to the priority (only if ``n_free > 0``); rejecting pays
zero. We convert to the cost convention used elsewhere in this package by
negating, then shifting so costs are nonnegative.

State indexing: ``state_id = n_free * len(priorities) + prio_idx``, giving
``(n_servers + 1) * len(priorities)`` states.
Action indexing: 0 = reject, 1 = accept.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom

from avg_reward_pmd.mdp import MDP

DEFAULT_PRIORITIES: tuple[int, ...] = (1, 2, 4, 8)


def access_control_mdp(
    n_servers: int = 10,
    priorities: tuple[int, ...] = DEFAULT_PRIORITIES,
    p_free: float = 0.06,
) -> MDP:
    """Build the access-control queuing MDP in cost-minimization form.

    ``n_servers = 10``, ``priorities = (1, 2, 4, 8)``, ``p_free = 0.06``
    reproduces the standard instance.
    """
    n_prio = len(priorities)
    n_states = (n_servers + 1) * n_prio
    n_actions = 2  # 0 = reject, 1 = accept

    # Precompute Binomial(B, p_free) pmfs for B = 0, ..., n_servers.
    pmfs = np.zeros((n_servers + 1, n_servers + 1))
    for B in range(n_servers + 1):
        pmfs[B, : B + 1] = binom.pmf(np.arange(B + 1), B, p_free)

    P = np.zeros((n_states, n_actions, n_states))
    r = np.zeros((n_states, n_actions))

    for n_free in range(n_servers + 1):
        for prio_idx, prio in enumerate(priorities):
            s = n_free * n_prio + prio_idx
            B = n_servers - n_free  # number of busy servers

            for a in (0, 1):
                # Accept only effective if n_free > 0.
                effective_accept = (a == 1) and (n_free > 0)
                if effective_accept:
                    n_free_after_decision = n_free - 1
                    reward = float(prio)
                else:
                    n_free_after_decision = n_free
                    reward = 0.0

                # k busy servers finish, so B' = B + (0 or 1 fewer if accepted) - k.
                # Equivalently: new n_free = n_free_after_decision + k.
                B_after_decision = n_servers - n_free_after_decision
                for k in range(B_after_decision + 1):
                    n_free_next = n_free_after_decision + k
                    prob = pmfs[B_after_decision, k]
                    # Next priority is uniform.
                    for prio_next_idx in range(n_prio):
                        s_next = n_free_next * n_prio + prio_next_idx
                        P[s, a, s_next] += prob / n_prio

                # Cost convention: negate reward (then shift at the end).
                r[s, a] = -reward

    # Shift costs to be nonnegative (doesn't change the optimal policy or chi_t).
    r = r - r.min()

    return MDP(P=P, r=r)
