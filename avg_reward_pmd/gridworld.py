"""Slip-prone GridWorld with reward-and-teleport (canonical avg-reward setup).

A standard 5x5 GridWorld in cost-minimization form. The agent picks one of
four cardinal moves (N, E, S, W). With probability ``1 - slip`` the move is
executed as intended; with probability ``slip``, the agent moves uniformly at
random over the four cardinal directions instead. Any move that would leave
the grid keeps the agent in place. A unit cost is incurred on every step
*except* at the goal state (top-right corner by default), where the cost is
zero and the next state is the start (bottom-left corner) -- giving an
ergodic chain whose average cost is minimised by short paths to the goal.

State indexing: ``state_id = row * grid_size + col``.
Action indexing: 0 = N (row -1), 1 = E (col +1), 2 = S (row +1), 3 = W (col -1).
"""

from __future__ import annotations

import numpy as np

from avg_reward_pmd.mdp import MDP

_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W


def gridworld_mdp(
    grid_size: int = 5,
    slip: float = 0.1,
    start: tuple[int, int] = (4, 0),
    goal: tuple[int, int] = (0, 4),
) -> MDP:
    """Build a slip-prone GridWorld in cost-minimization form."""
    n_states = grid_size * grid_size
    n_actions = 4
    start_id = start[0] * grid_size + start[1]
    goal_id = goal[0] * grid_size + goal[1]

    def step(r: int, c: int, dr: int, dc: int) -> int:
        """Apply intended move; bounce off walls."""
        nr = max(0, min(grid_size - 1, r + dr))
        nc = max(0, min(grid_size - 1, c + dc))
        return nr * grid_size + nc

    P = np.zeros((n_states, n_actions, n_states))
    r = np.zeros((n_states, n_actions))

    for s in range(n_states):
        if s == goal_id:
            for a in range(n_actions):
                P[s, a, start_id] = 1.0
                r[s, a] = 0.0
            continue

        row, col = divmod(s, grid_size)
        for a in range(n_actions):
            for b, (dr, dc) in enumerate(_DIRS):
                prob = (1.0 - slip if b == a else 0.0) + slip / n_actions
                P[s, a, step(row, col, dr, dc)] += prob
            r[s, a] = 1.0

    return MDP(P=P, r=r)
