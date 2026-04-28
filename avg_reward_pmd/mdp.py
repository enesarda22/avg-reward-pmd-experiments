"""MDP definition and the three-state test instance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MDP:
    """A finite average-cost MDP with deterministic cost function.

    Attributes
    ----------
    P : ndarray, shape (S, A, S)
        Transition kernel; ``P[s, a, s']`` is the probability of transitioning
        to state ``s'`` from ``(s, a)``.
    r : ndarray, shape (S, A)
        Per-step cost.
    """

    P: np.ndarray
    r: np.ndarray

    @property
    def n_states(self) -> int:
        return self.r.shape[0]

    @property
    def n_actions(self) -> int:
        return self.r.shape[1]

    def __post_init__(self) -> None:
        if self.P.shape != (self.n_states, self.n_actions, self.n_states):
            raise ValueError(f"P has shape {self.P.shape}; expected (S, A, S)")
        if not np.allclose(self.P.sum(axis=-1), 1.0):
            raise ValueError("Transition rows must sum to 1")


def three_state_mdp(p: float = 0.05, unique_optimum: bool = False, eps: float = 1e-3) -> MDP:
    """Three-state, two-action ergodic average-cost MDP.

    State 2 is the only "costly" state; states 0 and 1 differ only in a
    tie-breaker cost controlled by ``eps``. With ``unique_optimum=False``
    (``eps`` unused) multiple policies are optimal; with ``unique_optimum=True``
    the tie-breaker makes the optimum unique and deterministic.

    Parameters
    ----------
    p : float
        Self-loop / return probability at state 2; also controls the mixing
        rate. Smaller ``p`` means slower mixing.
    unique_optimum : bool
        Whether to add the tie-breaker that makes the optimum unique.
    eps : float
        Tie-breaker magnitude (only used when ``unique_optimum=True``).
    """
    P = np.zeros((3, 2, 3))
    P[0, 0] = (1 - p, 0.0, p)
    P[0, 1] = (0.0, 1 - p, p)
    P[1, 0] = (0.0, 1 - p, p)
    P[1, 1] = (1 - p, 0.0, p)
    P[2, 0] = (0.19, 0.80, 0.01)
    P[2, 1] = (0.95, 0.04, 0.01)

    r = np.zeros((3, 2))
    r[2, 0] = 0.0
    r[2, 1] = 0.2
    if unique_optimum:
        r[0, 0] = eps
        r[1, 1] = eps

    return MDP(P=P, r=r)
