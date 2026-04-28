"""Exact KL-PMD for average-cost MDPs.

Implements the update
    pi^{t+1}(a|s) \\propto pi^t(a|s) * exp(-eta_t * Q^{pi^t}(s, a)),
where ``Q^pi`` is the differential action-value function obtained from the
Poisson equation.

All policy arithmetic is done in log-space so that aggressive step sizes do
not lead to underflow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

from avg_reward_pmd.mdp import MDP


def _policy_kernel(mdp: MDP, pi: np.ndarray) -> np.ndarray:
    """State-to-state kernel P_pi(s' | s) induced by policy ``pi``."""
    return np.einsum("sa,sat->st", pi, mdp.P)


def stationary_distribution(P_pi: np.ndarray) -> np.ndarray:
    """Stationary distribution of an irreducible aperiodic kernel.

    Computed as the dominant left eigenvector of ``P_pi``.
    """
    n = P_pi.shape[0]
    # Solve d (I - P_pi) = 0 under sum-to-one by stacking the constraint row.
    A = np.vstack([(np.eye(n) - P_pi).T, np.ones((1, n))])
    b = np.concatenate([np.zeros(n), [1.0]])
    d, *_ = np.linalg.lstsq(A, b, rcond=None)
    d = np.clip(d, 0.0, None)
    return d / d.sum()


def average_cost(mdp: MDP, pi: np.ndarray) -> float:
    """Long-run average cost rho(pi) = sum_s d^pi(s) <r(s, .), pi(.|s)>."""
    d = stationary_distribution(_policy_kernel(mdp, pi))
    return float((d * (pi * mdp.r).sum(axis=-1)).sum())


def differential_q(mdp: MDP, pi: np.ndarray, anchor_state: int = 0) -> tuple[np.ndarray, float]:
    """Differential action-value function Q^pi and average cost rho(pi).

    Solves the Poisson equation (I - P_pi) h = r_pi - rho * 1 with the anchor
    ``h(anchor_state) = 0``, then forms Q(s, a) = r(s, a) - rho + P(s, a)^T h.
    """
    P_pi = _policy_kernel(mdp, pi)
    r_pi = (pi * mdp.r).sum(axis=-1)
    d = stationary_distribution(P_pi)
    rho = float(d @ r_pi)

    n = mdp.n_states
    anchor_row = np.zeros(n)
    anchor_row[anchor_state] = 1.0
    A = np.vstack([np.eye(n) - P_pi, anchor_row])
    b = np.concatenate([r_pi - rho, [0.0]])
    h, *_ = np.linalg.lstsq(A, b, rcond=None)

    q = mdp.r - rho + mdp.P @ h
    return q, rho


def kl_pmd_step(log_pi: np.ndarray, q: np.ndarray, eta: float) -> np.ndarray:
    """One KL-PMD step performed entirely in log-space.

    Returns the updated ``log_pi`` with rows that log-sum-exp to 0.
    """
    unnormalized = log_pi - eta * q
    normalizer = logsumexp(unnormalized, axis=-1, keepdims=True)
    return unnormalized - normalizer


def weighted_kl(p: np.ndarray, q: np.ndarray, d: np.ndarray) -> float:
    """State-weighted KL divergence sum_s d(s) * KL(p(.|s) || q(.|s)).

    Uses a small floor to guard against log(0) when ``p`` is deterministic.
    """
    q_clip = np.clip(q, 1e-300, None)
    p_clip = np.clip(p, 1e-300, None)
    per_state = np.sum(p_clip * (np.log(p_clip) - np.log(q_clip)), axis=-1)
    return float(d @ per_state)


@dataclass
class RunResult:
    """History produced by :func:`run_pmd`."""

    eta: np.ndarray
    rho: np.ndarray
    delta: np.ndarray
    chi: np.ndarray
    policies: list[np.ndarray] = field(default_factory=list)

    @property
    def horizon(self) -> int:
        return len(self.rho)


def run_pmd(
    mdp: MDP,
    pi0: np.ndarray,
    eta_schedule: np.ndarray,
    pi_star: np.ndarray,
    rho_star: float,
    *,
    epsilon_q: float = 0.0,
    rng: np.random.Generator | None = None,
    store_policies: bool = False,
) -> RunResult:
    """Run KL-PMD and collect diagnostics at every iterate.

    Parameters
    ----------
    mdp : MDP
    pi0 : ndarray, shape (S, A)
        Initial stochastic policy (rows sum to 1).
    eta_schedule : ndarray, shape (T,)
        Step sizes ``eta_0, ..., eta_{T-1}``.
    pi_star : ndarray, shape (S, A)
        Reference optimal policy used to compute chi_t = ||d^{pi_star} / d^t||_inf.
    rho_star : float
        Optimal average cost.
    epsilon_q : float
        Sup-norm critic error. When positive, each update replaces the exact
        ``Q^{pi^t}`` with ``Q^{pi^t} + xi^t`` where ``xi^t`` is sampled
        uniformly on ``[-epsilon_q, epsilon_q]^{S x A}``; this realizes
        Assumption 4.1 (uniform critic error) with bound ``epsilon_q``.
    rng : np.random.Generator
        Required when ``epsilon_q > 0``.
    store_policies : bool
        If True, keep the policy sequence in memory.

    Returns
    -------
    RunResult
    """
    if epsilon_q > 0 and rng is None:
        raise ValueError("rng must be provided when epsilon_q > 0")

    T = len(eta_schedule)
    log_pi = np.log(pi0 + 1e-300)
    log_pi -= logsumexp(log_pi, axis=-1, keepdims=True)

    d_star = stationary_distribution(_policy_kernel(mdp, pi_star))

    rho = np.empty(T + 1)
    delta = np.empty(T + 1)
    chi = np.empty(T + 1)
    policies: list[np.ndarray] = []

    pi = np.exp(log_pi)
    d_pi = stationary_distribution(_policy_kernel(mdp, pi))
    rho[0] = average_cost(mdp, pi)
    delta[0] = rho[0] - rho_star
    chi[0] = float(np.max(d_star / np.clip(d_pi, 1e-300, None)))
    if store_policies:
        policies.append(pi.copy())

    for t, eta in enumerate(eta_schedule):
        q, _ = differential_q(mdp, pi)
        if epsilon_q > 0:
            q = q + rng.uniform(-epsilon_q, epsilon_q, size=q.shape)
        log_pi = kl_pmd_step(log_pi, q, float(eta))
        pi = np.exp(log_pi)

        d_pi = stationary_distribution(_policy_kernel(mdp, pi))
        rho[t + 1] = average_cost(mdp, pi)
        delta[t + 1] = rho[t + 1] - rho_star
        chi[t + 1] = float(np.max(d_star / np.clip(d_pi, 1e-300, None)))
        if store_policies:
            policies.append(pi.copy())

    return RunResult(
        eta=np.asarray(eta_schedule, dtype=float),
        rho=rho,
        delta=delta,
        chi=chi,
        policies=policies,
    )
