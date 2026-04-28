"""Paper-ready matplotlib figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from avg_reward_pmd.metrics import tail_chi
from avg_reward_pmd.pmd import RunResult

# Okabe-Ito colorblind-safe palette.
_C_DATA = "#0072B2"  # blue
_C_REF1 = "#D55E00"  # vermilion (worst-case bound)
_C_REF2 = "#009E73"  # bluish green (asymptotic reference)

_RC = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIX", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "lines.linewidth": 1.3,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# NeurIPS 2026 text width is 5.5 in; figures are rendered at ``\linewidth`` so
# the matplotlib figsize must match to avoid LaTeX-side font scaling.
_FIGSIZE = (5.5, 2.1)


def _apply_style() -> None:
    plt.rcParams.update(_RC)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.22,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _truncate_to_convergence(
    delta: np.ndarray, floor: float = 1e-12, tail_pad: int = 5
) -> int:
    """Return the last iteration index to display.

    Cuts the trajectory ``tail_pad`` iterations after it first crosses
    ``floor``, so the plot does not stretch across empty machine-precision
    territory.
    """
    below = np.where(delta <= floor)[0]
    if below.size == 0:
        return len(delta)
    return min(len(delta), int(below[0]) + tail_pad)


def inexact_sweep_figure(
    by_eta0: dict[float, np.ndarray],
    by_epsq: dict[float, np.ndarray],
    time_axis: np.ndarray,
    fixed_eta0: float,
    fixed_epsq: float,
    out_path: Path,
) -> None:
    """Two-panel figure illustrating Theorem 4.2's floor structure.

    Panel (a): step-size-independence. Fix ``epsilon_q = fixed_epsq``, sweep
    ``eta_0`` across ``by_eta0`` keys. Trajectories at different step sizes
    converge at different rates to essentially the same floor, empirically
    realizing the decoupling described in Section 3.1.

    Panel (b): floor depends on critic error. Fix ``eta_0 = fixed_eta0``,
    sweep ``epsilon_q`` across ``by_epsq`` keys. Different noise magnitudes
    give monotonically ordered floors.
    """
    _apply_style()

    fig, (ax_eta, ax_eps) = plt.subplots(1, 2, figsize=_FIGSIZE)

    linestyles = ["-", "--", ":"]
    colors = [_C_DATA, _C_REF1, _C_REF2]

    for idx, (eta0, traj) in enumerate(sorted(by_eta0.items())):
        ax_eta.semilogy(
            time_axis,
            np.clip(traj, 1e-13, None),
            color=_C_DATA,
            linestyle=linestyles[idx % len(linestyles)],
            label=rf"$\eta_0={eta0:g}$",
        )
    ax_eta.set_xlabel(r"Iteration $t$")
    ax_eta.set_ylabel(r"$\Delta_t = \rho(\pi^t) - \rho^\star$")
    ax_eta.set_title(rf"Fixed $\varepsilon_Q={fixed_epsq:g}$", fontsize=9)
    ax_eta.legend(loc="upper right", framealpha=0.92, handlelength=2.2)
    _panel_label(ax_eta, "(a)")

    for idx, (eps, traj) in enumerate(sorted(by_epsq.items())):
        ax_eps.semilogy(
            time_axis,
            np.clip(traj, 1e-13, None),
            color=colors[idx % len(colors)],
            label=rf"$\varepsilon_Q={eps:g}$",
        )
    ax_eps.set_xlabel(r"Iteration $t$")
    ax_eps.set_ylabel(r"$\Delta_t = \rho(\pi^t) - \rho^\star$")
    ax_eps.set_title(rf"Fixed $\eta_0={fixed_eta0:g}$", fontsize=9)
    ax_eps.legend(loc="upper right", framealpha=0.92, handlelength=2.2)
    _panel_label(ax_eps, "(b)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def rate_and_chi_figure(
    result: RunResult,
    out_path: Path,
    psi_0: float | None = None,
) -> None:
    """Two-panel diagnostic: suboptimality (left) and chi_t trajectory (right).

    Left panel overlays the theorem's non-asymptotic upper bound
    ``(1 - 1/chi_bar)^t * psi_0`` (Lemma 3.7(i)) and the asymptotic-rate
    reference ``(1 - 1/chi)^t * psi_0`` (Lemma 3.7(ii) / Corollary 4.3).
    If ``psi_0`` is not supplied, ``Delta_0`` is used as a fallback.

    No suptitle is drawn; use the enclosing LaTeX ``\\caption{...}`` in the
    paper to describe the figure.
    """
    _apply_style()

    chi_bar = float(np.max(result.chi))
    chi_hat = tail_chi(result.chi)

    delta_raw = result.delta
    t_full = np.arange(len(delta_raw))
    t_end = _truncate_to_convergence(delta_raw, floor=1e-12, tail_pad=5)
    t = t_full[:t_end]
    delta = np.clip(delta_raw[:t_end], 1e-13, None)
    chi = result.chi[:t_end]
    c0 = float(psi_0) if psi_0 is not None else float(delta[0])

    fig, (ax_delta, ax_chi) = plt.subplots(1, 2, figsize=_FIGSIZE)

    ax_delta.semilogy(t, delta, color=_C_DATA, label=r"$\Delta_t$")
    if chi_bar > 1.0:
        ax_delta.semilogy(
            t,
            c0 * (1 - 1 / chi_bar) ** t,
            color=_C_REF1,
            linestyle="--",
            label=rf"$(1-1/\bar\chi)^t\,\Psi_0,\;\bar\chi\approx{chi_bar:.1f}$",
        )
    if chi_hat > 1.0:
        ax_delta.semilogy(
            t,
            c0 * (1 - 1 / chi_hat) ** t,
            color=_C_REF2,
            linestyle=":",
            label=rf"$(1-1/\chi)^t\,\Psi_0,\;\chi\approx{chi_hat:.2f}$",
        )
    ax_delta.set_xlabel(r"Iteration $t$")
    ax_delta.set_ylabel(r"$\Delta_t = \rho(\pi^t) - \rho^\star$")
    ax_delta.set_ylim(bottom=1e-13)
    ax_delta.legend(loc="lower left", framealpha=0.92, handlelength=2.2)
    _panel_label(ax_delta, "(a)")

    ax_chi.semilogy(t, chi, color=_C_DATA, label=r"$\chi_t$")
    ax_chi.axhline(
        chi_bar,
        color=_C_REF1,
        linestyle="--",
        label=rf"$\bar\chi\approx{chi_bar:.1f}$",
    )
    ax_chi.axhline(
        chi_hat,
        color=_C_REF2,
        linestyle=":",
        label=rf"$\chi\approx{chi_hat:.2f}$",
    )
    ax_chi.set_xlabel(r"Iteration $t$")
    ax_chi.set_ylabel(r"$\chi_t = \|d^{\pi^\star}/d^{\pi^t}\|_\infty$")
    ax_chi.legend(loc="center right", framealpha=0.92, handlelength=2.2)
    _panel_label(ax_chi, "(b)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
