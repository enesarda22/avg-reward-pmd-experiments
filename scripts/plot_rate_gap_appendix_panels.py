"""Per-ensemble (a)(b) panels for the rate-gap appendix figure.

Reads the four ``rate_gap_*_data_capped.json`` outputs and produces a single
4-row x 2-column figure (one row per MDP family) in the same visual style as
``avg_reward_pmd.plotting.rate_and_chi_figure``: panel (a) shows median
``Delta_t`` with non-asymptotic and asymptotic-rate references; panel (b)
shows median ``chi_t`` with horizontal references at the ensemble-median
``chi_bar`` and ``chi_tail``. IQR over the ensemble is shaded in both.

Run:
    uv run python scripts/plot_rate_gap_appendix_panels.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
RANDOM_INPUT = FIG_DIR / "rate_gap_data_capped.json"
QUEUING_INPUT = FIG_DIR / "rate_gap_queuing_data_capped.json"
GRIDWORLD_INPUT = FIG_DIR / "rate_gap_gridworld_data_capped.json"
JCR_INPUT = FIG_DIR / "rate_gap_jcr_data_capped.json"

# Match the appendix style.
_C_DATA = "#0072B2"  # blue (median data)
_C_REF1 = "#D55E00"  # vermilion (chi_bar / non-asymptotic bound)
_C_REF2 = "#009E73"  # green (chi_tail / asymptotic-rate reference)

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

# Single column wide, 4 rows of ~2.1 in panels.
_FIGSIZE = (5.5, 8.4)

RANDOM_CONFIG_TO_PLOT: tuple[int, int] = (20, 4)


def _load(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _ensemble_arrays(records: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    deltas = np.stack([np.asarray(r["delta_traj"]) for r in records])
    chis = np.stack([np.asarray(r["chi_traj"]) for r in records])
    psi0s = np.array([r["psi_0"] for r in records])
    return deltas, chis, psi0s


def _band(ax, t, traj, color, label):
    med = np.median(traj, axis=0)
    q25 = np.quantile(traj, 0.25, axis=0)
    q75 = np.quantile(traj, 0.75, axis=0)
    ax.fill_between(t, q25, q75, color=color, alpha=0.15, linewidth=0)
    ax.semilogy(t, med, color=color, linewidth=1.5, label=label)
    return med


def _row_label(ax_left, label: str) -> None:
    ax_left.text(
        -0.30,
        1.18,
        label,
        transform=ax_left.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _panel_label(ax, label: str) -> None:
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


def _plot_one_row(
    ax_delta, ax_chi, deltas: np.ndarray, chis: np.ndarray, psi0s: np.ndarray, ensemble_name: str
) -> None:
    t = np.arange(deltas.shape[1])

    # Per-run summary stats for the references.
    chi_bars = chis.max(axis=1)
    n_tail = max(1, int(0.2 * chis.shape[1]))
    chi_tails = np.median(chis[:, -n_tail:], axis=1)
    chi_bar_med = float(np.median(chi_bars))
    chi_tail_med = float(np.median(chi_tails))
    psi0_med = float(np.median(psi0s))

    # --- Panel (a): Delta_t ---
    deltas_clipped = np.clip(deltas, 1e-15, None)
    delta_floor = max(float(deltas_clipped.min()) / 10, 1e-15)
    delta_ceiling = float(deltas_clipped.max()) * 5
    _band(ax_delta, t, deltas_clipped, _C_DATA, r"median $\Delta_t$")
    if chi_bar_med > 1.0:
        ref_bar = np.clip(psi0_med * (1 - 1 / chi_bar_med) ** t, delta_floor, None)
        ax_delta.semilogy(
            t, ref_bar, color=_C_REF1, linestyle="--",
            label=rf"$(1-1/\bar\chi)^t\,\Psi_0,\;\bar\chi\approx{chi_bar_med:.1f}$",
        )
    if chi_tail_med > 1.05:  # skip when essentially superlinear
        ref_tail = np.clip(
            psi0_med * (1 - 1 / chi_tail_med) ** t, delta_floor, None
        )
        ax_delta.semilogy(
            t, ref_tail, color=_C_REF2, linestyle=":",
            label=rf"$(1-1/\chi)^t\,\Psi_0,\;\chi\approx{chi_tail_med:.2f}$",
        )
    ax_delta.set_xlabel(r"Iteration $t$")
    ax_delta.set_ylabel(r"$\Delta_t = \rho(\pi^t) - \rho^\star$")
    ax_delta.set_ylim(bottom=delta_floor, top=delta_ceiling)
    ax_delta.legend(loc="lower left", framealpha=0.92, handlelength=2.2)
    _panel_label(ax_delta, "(a)")

    # --- Panel (b): chi_t ---
    _band(ax_chi, t, chis, _C_DATA, r"median $\chi_t$")
    ax_chi.axhline(
        chi_bar_med,
        color=_C_REF1,
        linestyle="--",
        label=rf"$\bar\chi\approx{chi_bar_med:.1f}$",
    )
    ax_chi.axhline(
        chi_tail_med,
        color=_C_REF2,
        linestyle=":",
        label=rf"$\chi\approx{chi_tail_med:.2f}$",
    )
    ax_chi.set_xlabel(r"Iteration $t$")
    ax_chi.set_ylabel(r"$\chi_t = \|d^{\pi^\star}/d^{\pi^t}\|_\infty$")
    ax_chi.legend(loc="center right", framealpha=0.92, handlelength=2.2)
    _panel_label(ax_chi, "(b)")

    # Row label on far left.
    _row_label(ax_delta, ensemble_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=FIG_DIR / "rate_gap_appendix_panels.pdf")
    args = parser.parse_args()

    plt.rcParams.update(_RC)

    random_payload = _load(RANDOM_INPUT)
    queuing_payload = _load(QUEUING_INPUT)
    gridworld_payload = _load(GRIDWORLD_INPUT)
    jcr_payload = _load(JCR_INPUT)

    random_recs = [
        r for r in random_payload["records"]
        if (r["n_states"], r["n_actions"]) == RANDOM_CONFIG_TO_PLOT
    ]
    q_recs = [r for r in queuing_payload["records"] if r.get("tag") == "random"]
    g_recs = [r for r in gridworld_payload["records"] if r.get("tag") == "random"]
    j_recs = [r for r in jcr_payload["records"] if r.get("tag") == "random"]

    rows = [
        (random_recs, r"Random Dirichlet ($|S|=20,\;|A|=4$)"),
        (q_recs, "Access-control queuing"),
        (g_recs, r"GridWorld $5\times 5$"),
        (j_recs, "Jack's car rental"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=_FIGSIZE)
    for (recs, name), (ax_delta, ax_chi) in zip(rows, axes):
        deltas, chis, psi0s = _ensemble_arrays(recs)
        _plot_one_row(ax_delta, ax_chi, deltas, chis, psi0s, name)

    fig.tight_layout(h_pad=0.6, w_pad=1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
