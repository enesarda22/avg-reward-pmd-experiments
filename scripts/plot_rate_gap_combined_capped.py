"""Combined rate-gap plot for the *capped super-geometric* variant.

Sibling of ``plot_rate_gap_combined.py``. Reads the
``rate_gap_data_capped.json`` and ``rate_gap_queuing_data_capped.json``
outputs and produces ``rate_gap_combined_capped.pdf``.

Run:
    uv run python scripts/plot_rate_gap_combined_capped.py
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

# Okabe-Ito: blue, vermilion, green, reddish purple.
_C_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

_RC = {
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIX", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.30,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
}

_FIGSIZE = (5.9, 2.8)

RANDOM_CONFIGS_TO_PLOT: list[tuple[int, int]] = [(20, 4)]


def _stack_normalized(recs):
    return np.stack([
        np.asarray(r["chi_traj"]) / float(np.max(r["chi_traj"]))
        for r in recs
    ])


def _plot_band(ax, t, traj, color, label):
    med = np.median(traj, axis=0)
    q25 = np.quantile(traj, 0.25, axis=0)
    q75 = np.quantile(traj, 0.75, axis=0)
    ax.fill_between(t, q25, q75, color=color, alpha=0.18, linewidth=0)
    ax.semilogy(t, med, color=color, linewidth=2.0, label=label)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random-input", type=Path, default=RANDOM_INPUT)
    parser.add_argument("--queuing-input", type=Path, default=QUEUING_INPUT)
    parser.add_argument("--gridworld-input", type=Path, default=GRIDWORLD_INPUT)
    parser.add_argument("--jcr-input", type=Path, default=JCR_INPUT)
    parser.add_argument(
        "--output", type=Path,
        default=FIG_DIR / "rate_gap_combined_capped.pdf",
    )
    args = parser.parse_args()

    for p in (args.random_input, args.queuing_input, args.gridworld_input,
              args.jcr_input):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found.")

    with args.random_input.open() as f:
        random_payload = json.load(f)
    with args.queuing_input.open() as f:
        queuing_payload = json.load(f)
    with args.gridworld_input.open() as f:
        gridworld_payload = json.load(f)
    with args.jcr_input.open() as f:
        jcr_payload = json.load(f)

    plt.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=_FIGSIZE)

    random_groups: dict[tuple[int, int], list[dict]] = {}
    for r in random_payload["records"]:
        key = (r["n_states"], r["n_actions"])
        random_groups.setdefault(key, []).append(r)

    color_iter = iter(_C_PALETTE)

    for cfg in RANDOM_CONFIGS_TO_PLOT:
        recs = random_groups[cfg]
        traj = _stack_normalized(recs)
        t = np.arange(traj.shape[1])
        n_s, n_a = cfg
        _plot_band(ax, t, traj, next(color_iter),
                   rf"$|S|={n_s},\,|A|={n_a}$")

    q_recs = [r for r in queuing_payload["records"] if r.get("tag") == "random"]
    q_traj = _stack_normalized(q_recs)
    t_q = np.arange(q_traj.shape[1])
    _plot_band(ax, t_q, q_traj, next(color_iter),
               r"Queue admission")

    g_recs = [r for r in gridworld_payload["records"] if r.get("tag") == "random"]
    g_traj = _stack_normalized(g_recs)
    t_g = np.arange(g_traj.shape[1])
    _plot_band(ax, t_g, g_traj, next(color_iter),
               r"GridWorld $5\times 5$")

    j_recs = [r for r in jcr_payload["records"] if r.get("tag") == "random"]
    j_traj = _stack_normalized(j_recs)
    t_j = np.arange(j_traj.shape[1])
    _plot_band(ax, t_j, j_traj, next(color_iter),
               r"Jack's car rental")

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)

    ax.set_xlabel(r"Iteration $t$")
    ax.set_ylabel(r"$\dfrac{\chi_t}{\bar\chi}$", rotation=0, labelpad=10, va="center")
    ax.legend(
        loc="lower right",
        bbox_to_anchor=(0.83, 0.0),
        framealpha=0.92,
        edgecolor="0.85",
        fancybox=True,
        handlelength=1.6,
        ncol=1,
        borderpad=0.5,
        labelspacing=0.5,
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
