"""Shared editorial palette for in-memory analysis figures."""

import matplotlib.pyplot as plt
import seaborn as sns

NAVY = "#051C2C"
BLUE = "#2251FF"
TEAL = "#008C95"
GRAY = "#8593A0"
LIGHT = "#E8EDF1"
PALETTE = [BLUE, TEAL, NAVY, "#6784A3", "#87BDD0"]


def apply_theme():
    """Estética editorial: blanco, azul profundo y jerarquía sin decoración."""
    sns.set_theme(style="white", palette=PALETTE)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 16,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#B7C2CB",
            "axes.linewidth": 0.65,
            "axes.grid": False,
            "xtick.color": "#526371",
            "ytick.color": "#526371",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 2,
            "lines.markersize": 4,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "figure.dpi": 120,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
