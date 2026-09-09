"""Shared matplotlib styling for the analysis.

Palette is the validated, colorblind-safe categorical set from the dataviz design
system (CVD Delta E >= 8 between adjacent hues). Assigned in fixed order, never
cycled. Sequential magnitude uses a single blue ramp; polarity uses a warm/cool
diverging pair with a neutral midpoint.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical hues in fixed order (light mode).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = "#2a78d6"          # single-hue magnitude
DIVERGING = ("#2a78d6", "#f0efec", "#eb6834")  # cool | neutral | warm

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


def apply_style() -> None:
    """Install a clean, recessive-gridline style with the validated palette."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
            "axes.edgecolor": INK_MUTED,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.labelsize": 10,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def despine_labels(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax
