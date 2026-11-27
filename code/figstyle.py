"""Common figure format: every panel is a separate file of the same size."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANEL = (3.4, 2.7)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.2,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_MAIN = "#1f3b73"
C_IN = "#1b7f3a"
C_OUT = "#b3261e"
C_GREY = "0.55"


def panel():
    fig, ax = plt.subplots(figsize=PANEL)
    return fig, ax
