"""
Shared figure style for the FIRM boundary-closure paper, matching the 2017 JCP paper
(Basic, Degiuli & Ban): serif TeX fonts, grey/light-grey grids, and a small marker/colour
palette. Also provides ``solution_map`` for the qualitative contour+particle plots
(numerical solution on the cloud, exact-solution contours overlaid) in the 2017 style.

Note: with usetex on, every label is rendered by LaTeX, so passed strings must escape
``%`` as ``\\%`` and must not contain bare underscores.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.path import Path


def apply(usetex=True):
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Computer Modern Roman", "serif"]
    rcParams["font.size"] = 14
    rcParams["axes.titlesize"] = 14
    rcParams["axes.labelsize"] = 15
    rcParams["legend.fontsize"] = 10
    rcParams["mathtext.fontset"] = "cm"
    rcParams["axes.grid"] = False
    rcParams["savefig.bbox"] = "tight"
    try:
        rcParams["text.usetex"] = bool(usetex)
        if usetex:
            rcParams["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{bm}"
    except Exception:  # pragma: no cover
        rcParams["text.usetex"] = False


# 2017 marker/colour palette (FD black, then o/s/*/D/v in r/b/g/grey/purple)
SERIES = [
    dict(marker="o", color="tab:red", ms=7),
    dict(marker="s", color="tab:blue", ms=6),
    dict(marker="*", color="tab:green", ms=10),
    dict(marker="D", color="0.45", ms=6),
    dict(marker="v", color="tab:purple", ms=7),
]


def grid(ax):
    ax.grid(which="major", c="grey", lw=0.5, alpha=0.6)
    ax.grid(which="minor", c="lightgrey", lw=0.4, alpha=0.5)


def loglog(ax, dxs, series):
    """series: list of (label, errs, style_index). Draws a 2017-style log-log error plot."""
    for label, errs, k in series:
        s = SERIES[k % len(SERIES)]
        ax.loglog(dxs, errs, marker=s["marker"], color=s["color"], ms=s["ms"], lw=1.5,
                  ls="-", label=label)
    ax.set_xlabel(r"$\Delta$")
    ax.invert_xaxis()
    grid(ax)


def order_ref(ax, dxs, anchor, slope, label=None):
    """Faint reference line of a given slope through the first point, for the eye."""
    dxs = np.asarray(dxs, float)
    ref = anchor * (dxs / dxs[0]) ** slope
    ax.loglog(dxs, ref, "k--", lw=0.9, alpha=0.4, label=label)


def solution_map(ax, pos, p, field, poly, title="", fill_h=None, ncont=11, cmap="viridis",
                 s=14):
    """Qualitative solution map: particles coloured by the numerical solution ``p`` with
    exact-solution contour lines overlaid on the (polygon-masked) domain, in the 2017 style.
    Returns the scatter handle for a colourbar."""
    pos = np.asarray(pos, float)
    poly = np.asarray(poly, float)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=p, s=s, cmap=cmap, edgecolors="none", zorder=2)
    # exact-solution contours on a fine background grid, masked to the polygon
    xmin, ymin = poly.min(0)
    xmax, ymax = poly.max(0)
    gx = np.linspace(xmin, xmax, 200)
    gy = np.linspace(ymin, ymax, 200)
    GX, GY = np.meshgrid(gx, gy)
    inside = Path(poly).contains_points(np.column_stack([GX.ravel(), GY.ravel()])).reshape(GX.shape)
    Z = field.value(np.stack([GX, GY], axis=-1))
    if fill_h is not None:
        inside &= (GY <= fill_h)
    Z = np.where(inside, Z, np.nan)
    ax.contour(GX, GY, Z, levels=ncont, colors="k", linewidths=0.7, zorder=3, alpha=0.85)
    pp = np.vstack([poly, poly[0]])
    ax.plot(pp[:, 0], pp[:, 1], "k-", lw=1.3, zorder=4)
    if fill_h is not None:
        ax.axhline(fill_h, color="royalblue", ls="--", lw=1.2, zorder=4)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    return sc
