"""
Paper figures for the FIRM boundary-closure manuscript, in the 2017 JCP house style
(serif TeX fonts, grey grids, o/s/*/D markers) -- see plotstyle.py. Complements the
validation-suite figures in make_figures.py.

By default the convergence panels are drawn from the cached figures/paper_numbers.json
(fast restyle, no re-solving); pass --recompute to rerun the benchmark battery first.
The qualitative contour+particle solution maps are always solved fresh (one solve each).

Produces, in firm/figures/:
  fig_b1_dirichlet.png      square Dirichlet: FIRM vs GFDM at chaos 30/60/90%
  fig_neumann_baseline.png  all-Neumann box: FIRM projection/ghost vs GFDM constraint/penalty
  fig_b2_curved.png         star domain: Dirichlet and Neumann closures
  fig_b3_robin.png          flower Robin convergence
  fig_b5_surface.png        free-surface detection (sigma map + natural-vs-exact convergence)
  fig_solution_maps.png     contour+particle solution maps for every domain (B1/B2/B3/tank)
  fig_schematic.png         the three closures (projection, algebraic ghost, surface detection)

Run:  python3 paper_figures.py [--recompute] [--quick]
"""
import os
import sys
import json

import numpy as np

import plotstyle
plotstyle.apply(usetex=True)
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import poisson as ps
import bvp
import paper_benchmarks as pb
from convergence import CHAOS

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
NUM = os.path.join(OUT, "paper_numbers.json")
DXS = pb.DXS


def _save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("saved", name)


# ============================================================ Franke operator
def fig_franke(d):
    r = d["franke_operator"]
    chaos = [0, 30, 60, 90]
    dxs = pb.DXS_FRANKE
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.6), sharey=True)
    for k, cp in enumerate(chaos):
        cc = f"chaos{cp}"
        series = [
            (r"renormalised ($%.2f$)" % r[cc]["new"]["order"], r[cc]["new"]["errs"], 0),
            (r"GFDM ($%.2f$)" % r[cc]["gfdm"]["order"], r[cc]["gfdm"]["errs"], 1)]
        if "fi" in r[cc]:
            series.append((r"FI ($%.2f$)" % r[cc]["fi"]["order"], r[cc]["fi"]["errs"], 2))
        plotstyle.loglog(ax[k], dxs, series)
        ttl = r"regular" if cp == 0 else r"disorder $%d\%%$" % cp
        ax[k].set_title(ttl)
        ax[k].legend(loc="best")
    ax[0].set_ylabel(r"Error")
    fig.suptitle(r"Franke function, pointwise Laplacian approximation error vs $\Delta$")
    fig.tight_layout()
    _save(fig, "fig_franke_operator.png")


# ============================================================ Neumann normalisation
def fig_neumann_straight(d):
    r = d["neumann_straight"]
    dxs = pb.DXS_NEU
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    keymap = [("projection", r"projection", 0), ("ghost-trace", r"ghost, trace $N$", 2),
              ("ghost-denom", r"ghost, denominator $N$", 1)]
    for key, lab, k in keymap:
        plotstyle.loglog(ax, dxs, [(lab + r" ($%.2f$)" % r[key]["order"], r[key]["errs"], k)])
    plotstyle.order_ref(ax, dxs, r["ghost-denom"]["errs"][0], 2.0, label=r"second order")
    ax.set_ylabel(r"Neumann-region Error"); ax.legend(loc="best")
    ax.set_title(r"Straight Neumann edge, normalisation of the ghost closure")
    fig.tight_layout()
    _save(fig, "fig_neumann_straight.png")


# ============================================================ convergence panels
def fig_b1(d):
    r = d["B1_square_dirichlet"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for k, c in enumerate(CHAOS):
        cc = f"chaos{int(c*100)}"
        series = [
            (r"FIRM ($%.2f$)" % r[cc]["firm"]["order"], r[cc]["firm"]["errs"], 0),
            (r"GFDM ($%.2f$)" % r[cc]["gfdm"]["order"], r[cc]["gfdm"]["errs"], 1)]
        if "fi" in r[cc]:
            series.append((r"FI ($%.2f$)" % r[cc]["fi"]["order"], r[cc]["fi"]["errs"], 2))
        plotstyle.loglog(ax[k], DXS, series)
        ax[k].set_title(r"disorder $%d\%%$" % int(c * 100))
        ax[k].legend(loc="best")
    ax[0].set_ylabel(r"Error")
    fig.suptitle(r"Square domain, Dirichlet, renormalised FIRM and the second-order "
                 r"GFDM and Full-Inverse operators")
    fig.tight_layout()
    _save(fig, "fig_b1_dirichlet.png")


def fig_neumann(d):
    res = d["neumann_baseline"]
    labels = [("firm-proj", r"FIRM projection", 0), ("firm-ghost", r"FIRM ghost", 2),
              ("gfdm-constraint", r"GFDM constraint", 4), ("gfdm-penalty", r"GFDM penalty", 3)]
    chaos = sorted(int(k.replace("chaos", "")) for k in res)
    fig, ax = plt.subplots(1, len(chaos), figsize=(6 * len(chaos), 5), sharey=True)
    ax = np.atleast_1d(ax)
    for j, cp in enumerate(chaos):
        cc = f"chaos{cp}"
        plotstyle.loglog(ax[j], DXS, [
            (lab + r" ($%.2f$)" % res[cc][key]["order"], res[cc][key]["errs"], k)
            for key, lab, k in labels])
        ax[j].set_title(r"all-Neumann box, disorder $%d\%%$" % cp)
        ax[j].legend(loc="best")
    ax[0].set_ylabel(r"Error (mean-removed)")
    fig.suptitle(r"Neumann boundaries, mirror-ghost and flux-only closures")
    fig.tight_layout()
    _save(fig, "fig_neumann_baseline.png")


def fig_b2(d):
    r = d["B2_star"]
    keymap = [("dir_firm", r"FIRM Dirichlet", 0), ("dir_gfdm", r"GFDM Dirichlet", 1),
              ("neu_proj", r"FIRM Neumann proj.", 2), ("neu_ghost", r"FIRM Neumann ghost", 3),
              ("neu_gfdm", r"GFDM Neumann", 4)]
    chaos = sorted(int(k.replace("chaos", "")) for k in r)
    fig, ax = plt.subplots(1, len(chaos), figsize=(6 * len(chaos), 5), sharey=True)
    ax = np.atleast_1d(ax)
    for j, cp in enumerate(chaos):
        cc = f"chaos{cp}"
        plotstyle.loglog(ax[j], DXS, [
            (lab + r" ($%.2f$)" % r[cc][key]["order"], r[cc][key]["errs"], k)
            for key, lab, k in keymap])
        ax[j].set_title(r"star domain, disorder $%d\%%$" % cp)
        ax[j].legend(loc="best")
    ax[0].set_ylabel(r"Error")
    fig.suptitle(r"Star domain, value and flux closures")
    fig.tight_layout()
    _save(fig, "fig_b2_curved.png")


def fig_b3(d):
    r = d["B3_flower_robin"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    plotstyle.loglog(ax, DXS, [
        (r"disorder $%d\%%$ ($%.2f$)" % (int(c * 100), r[f"chaos{int(c*100)}"]["order"]),
         r[f"chaos{int(c*100)}"]["errs"], k) for k, c in enumerate(CHAOS)])
    ax.set_ylabel(r"Error")
    ax.legend(loc="best")
    ax.set_title(r"Flower domain, Robin condition $\partial u/\partial n + u = f$")
    fig.tight_layout()
    _save(fig, "fig_b3_robin.png")


def fig_b5(d):
    conv = d["B5_surface"]["convergence"]
    dx = 0.045
    pos = g2.tank_cloud(dx, pb.FILL_H, pb.TANK, jitter=0.30, seed=7)
    A, b, info = ps.assemble(pos, dx, mf.complex_field(np.pi, 0.3), poly=pb.TANK, fill_h=pb.FILL_H,
                             wall_proj="GGP", surface_mode="natural", activation="smoothstep",
                             wall_closure="projection")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    pp = np.vstack([pb.TANK, pb.TANK[0]])
    ax[0].plot(pp[:, 0], pp[:, 1], "k-", lw=1.3)
    ax[0].axhline(pb.FILL_H, color="royalblue", ls="--", lw=1.2)
    sc = ax[0].scatter(pos[:, 0], pos[:, 1], c=info["sigma"], s=12, cmap="viridis", vmin=0, vmax=1)
    ax[0].set_aspect("equal"); ax[0].set_xticks([]); ax[0].set_yticks([])
    fig.colorbar(sc, ax=ax[0], shrink=0.8, label=r"detector $\sigma_i$")
    ax[0].set_title(r"(a) detected free surface $\sigma_i$ (interior $\equiv 0$)")
    plotstyle.loglog(ax[1], DXS, [
        (r"prescribed ($%.2f$)" % conv["exact"]["order"], conv["exact"]["errs"], 3),
        (r"detected ($%.2f$)" % conv["natural"]["order"], conv["natural"]["errs"], 2)])
    ax[1].set_ylabel(r"surface-region RMS"); ax[1].legend(loc="best")
    ax[1].set_title(r"(b) detected vs prescribed surface (no tuning)")
    fig.suptitle(r"Free-surface detection from the cloud")
    fig.tight_layout()
    _save(fig, "fig_b5_surface.png")

    # Save as 2 separate figures
    fig_a, ax_a = plt.subplots(figsize=(7, 5))
    ax_a.plot(pp[:, 0], pp[:, 1], "k-", lw=1.3)
    ax_a.axhline(pb.FILL_H, color="royalblue", ls="--", lw=1.2)
    sc = ax_a.scatter(pos[:, 0], pos[:, 1], c=info["sigma"], s=12, cmap="viridis", vmin=0, vmax=1)
    ax_a.set_aspect("equal"); ax_a.set_xticks([]); ax_a.set_yticks([])
    fig_a.colorbar(sc, ax=ax_a, shrink=0.8, label=r"detector $\sigma_i$")
    ax_a.set_title(r"detected free surface $\sigma_i$ (interior $\equiv 0$)")
    fig_a.tight_layout()
    _save(fig_a, "fig_b5_surface_a.png")

    fig_b, ax_b = plt.subplots(figsize=(6.5, 5))
    plotstyle.loglog(ax_b, DXS, [
        (r"prescribed ($%.2f$)" % conv["exact"]["order"], conv["exact"]["errs"], 3),
        (r"detected ($%.2f$)" % conv["natural"]["order"], conv["natural"]["errs"], 2)])
    ax_b.set_ylabel(r"surface-region RMS"); ax_b.legend(loc="best")
    ax_b.set_title(r"detected vs prescribed surface (no tuning)")
    fig_b.tight_layout()
    _save(fig_b, "fig_b5_surface_b.png")


# ============================================================ solution maps (fresh)
def fig_solution_maps():
    dx = 0.03
    star = g2.star_polygon(n=720, r0=0.5, amp=0.2, k=5, center=(0.5, 0.5))
    flower = g2.flower_polygon(n=720, base=0.5, amp=0.08, k=8, center=(0.5, 0.5))
    trig = mf.trig_field(np.pi)
    cf = mf.complex_field(np.pi, 0.3)

    # B1 square Dirichlet
    p1 = g2.jittered_box(dx, 0.30, 7)
    A, b, info = bvp.assemble(p1, dx, trig, pb.SQUARE, "dirichlet"); s1 = bvp.solve(A, b)
    # B2 star Dirichlet
    p2 = g2.polygon_cloud(star, dx, 0.30, 7)
    A, b, info = bvp.assemble(p2, dx, trig, star, "dirichlet"); s2 = bvp.solve(A, b)
    # B3 flower Robin
    p3 = g2.polygon_cloud(flower, dx, 0.30, 7)
    A, b, info = bvp.assemble(p3, dx, trig, flower, "robin", robin_alpha=1.0); s3 = bvp.solve(A, b)
    # tank free surface + wedge (complex field)
    p4 = g2.tank_cloud(0.05, pb.FILL_H, pb.TANK, jitter=0.30, seed=7)
    A, b, info = ps.assemble(p4, 0.05, cf, poly=pb.TANK, fill_h=pb.FILL_H, wall_proj="GGP",
                             surface_mode="natural", activation="smoothstep", wall_closure="ghost")
    s4 = ps.solve(A, b)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    sc = plotstyle.solution_map(ax[0], p1, s1, trig, pb.SQUARE, r"(a) square, Dirichlet")
    fig.colorbar(sc, ax=ax[0], shrink=0.75)
    sc = plotstyle.solution_map(ax[1], p2, s2, trig, star, r"(b) star, Dirichlet")
    fig.colorbar(sc, ax=ax[1], shrink=0.75)
    sc = plotstyle.solution_map(ax[2], p3, s3, trig, flower, r"(c) flower, Robin")
    fig.colorbar(sc, ax=ax[2], shrink=0.75)
    sc = plotstyle.solution_map(ax[3], p4, s4, cf, pb.TANK, r"(d) tank, walls + free surface",
                                fill_h=pb.FILL_H)
    fig.colorbar(sc, ax=ax[3], shrink=0.75)
    fig.suptitle(r"Numerical solution in colour with exact-solution contours, "
                 r"disorder $60\%$")
    fig.tight_layout()
    _save(fig, "fig_solution_maps.png")

    # Save as 4 separate figures
    fig_a, ax_a = plt.subplots(figsize=(5.5, 5))
    sc = plotstyle.solution_map(ax_a, p1, s1, trig, pb.SQUARE, r"square, Dirichlet")
    fig_a.colorbar(sc, ax=ax_a, shrink=0.75)
    fig_a.tight_layout()
    _save(fig_a, "fig_solution_maps_a.png")

    fig_b, ax_b = plt.subplots(figsize=(5.5, 5))
    sc = plotstyle.solution_map(ax_b, p2, s2, trig, star, r"star, Dirichlet")
    fig_b.colorbar(sc, ax=ax_b, shrink=0.75)
    fig_b.tight_layout()
    _save(fig_b, "fig_solution_maps_b.png")

    fig_c, ax_c = plt.subplots(figsize=(5.5, 5))
    sc = plotstyle.solution_map(ax_c, p3, s3, trig, flower, r"flower, Robin")
    fig_c.colorbar(sc, ax=ax_c, shrink=0.75)
    fig_c.tight_layout()
    _save(fig_c, "fig_solution_maps_c.png")

    fig_d, ax_d = plt.subplots(figsize=(5.5, 5))
    sc = plotstyle.solution_map(ax_d, p4, s4, cf, pb.TANK, r"tank, walls + free surface",
                                fill_h=pb.FILL_H)
    fig_d.colorbar(sc, ax=ax_d, shrink=0.75)
    fig_d.tight_layout()
    _save(fig_d, "fig_solution_maps_d.png")


# ============================================================ schematic
def fig_schematic():
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    rng = np.random.default_rng(1)

    a = ax[0]
    a.axhline(0.0, color="k", lw=2)
    pts = np.array([[0.0, 0.18], [-0.22, 0.30], [0.25, 0.33], [-0.10, 0.55], [0.15, 0.6], [0.0, 0.85]])
    a.scatter(pts[:, 0], pts[:, 1], c="tab:blue", s=45, zorder=3, label=r"fluid neighbours")
    a.scatter([0], [0.18], c="black", s=70, zorder=4, label=r"particle $i$")
    ghosts = pts.copy(); ghosts[:, 1] *= -1
    a.scatter(ghosts[:, 0], ghosts[:, 1], facecolors="none", edgecolors="tab:green", s=45,
              zorder=3, label=r"mirror ghosts")
    for p, gp in zip(pts, ghosts):
        a.plot([p[0], gp[0]], [p[1], gp[1]], color="tab:green", lw=0.5, alpha=0.4)
    a.annotate("", xy=(0.0, -0.05), xytext=(0.0, 0.18),
               arrowprops=dict(arrowstyle="->", color="tab:red"))
    a.text(0.04, 0.04, r"$g=\partial p/\partial n$", color="tab:red")
    a.set_title(r"(a) algebraic-ghost wall closure"); a.set_xlim(-0.5, 0.5); a.set_ylim(-1.0, 1.0)
    a.legend(loc="upper right"); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])

    b = ax[1]
    b.axhline(0.0, color="k", lw=2); b.axvline(0.0, color="k", lw=2)
    src = np.array([0.35, 0.45])
    b.scatter(*src, c="black", s=70, zorder=4); b.text(src[0] + 0.04, src[1], r"$i$")
    r1 = np.array([src[0], -src[1]]); r2 = np.array([-src[0], src[1]]); r12 = np.array([-src[0], -src[1]])
    for g, lab in [(r1, r"$R_1$"), (r2, r"$R_2$"), (r12, r"$R_2 R_1$")]:
        b.scatter(*g, facecolors="none", edgecolors="tab:green", s=55, zorder=3)
        b.text(g[0] + 0.04, g[1], lab, color="tab:green")
    b.set_title(r"(b) corner reflection group"); b.set_xlim(-0.8, 0.8); b.set_ylim(-0.8, 0.8)
    b.set_aspect("equal"); b.set_xticks([]); b.set_yticks([])

    c = ax[2]
    cloud = rng.uniform(-0.5, 0.0, (40, 2)); cloud[:, 0] = rng.uniform(-0.5, 0.5, 40)
    c.scatter(cloud[:, 0], cloud[:, 1], c="tab:blue", s=20, alpha=0.6)
    c.scatter([0], [-0.07], c="black", s=70, zorder=4); c.text(0.04, -0.07, r"$i$")
    c.axhline(0.0, color="royalblue", ls="--", lw=1.2)
    c.text(0.2, 0.04, r"free surface", color="royalblue")
    c.annotate("", xy=(0.0, -0.32), xytext=(0.0, -0.07),
               arrowprops=dict(arrowstyle="->", color="tab:orange", lw=2))
    c.text(0.04, -0.25, r"$\mathbf{o}_i$", color="tab:orange")
    c.annotate("", xy=(0.0, 0.12), xytext=(0.0, -0.07),
               arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
    c.text(-0.5, 0.05, r"$\mathbf{n}_s=-\mathbf{r}_i/|\mathbf{r}_i|$", color="tab:red")
    c.set_title(r"(c) surface detected from $\mathbf{o}_i,\mathbf{r}_i$")
    c.set_xlim(-0.6, 0.6); c.set_ylim(-0.6, 0.3); c.set_aspect("equal")
    c.set_xticks([]); c.set_yticks([])

    fig.suptitle(r"FIRM boundary closures")
    fig.tight_layout()
    _save(fig, "fig_schematic.png")


def main(argv):
    if "--recompute" in argv:
        os.system(f"{sys.executable} {os.path.join(HERE, 'paper_figures_compute.py')}"
                  + (" --quick" if "--quick" in argv else ""))
    if not os.path.exists(NUM):
        sys.exit("paper_numbers.json not found; run with --recompute first")
    d = json.load(open(NUM))
    fig_schematic()
    fig_solution_maps()
    if "franke_operator" in d:
        fig_franke(d)
    if "neumann_straight" in d:
        fig_neumann_straight(d)
    fig_b1(d)
    fig_neumann(d)
    fig_b2(d)
    fig_b3(d)
    fig_b5(d)


if __name__ == "__main__":
    main(sys.argv[1:])
