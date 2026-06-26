"""
Generate summary figures for the FIRM validation suite.

Reuses the test/capstone helpers so the plots reflect exactly what the suite
measures. Writes PNGs into firm/figures/.
  fig_convergence.png   wall closures + capstone variants (rel-L2 vs dx)
  fig_jitter.png        jitter/anisotropy robustness (4 panels)
  fig_kernel_radius.png kernel shape & support radius (3 panels)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

import plotstyle
plotstyle.apply(usetex=True)
import matplotlib.pyplot as plt

import firm_core as fc
import geometry2d as g2
import manufactured as mf
import poisson as ps
import compare_normalization as cn
import capstone_poisson as cap
import hodge_projection as hp
import test_11_jitter_robustness as t11
import test_12_kernel_radius as t12
from testkit import observed_order

OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
CF = mf.complex_field(np.pi, 0.3)


# ----------------------------------------------------------------- convergence
def _tank_errs(dx, seed, wc):
    """Wedge-tank wall-region RMS and overall rel-L2 for a wall closure, one seed."""
    pos = g2.tank_cloud(dx, cap.FILL_H, cap.TANK, jitter=0.30, seed=seed)
    A, b, info = ps.assemble(pos, dx, CF, poly=cap.TANK, fill_h=cap.FILL_H, wall_proj="GGP",
                             surface_mode="exact", wall_closure=wc)
    p = ps.solve(A, b)
    pe = CF.value(pos)
    sc = max(np.max(np.abs(pe)), 1e-30)
    wall = info["is_wall"]
    wrms = float(np.sqrt(np.mean(((p - pe)[wall]) ** 2)) / sc)
    l2 = float(np.linalg.norm(p - pe) / np.linalg.norm(pe))
    return wrms, l2


def fig_convergence():
    dxs = [0.06, 0.045, 0.033, 0.024, 0.018]
    seeds = [7, 11, 19, 23, 31, 42]
    data = {}
    for wc in ("projection", "ghost"):
        wall, l2 = [], []
        for dx in dxs:
            vals = [_tank_errs(dx, s, wc) for s in seeds]
            wall.append(float(np.median([v[0] for v in vals])))
            l2.append(float(np.median([v[1] for v in vals])))
        data[wc] = (wall, l2)

    sty = {"projection": ("tab:red", "o"), "ghost": ("tab:green", "s")}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for wc in ("projection", "ghost"):
        o = observed_order(dxs, data[wc][0])
        ax[0].loglog(dxs, data[wc][0], sty[wc][1] + "-", color=sty[wc][0], lw=1.5, ms=7,
                     label=wc + r" ($%.2f$)" % o)
    ax[0].set_xlabel(r"$\Delta$"); ax[0].set_ylabel(r"Neumann-region RMS"); ax[0].legend()
    plotstyle.grid(ax[0]); ax[0].invert_xaxis()
    ax[0].set_title(r"Neumann region, projection vs algebraic ghost")
    for wc in ("projection", "ghost"):
        o = observed_order(dxs, data[wc][1])
        ax[1].loglog(dxs, data[wc][1], sty[wc][1] + "-", color=sty[wc][0], lw=1.5, ms=7,
                     label=wc + r" ($%.2f$)" % o)
    ax[1].set_xlabel(r"$\Delta$"); ax[1].set_ylabel(r"Error"); ax[1].legend()
    plotstyle.grid(ax[1]); ax[1].invert_xaxis()
    ax[1].set_title(r"Overall solution, projection vs algebraic ghost")
    fig.suptitle(r"Wedge tank, seed-averaged convergence")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_convergence.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- jitter
def fig_jitter():
    jits = [0.0, 0.1, 0.2, 0.3, 0.4]
    dx = t11.DX
    pois_trb, pois_den, ratio, cond, lam_med, smooth, ratl = [], [], [], [], [], [], []
    for j in jits:
        et = cn.poisson_dirichlet(CF, [dx], j, "trb")[0]
        ed = cn.poisson_dirichlet(CF, [dx], j, "denom")[0]
        pois_trb.append(et); pois_den.append(ed); ratio.append(ed / et)
        cond.append(t11._linear_exact(j)[2])
        lm, sm, rt = t11._interior_activation(j)
        lam_med.append(lm); smooth.append(sm); ratl.append(rt)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for a in ax.ravel():
        plotstyle.grid(a)
    ax[0, 0].plot(jits, pois_trb, "o-", color="tab:blue", label=r"$\mathrm{tr}\,\mathbf{B}$")
    ax[0, 0].plot(jits, pois_den, "s-", color="tab:red", label=r"denominator")
    ax[0, 0].set_xlabel(r"chaos $c$ (fraction of $\Delta$)"); ax[0, 0].set_ylabel(r"Poisson Error")
    ax[0, 0].set_title(r"(a) solution error vs disorder ($\Delta=%.3f$)" % dx); ax[0, 0].legend()

    ax[0, 1].plot(jits, ratio, "o-", color="tab:purple")
    ax[0, 1].axhline(1.0, color="k", ls="--", alpha=0.4)
    ax[0, 1].set_xlabel(r"chaos $c$"); ax[0, 1].set_ylabel(r"denominator / trace error ratio")
    ax[0, 1].set_title(r"(b) denominator edge widens with anisotropy")

    ax[1, 0].plot(jits, cond, "o-", color="tab:orange")
    ax[1, 0].set_xlabel(r"chaos $c$"); ax[1, 0].set_ylabel(r"$\max\,\mathrm{cond}(\mathbf{M})$")
    ax[1, 0].set_title(r"(c) conditioning grows but $\mathbf{M}$ stays invertible")

    ax[1, 1].plot(jits, lam_med, "o-", label=r"median $\lambda$ (interior)")
    ax[1, 1].plot(jits, smooth, "s-", label=r"smoothstep $\sigma$ (max)")
    ax[1, 1].plot(jits, ratl, "^-", label=r"rational $\sigma$, $c=0.2$ (median)")
    ax[1, 1].axhline(2 / 3, color="k", ls=":", alpha=0.5, label=r"threshold $2/3$")
    ax[1, 1].set_xlabel(r"chaos $c$"); ax[1, 1].set_ylabel(r"activation")
    ax[1, 1].set_title(r"(d) interior surface false-activation"); ax[1, 1].legend()

    fig.suptitle(r"Jitter / anisotropy robustness")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_jitter.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------- kernel / radius
def fig_kernel_radius():
    radii = t12.RADII
    dxs = [0.06, 0.045, 0.033]
    seeds = [7, 11, 19]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    for kname, kfn in t12.KERNELS.items():
        ptw, pois, minnb = [], [], []
        for hf in radii:
            g, l, p, mnb = t12._linear_and_pointwise(0.045, hf, kfn, CF)
            ptw.append(p); minnb.append(mnb)
            e = [float(np.median([t12._poisson_dir(dx, hf, kfn, CF, s) for s in seeds])) for dx in dxs]
            pois.append(e[-1])
        lbl = kname.split()[0]
        ax[0].plot(radii, ptw, "o-", label=lbl)
        ax[1].plot(radii, pois, "o-", label=lbl)
        ax[2].plot(radii, minnb, "o-", label=lbl)
    for a in ax:
        plotstyle.grid(a)
    ax[0].set_xlabel(r"support radius $h/\Delta$"); ax[0].set_ylabel(r"pointwise Laplacian Error")
    ax[0].set_title(r"(a) pointwise Laplacian error"); ax[0].legend()
    ax[1].set_xlabel(r"support radius $h/\Delta$"); ax[1].set_ylabel(r"Poisson Error (finest $\Delta$)")
    ax[1].set_title(r"(b) Poisson solution error"); ax[1].set_yscale("log"); ax[1].legend()
    ax[2].axhline(3, color="r", ls="--", alpha=0.5, label=r"2D rank floor ($d{+}1$)")
    ax[2].set_xlabel(r"support radius $h/\Delta$"); ax[2].set_ylabel(r"min.\ neighbours")
    ax[2].set_title(r"(c) min neighbour count (chaos $60\%$)"); ax[2].legend()
    fig.suptitle(r"Kernel shape \& support radius robustness (chaos $60\%$)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_kernel_radius.png"), dpi=130)
    plt.close(fig)


def fig_error_field():
    """Capstone error field |p - p*| on the wedge tank (GGP/exact, finest dx), serif style."""
    r = cap.run_case(cap.DXS[-1], "GGP", "exact")
    fig, ax = plt.subplots(figsize=(9, 6))
    poly = np.vstack([cap.TANK, cap.TANK[0]])
    ax.plot(poly[:, 0], poly[:, 1], "k-", lw=1.3)
    ax.axhline(cap.FILL_H, color="royalblue", ls="--", lw=1.2, alpha=0.7)
    err = np.abs(r["p"] - r["pe"]) / max(np.max(np.abs(r["pe"])), 1e-30)
    sc = ax.scatter(r["pos"][:, 0], r["pos"][:, 1], c=err, s=11, cmap="magma")
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, shrink=0.8, label=r"$|p-p^\star|/\max|p^\star|$")
    ax.set_title(r"Error concentrates on the walls/wedge, vanishing toward the free surface")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "capstone_error_field.png"), dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- hodge / projection
def fig_hodge():
    """Helmholtz--Hodge decomposition with the FIRM operators (hodge_projection.py):
    (left) single-decomposition residual divergence |Du|_interior vs dx -- it
    converges (order > 0), so L is decomposition-consistent; (middle) iterated defect
    correction r_k/r0 -- it drops to a floor then stalls, while feeding the raw
    wall-truncated divergence diverges; (right) the choice of Laplacian."""
    # ---- panel 1: |Du| vs dx for box{proj,ghost} and tank{proj} ----
    box_dxs = hp.DXS_BOX
    seeds = (7, 11, 19)
    series = {}
    for wc in ("projection", "ghost"):
        du = []
        for dx in box_dxs:
            cs = [hp.run_case("box", dx, wc, "trb", s) for s in seeds]
            du.append(float(np.median([c["du1"] for c in cs])))
        series[("box", wc)] = (box_dxs, du)
    tank_dxs = cap.DXS
    series[("tank", "projection")] = (tank_dxs,
                                       [hp.run_case("tank", dx, "projection", "trb", 7)["du1"] for dx in tank_dxs])

    sty = {("box", "projection"): ("tab:red", "o", "box, projection"),
           ("box", "ghost"): ("tab:green", "s", "box, ghost"),
           ("tank", "projection"): ("tab:blue", "^", "tank, projection")}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for key, (dxs, du) in series.items():
        col, mk, lbl = sty[key]
        o = observed_order(dxs, du)
        ax[0].loglog(dxs, du, mk + "-", color=col, lw=1.5, ms=7, label=lbl + r" ($%.2f$)" % o)
    # order-1 reference through the finest tank point
    dref = np.array([min(min(s[0]) for s in series.values()), max(max(s[0]) for s in series.values())])
    anchor = series[("tank", "projection")]
    c1 = anchor[1][-1] / anchor[0][-1]
    ax[0].loglog(dref, c1 * dref, "k--", lw=1.0, alpha=0.5, label=r"order 1")
    ax[0].set_xlabel(r"$\Delta$"); ax[0].set_ylabel(r"residual divergence $\|\nabla\!\cdot u\|$")
    ax[0].legend(); plotstyle.grid(ax[0]); ax[0].invert_xaxis()
    ax[0].set_title(r"(a) one decomposition: residual $\to 0$ as $\Delta\to0$")

    # ---- panel 2: iterated decay r_k/r0 -- fixed unit step vs minimal-residual step ----
    # Interior-source runs (consistent): both fixed and MR stall at the defect near-null floor.
    # Raw-source runs (full truncated wall divergence): the FIXED step DIVERGES (rho>1) while the
    # MR step stays non-expansive (rho<=1) -- the manufactured-field analogue of the unsteady solver.
    runs = [("box", "projection", 0.3, False, "tab:red", "box proj, interior src"),
            ("box", "projection", 0.3, True, "tab:gray", "box proj, raw src"),
            ("tank", "projection", 0.3, True, "tab:blue", "tank proj, raw src")]
    for domain, wc, jit, raw, col, lbl in runs:
        case = hp.run_case(domain, 0.045, wc, "trb", 7, jit)
        rs = np.array(hp.iterate(case, kmax=60, tol=1e-12, raw=raw))
        ax[1].semilogy(np.arange(len(rs)), rs / rs[0], "-", color=col, lw=1.6,
                       label="fixed: " + lbl)
        if raw:  # the minimal-residual cure for the diverging (raw-source) cases
            rm = np.array(hp.iterate_mr(case, kmax=60, tol=1e-12, raw=raw))
            ax[1].semilogy(np.arange(len(rm)), rm / rm[0], "--", color=col, lw=1.6,
                           label="MR: " + lbl)
    ax[1].set_xlabel(r"projection iteration $k$"); ax[1].set_ylabel(r"$\|\nabla\!\cdot u^{(k)}\| / \|\nabla\!\cdot u^*\|$")
    ax[1].legend(fontsize=8); plotstyle.grid(ax[1])
    ax[1].set_title(r"(b) iterated defect correction: fixed step diverges, MR step bounded")

    # ---- panel 3: operator comparison (Dirichlet box, common gradient) ----
    res = hp.compare_operators(dxs=hp.DXS_BOX, seeds=(7, 11, 19))
    opsty = {"sum": ("tab:gray", "x"), "new": ("tab:red", "o"), "denom": ("tab:orange", "v"),
             "gfdm": ("tab:purple", "s"), "fi": ("tab:blue", "^")}
    for op in hp.OPERATORS:
        rows, order = res[op]
        du = [r["du1"] for r in rows]
        col, mk = opsty[op]
        ax[2].loglog(hp.DXS_BOX, du, mk + "-", color=col, lw=1.5, ms=7,
                     label=hp.OP_LABEL[op] + r" ($%+.2f$)" % order)
    ax[2].set_xlabel(r"$\Delta$"); ax[2].set_ylabel(r"residual $\|\nabla\!\cdot u\|$ (interior)")
    ax[2].legend(fontsize=8); plotstyle.grid(ax[2]); ax[2].invert_xaxis()
    ax[2].set_title(r"(c) which Laplacian? (shared gradient, Dirichlet box)")

    fig.suptitle(r"Helmholtz--Hodge decomposition with the renormalised FIRM operators")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_hodge.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_convergence(); print("saved fig_convergence.png")
    fig_jitter(); print("saved fig_jitter.png")
    fig_kernel_radius(); print("saved fig_kernel_radius.png")
    fig_error_field(); print("saved capstone_error_field.png")
    fig_hodge(); print("saved fig_hodge.png")
