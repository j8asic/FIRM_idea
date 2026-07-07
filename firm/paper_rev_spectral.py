"""
Revision tooling (paper_rev_*): SPECTRAL study of the boundary-closed FIRM operator.

The paper claims the boundary-closed system stays invertible but proves nothing.
This script computes the smallest eigenvalues of the closed operator under
refinement and disorder, to detect spurious near-null modes, and checks whether
boundary rows (where the correction weight w_ij = 1 - x_ij . V_i can change sign)
generate them.

Row scaling.  Every assembled row of A equals (a discrete Laplacian) / N_i,
where N_i is the per-row normalisation the assembly divides the RHS source by
(bvp.assemble: b[i] = laplacian/N_i + BC terms; poisson.assemble returns
info["N"]).  bvp.assemble does not return N_i, but A is field-independent and
the BC data of a field with value==0, grad==0 vanish from b, so assembling with
a fake field of unit Laplacian gives b[i] = 1/N_i exactly on every row and every
closure (verified in _selfcheck).  The scaled operator  A_s = -diag(N) A  then
approximates the continuous -Laplacian nodewise; its eigenvalues are compared
against the continuous anchors

  * all-Dirichlet unit square:  lambda_1 = 2 pi^2 ~= 19.7392
  * all-Neumann unit square (no pin / no zero-mean constraint):
    lambda_1 = 0 (constant null mode), lambda_2 = pi^2 ~= 9.8696.

Configurations (medians over seeds, halving refinement, disorder = 2017 chaos %
so jitter = disorder/200 per geometry2d convention):
  a. unit square, all-Dirichlet value closure (bvp)          -> anchor 2 pi^2
  b. all-Neumann unit box, projection AND algebraic-ghost    -> anchors 0, pi^2
  c. wedge tank, full closure stack (poisson.assemble, GGP walls + surface),
     projection/ghost with exact surface + ghost with the detected (natural,
     smoothstep) surface, at the capstone's 30% disorder     -> no anchor,
     invertibility = lambda_1 bounded away from 0 under refinement.

Per case also recorded: boundary-l2-mass fraction of the 6 smallest-|lambda|
right eigenvectors (boundary-localisation indicator), the fraction of boundary
rows with >=1 negative off-diagonal entry of A (i.e. W_ij w_ij < 0 <=> a
sign-flipped correction weight) and the overall negative-off-diagonal fraction
in boundary rows (interior rows kept as the baseline).

Numerics: scipy.sparse.linalg.eigs shift-invert (sigma = -1, safely left of the
spectrum; sigma = 0 is exactly singular for the pure-Neumann null mode), k = 8,
sorted by |lambda|.  The matrices are non-symmetric so eigenvalues may be
complex; magnitudes are reported and any significant imaginary part is flagged.
Cross-checks: dense eigvals + dense SVD at the coarsest resolution, and a sparse
svds(which='SM') on the Dirichlet case.

Writes figures/paper_extra_numbers.json  key 'spectral'  and figures/fig_spectra.png.
Run:  python3 paper_rev_spectral.py
"""
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bvp
import capstone_poisson as cap
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import poisson as ps

import scipy.linalg as sla
import scipy.sparse as sps
import scipy.sparse.linalg as spla

OUT = os.path.join(HERE, "figures", "paper_extra_numbers.json")
FIG = os.path.join(HERE, "figures", "fig_spectra.png")

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
DXS = [0.05, 0.025, 0.0125]          # halving; ~400 / ~1600 / ~6400 nodes
TANK_DXS = [0.1, 0.05, 0.025]        # halving; ~800 / ~3300 / ~13000 nodes
DISORDER = [(0, 0.0), (30, 0.15), (60, 0.30)]   # (chaos %, jitter = chaos/200)
SEEDS = [0, 1, 2]
KEIG = 8                             # eigenpairs from ARPACK
NREPORT = 6                          # modes used for localisation / tables
SIGMA_SHIFT = -1.0                   # shift-invert target (left of the spectrum)
ANCHOR_D1 = 2.0 * np.pi ** 2         # Dirichlet lambda_1
ANCHOR_N2 = np.pi ** 2               # Neumann lambda_2 (lambda_1 = 0)


# ----------------------------------------------------------- fake unit-Laplacian field
def _zval(P):
    P = np.asarray(P, float)
    return np.zeros(P.shape[:-1])


def _zgrad(P):
    P = np.asarray(P, float)
    return np.zeros(P.shape)


def _onelap(P):
    P = np.asarray(P, float)
    return np.ones(P.shape[:-1])


UNIT_LAP = mf.Field("unit-laplacian", _zval, _zgrad, _onelap)


def nvec_from_b(b):
    """b[i] = 1/N_i for the unit-Laplacian fake field (BC data all zero)."""
    b = np.asarray(b, float)
    bad = np.abs(b) < 1e-300
    N = np.where(bad, 1.0, 1.0 / np.where(bad, 1.0, b))
    return N, int(bad.sum())


# ----------------------------------------------------------- assembly per configuration
def build(config, closure, surface_mode, dx, jit, seed):
    """Assemble one closed system; return (pos, A, A_scaled_neg, Nvec, bnd_mask)."""
    if config == "tank":
        pos = g2.tank_cloud(dx, cap.FILL_H, cap.TANK, jitter=jit, seed=seed)
        activation = "smoothstep" if surface_mode == "natural" else "rational"
        A, b, info = ps.assemble(pos, dx, UNIT_LAP, poly=cap.TANK, fill_h=cap.FILL_H,
                                 wall_proj="GGP", surface_mode=surface_mode,
                                 activation=activation, wall_closure=closure)
        Nvec = np.asarray(info["N"], float)
        mask = info["is_wall"] | info["is_surf"]
    else:
        pos = g2.jittered_box(dx, jit, seed)
        if config == "dirichlet":
            A, b, info = bvp.assemble(pos, dx, UNIT_LAP, SQUARE, "dirichlet")
        else:  # all-Neumann, no pin, no zero-mean constraint
            A, b, info = bvp.assemble(pos, dx, UNIT_LAP, SQUARE, "neumann",
                                      wall_closure=closure)
        Nvec, nbad = nvec_from_b(b)
        if nbad:
            print(f"    WARNING: {nbad} isolated/pinned rows (N_i set to 1)")
        mask = info["is_bnd"]
    if Nvec.min() <= 0.0:
        print(f"    WARNING: non-positive row normalisation, min N_i = {Nvec.min():.3e}")
    A = A.tocsr()
    As = (-sps.diags(Nvec) @ A).tocsc()
    return pos, A, As, Nvec, np.asarray(mask, bool)


# ----------------------------------------------------------- eigensolver + indicators
def small_eigs(As, k=KEIG, sigma=SIGMA_SHIFT):
    """k eigenpairs of the scaled operator nearest sigma (ARPACK shift-invert),
    returned sorted by |lambda| ascending."""
    rng = np.random.default_rng(0)
    v0 = rng.standard_normal(As.shape[0])
    vals, vecs = spla.eigs(As, k=k, sigma=sigma, which="LM", v0=v0, maxiter=5000)
    order = np.argsort(np.abs(vals))
    return vals[order], vecs[:, order]


def offdiag_stats(A, mask):
    """Sign statistics of the off-diagonal entries of the assembled A on rows in
    mask.  A_ij = W_ij w_ij (>0 for an M-matrix row); negative entries mark
    sign-flipped correction weights (or ghost-folded coefficients)."""
    A = A.tocsr()
    idx = np.where(mask)[0]
    rows_neg, nneg, noff = 0, 0, 0
    for i in idx:
        s, e = A.indptr[i], A.indptr[i + 1]
        cols, vals = A.indices[s:e], A.data[s:e]
        if len(vals) == 0:
            continue
        tol = 1e-13 * np.abs(vals).max()
        off = (cols != i) & (np.abs(vals) > tol)
        v = vals[off]
        noff += v.size
        kneg = int((v < 0.0).sum())
        nneg += kneg
        rows_neg += int(kneg > 0)
    nr = len(idx)
    return dict(rows=int(nr),
                frac_rows_with_neg=rows_neg / max(nr, 1),
                frac_neg_offdiag=nneg / max(noff, 1))


def bnd_mass_fracs(vecs, mask, nmodes=NREPORT):
    """Fraction of each eigenvector's l2 mass on boundary rows (nodes within h
    of the boundary, exactly the rows the assembly flags)."""
    m = np.abs(vecs[:, :nmodes]) ** 2
    tot = np.maximum(m.sum(0), 1e-300)
    return (m[mask].sum(0) / tot).tolist()


def _median_stats(dicts):
    keys = dicts[0].keys()
    return {k: float(np.median([d[k] for d in dicts])) for k in keys}


def run_case(config, closure, surface_mode, dx, jit, seeds):
    per = []
    for s in seeds:
        t0 = time.perf_counter()
        pos, A, As, Nvec, mask = build(config, closure, surface_mode, dx, jit, s)
        t_asm = time.perf_counter() - t0
        t0 = time.perf_counter()
        vals, vecs = small_eigs(As)
        t_eig = time.perf_counter() - t0
        per.append(dict(n=len(pos), vals=vals,
                        bmass=bnd_mass_fracs(vecs, mask), bfrac=float(mask.mean()),
                        st_b=offdiag_stats(A, mask), st_i=offdiag_stats(A, ~mask)))
        print(f"    seed {s}: N={len(pos)}  |lam|_min={np.abs(vals[0]):.4e}  "
              f"|lam|_2={np.abs(vals[1]):.4f}  (asm {t_asm:.1f}s eig {t_eig:.1f}s)")
    mags = np.median([np.abs(p["vals"]) for p in per], axis=0)
    res = np.median([np.real(p["vals"]) for p in per], axis=0)
    ims = np.median([np.abs(np.imag(p["vals"])) for p in per], axis=0)
    # imaginary-part flag: relative to |lambda|, skipping the (numerically zero)
    # null mode where the ratio is meaningless
    imfr = 0.0
    for p in per:
        v = p["vals"]
        scale = np.abs(v).max()
        for x in v[:NREPORT]:
            if abs(x) > 1e-8 * scale:
                imfr = max(imfr, abs(x.imag) / abs(x))
    return dict(
        N_nodes=int(np.median([p["n"] for p in per])),
        n_seeds=len(seeds),
        eig_mag=[float(x) for x in mags],
        eig_re=[float(x) for x in res],
        eig_im_abs=[float(x) for x in ims],
        lambda1=float(mags[0]), lambda2=float(mags[1]),
        max_imag_frac=float(imfr),
        bnd_mass_frac=[float(np.median([p["bmass"][m] for p in per]))
                       for m in range(NREPORT)],
        bnd_node_frac=float(np.median([p["bfrac"] for p in per])),
        neg_offdiag_boundary=_median_stats([p["st_b"] for p in per]),
        neg_offdiag_interior=_median_stats([p["st_i"] for p in per]),
    )


# ----------------------------------------------------------- self-check + cross-checks
def _selfcheck():
    """Verify the 1/b row-normalisation extraction against firm_core geometry,
    poisson's info['N'], and the exact Neumann null vector."""
    out = {}
    dx, jit = 0.05, 0.15
    pos = g2.jittered_box(dx, jit, 0)
    A, b, info = bvp.assemble(pos, dx, UNIT_LAP, SQUARE, "dirichlet")
    Nvec, _ = nvec_from_b(b)
    h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    checked, maxdiff = 0, 0.0
    for i in range(len(pos)):
        if info["is_bnd"][i] or len(nl[i]) < 3:
            continue
        xij = pos[nl[i]] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        gm = fc.geom_quantities(xij, w)
        maxdiff = max(maxdiff, abs(Nvec[i] - gm.N) / gm.N)
        checked += 1
        if checked >= 50:
            break
    out["nvec_vs_geometry_maxreldiff"] = float(maxdiff)

    posT = g2.tank_cloud(0.1, cap.FILL_H, cap.TANK, jitter=0.15, seed=0)
    At, bt, infoT = ps.assemble(posT, 0.1, UNIT_LAP, poly=cap.TANK, fill_h=cap.FILL_H,
                                wall_proj="GGP", surface_mode="exact", wall_closure="ghost")
    NT, _ = nvec_from_b(bt)
    out["tank_nvec_vs_infoN_maxreldiff"] = float(
        np.max(np.abs(NT - infoT["N"]) / np.abs(infoT["N"])))

    An, bn, _ = bvp.assemble(pos, dx, UNIT_LAP, SQUARE, "neumann", wall_closure="ghost")
    Nn, _ = nvec_from_b(bn)
    Asn = (-sps.diags(Nn) @ An.tocsr())
    ones = np.ones(An.shape[0])
    out["neumann_null_vector_residual_inf"] = float(
        np.abs(Asn @ ones).max() / np.abs(Asn.diagonal()).max())
    return out


def cross_checks():
    """ARPACK shift-invert vs dense eig vs (dense + sparse) SVD."""
    out = {"selfcheck": _selfcheck()}
    for label, config, closure in [("dirichlet_dx0.05_d30_s0", "dirichlet", "projection"),
                                   ("neumann_ghost_dx0.05_d30_s0", "neumann", "ghost")]:
        pos, A, As, Nvec, mask = build(config, closure, "exact", 0.05, 0.15, 0)
        vals, _ = small_eigs(As)
        dense = sla.eigvals(As.toarray())
        dsmall = dense[np.argsort(np.abs(dense))][:KEIG]
        arp = np.abs(vals)
        dm = np.sort(np.abs(dsmall))
        reldiff = float(np.max(np.abs(np.sort(arp)[:NREPORT] - dm[:NREPORT])
                               / np.maximum(dm[:NREPORT], 1e-12)))
        sv = sla.svdvals(As.toarray())
        out[label] = dict(
            N=len(pos),
            arpack_mags=[float(x) for x in arp],
            dense_mags=[float(x) for x in dm],
            arpack_vs_dense_maxreldiff_nonnull=reldiff if config == "dirichlet" else
            float(np.max(np.abs(np.sort(arp)[1:NREPORT] - dm[1:NREPORT])
                         / dm[1:NREPORT])),
            dense_sigma_min=float(sv[-1]), dense_sigma_min2=float(sv[-2]),
            note="sigma_min <= |lambda_1| for the non-symmetric matrix",
        )
    # sparse svds cross-check on the Dirichlet case (spectrum away from 0)
    try:
        pos, A, As, Nvec, mask = build("dirichlet", "projection", "exact", 0.025, 0.15, 0)
        u, s, vt = spla.svds(As, k=2, which="SM", maxiter=20000)
        out["svds_dirichlet_dx0.025_d30_s0"] = dict(
            sigma_min=float(np.min(s)), sigma_min2=float(np.max(s)))
    except Exception as exc:  # pragma: no cover
        out["svds_dirichlet_dx0.025_d30_s0"] = dict(failed=str(exc))
    return out


# ----------------------------------------------------------- sweeps
def sweep_box(config, closure):
    res = {}
    for dpct, jit in DISORDER:
        res_d = {}
        seeds = SEEDS if jit > 0 else [0]        # jitter 0 is seed-independent
        for dx in DXS:
            print(f"  {config}/{closure}  disorder {dpct}%  dx {dx}")
            leaf = run_case(config, closure, "exact", dx, jit, seeds)
            if config == "dirichlet":
                leaf["rel_dev_lambda1_vs_2pi2"] = (leaf["lambda1"] - ANCHOR_D1) / ANCHOR_D1
            else:
                leaf["rel_dev_lambda2_vs_pi2"] = (leaf["lambda2"] - ANCHOR_N2) / ANCHOR_N2
            res_d[f"dx{dx}"] = leaf
        res[f"disorder{dpct}"] = res_d
    return res


def sweep_tank():
    res = {}
    jit, dpct = 0.15, 30                          # the timing benchmark's disorder level
    for label, closure, smode in [("projection_exact", "projection", "exact"),
                                  ("ghost_exact", "ghost", "exact"),
                                  ("ghost_natural", "ghost", "natural")]:
        res_v = {}
        for dx in TANK_DXS:
            print(f"  tank/{label}  disorder {dpct}%  dx {dx}")
            res_v[f"dx{dx}"] = run_case("tank", closure, smode, dx, jit, SEEDS)
        res[label] = res_v
    return res


# ----------------------------------------------------------- figure
def make_figure(payload):
    import plotstyle
    plotstyle.apply(usetex=shutil.which("latex") is not None)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for k, (dpct, _) in enumerate(DISORDER):
        s = plotstyle.SERIES[k]
        lam1 = [payload["dirichlet_box"][f"disorder{dpct}"][f"dx{dx}"]["lambda1"]
                for dx in DXS]
        ax[0].semilogx(DXS, lam1, marker=s["marker"], color=s["color"], ms=s["ms"],
                       lw=1.5, ls="-", label=rf"disorder ${dpct}\%$")
    ax[0].axhline(ANCHOR_D1, color="k", ls="--", lw=1.0, alpha=0.6)
    ax[0].text(DXS[1], ANCHOR_D1, r"$2\pi^2$", va="bottom", ha="center", fontsize=12)
    ax[0].set_xlabel(r"$\Delta$")
    ax[0].set_ylabel(r"$\lambda_1$")
    ax[0].set_title(r"Dirichlet box: smallest eigenvalue")
    ax[0].invert_xaxis()
    plotstyle.grid(ax[0])
    ax[0].legend()

    for k, (dpct, _) in enumerate(DISORDER):
        s = plotstyle.SERIES[k]
        l2g = [payload["neumann_box_ghost"][f"disorder{dpct}"][f"dx{dx}"]["lambda2"]
               for dx in DXS]
        l2p = [payload["neumann_box_projection"][f"disorder{dpct}"][f"dx{dx}"]["lambda2"]
               for dx in DXS]
        ax[1].semilogx(DXS, l2g, marker=s["marker"], color=s["color"], ms=s["ms"],
                       lw=1.5, ls="-", label=rf"ghost ${dpct}\%$")
        ax[1].semilogx(DXS, l2p, marker=s["marker"], color=s["color"], ms=s["ms"],
                       lw=1.2, ls="--", mfc="none", label=rf"projection ${dpct}\%$")
    ax[1].axhline(ANCHOR_N2, color="k", ls="--", lw=1.0, alpha=0.6)
    ax[1].text(DXS[1], ANCHOR_N2, r"$\pi^2$", va="bottom", ha="center", fontsize=12)
    ax[1].set_xlabel(r"$\Delta$")
    ax[1].set_ylabel(r"$\lambda_2$")
    ax[1].set_title(r"Neumann box: first nonconstant eigenvalue")
    ax[1].invert_xaxis()
    plotstyle.grid(ax[1])
    ax[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG, dpi=160)
    print(f"figure -> {FIG}")


# ----------------------------------------------------------- io
def save(key, payload):
    d = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            d = json.load(f)
    d[key] = payload
    with open(OUT, "w") as f:
        json.dump(d, f, indent=2)
    print(f"saved '{key}' -> {OUT}")


def main():
    t0 = time.perf_counter()
    meta = dict(
        protocol=("eigenvalues of A_s = -diag(N_i) A (row-scaled closed operator, "
                  "approximates the continuous -Laplacian nodewise); ARPACK "
                  "shift-invert eigs(k=8, sigma=-1, which='LM'), sorted by |lambda|; "
                  "medians over seeds; disorder % = 2017 chaos, jitter = disorder/200"),
        anchors=dict(dirichlet_lambda1=ANCHOR_D1, neumann_lambda1=0.0,
                     neumann_lambda2=ANCHOR_N2),
        dxs_box=DXS, dxs_tank=TANK_DXS, seeds=SEEDS,
        disorder_pct=[d for d, _ in DISORDER],
        boundary_mask="rows within h=2.5dx of a wall/surface (assembly's is_bnd flags)",
        neg_offdiag=("off-diagonal entries of the assembled A (= W_ij w_ij; negative "
                     "entry <=> sign-flipped correction weight / ghost-folded coeff), "
                     "|entry| > 1e-13*rowmax"),
        tank=("wedge tank (capstone TANK, FILL_H=1.8), GGP walls, disorder 30% only; "
              "'ghost_natural' uses the detected smoothstep surface (capstone default), "
              "'*_exact' the exact surface closure"),
        norm="trb (suite default) on all rows",
    )
    payload = {"meta": meta}

    print("== cross-checks (dense eig / SVD vs ARPACK) ==")
    payload["cross_checks"] = cross_checks()
    for k, v in payload["cross_checks"].items():
        print(f"  {k}: {v}")

    print("== (a) Dirichlet unit box (value closure) ==")
    payload["dirichlet_box"] = sweep_box("dirichlet", "projection")
    print("== (b) all-Neumann unit box, projection closure ==")
    payload["neumann_box_projection"] = sweep_box("neumann", "projection")
    print("== (b) all-Neumann unit box, algebraic-ghost closure ==")
    payload["neumann_box_ghost"] = sweep_box("neumann", "ghost")
    print("== (c) wedge tank, full closure stack ==")
    payload["wedge_tank"] = sweep_tank()

    save("spectral", payload)
    make_figure(payload)
    print(f"total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
