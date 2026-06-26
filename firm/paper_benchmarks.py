"""
Benchmark battery for the FIRM boundary-closure paper (JCP follow-up).

Each function returns structured results (errors, convergence orders) used by both
the printed report and ``make_figures.py``. All solves go through the validated
``firm_core`` operators (via ``bvp.py``/``poisson.py``) or the GFDM baseline
(``gfdm.py``); the 2017 chaos/refinement protocol is applied through
``convergence.py``.

  B1  square Dirichlet (Franke / trig)          -- FIRM vs GFDM interior accuracy
  NB  all-Neumann box                            -- FIRM projection/ghost vs GFDM
                                                    constraint-row / penalty (headline baseline)
  B2  curved (star) Dirichlet & Neumann          -- value closure + projection/ghost/GFDM on a curve
  B3  flower Robin (du/dn + alpha u = f)         -- the Robin (diagonal+RHS) closure
  B5  tank free-surface detection                -- geometry-detected, tuning-free
  B6  non-orthogonal wedge                        -- GGP vs AN; projection vs ghost (capstone)
  B7  kernel/support + jitter robustness          -- compare_kernels / compare_normalization

Run:  python3 paper_benchmarks.py [name ...] [--full]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import bvp
import gfdm
import fi
import poisson as ps
from convergence import refine, sweep_chaos, jitter_for_chaos, print_table, CHAOS
from testkit import observed_order

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
DXS = [0.05, 0.035, 0.025, 0.018]
DXS_COARSE = [0.06, 0.045, 0.033]
DXS_FRANKE = [0.045, 0.032, 0.022, 0.016]
DXS_NEU = [0.045, 0.034, 0.025, 0.018]


def _interior_relL2(p, pe, mask, mean_remove=False):
    e = (p - pe)[mask]
    r = pe[mask]
    if mean_remove:
        e = e - e.mean(); r = r - r.mean()
    return float(np.linalg.norm(e) / max(np.linalg.norm(r), 1e-30))


# ====================================== straight Neumann edge (normalisation study)
def _neu_edge(dx, seed, method):
    """Square with a straight Neumann edge at y=0 (Dirichlet on the other three
    edges, so the problem is well-posed). method in
    {'projection','ghost-trace','ghost-denom'}. Returns the Neumann-strip rel-L2."""
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve
    field = mf.complex_field(np.pi, 0.3)
    pos = g2.jittered_box(dx, 0.30, seed)
    n = len(pos); h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    isN = (pos[:, 1] < h) & (pos[:, 0] > h) & (pos[:, 0] < 1 - h)
    isD = (~isN) & ((pos[:, 0] < h) | (pos[:, 0] > 1 - h) | (pos[:, 1] > 1 - h) | (pos[:, 1] < h))
    nrm = np.array([0.0, -1.0])
    R, C, D = [], [], []; b = np.zeros(n)
    for i in range(n):
        if isD[i]:
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        nb = nl[i]; xij = pos[nb] - pos[i]; w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        if isN[i]:
            foot = pos[i].copy(); foot[1] = 0.0; gf = float(field.grad(foot) @ nrm)
            if method.startswith("ghost"):
                nm = "denom" if method.endswith("denom") else "trb"
                src, coeff, inc, N = fc.mirror_ghost_terms(
                    xij, w, np.array([nrm]), np.array([foot - pos[i]]), np.array([gf]), h, norm=nm)
                diag = 0.0
                for m in range(len(coeff)):
                    col = i if src[m] < 0 else int(nb[src[m]])
                    R.append(i); C.append(col); D.append(coeff[m]); diag -= coeff[m]; b[i] -= coeff[m] * inc[m]
                R.append(i); C.append(i); D.append(diag); b[i] += field.laplacian(pos[i]) / N; continue
            gm = fc.geom_quantities(xij, w)
            P, Nm, Gi = fc.proj_GGP(np.array([nrm]))
            V = gm.B @ (P @ gm.o); wij = fc.correction_weights(xij, V); ws = float((w * wij).sum())
            for k, j in enumerate(nb):
                R.append(i); C.append(int(j)); D.append(w[k] * wij[k])
            R.append(i); C.append(i); D.append(-ws)
            b[i] = field.laplacian(pos[i]) / gm.N + fc.wall_flux_rhs_GGP(Nm, Gi, gm.o, [gf]); continue
        gm = fc.geom_quantities(xij, w); V = gm.B @ gm.o; wij = fc.correction_weights(xij, V)
        ws = float((w * wij).sum())
        for k, j in enumerate(nb):
            R.append(i); C.append(int(j)); D.append(w[k] * wij[k])
        R.append(i); C.append(i); D.append(-ws); b[i] = field.laplacian(pos[i]) / gm.N
    A = sps.csr_matrix((D, (R, C)), shape=(n, n)); p = spsolve(A, b); pe = field.value(pos)
    return float(np.linalg.norm((p - pe)[isN]) / np.linalg.norm(pe[isN]))


def neumann_straight(dxs=DXS_NEU, seeds=None):
    seeds = seeds if seeds is not None else list(range(12))
    out = {}
    for m in ("projection", "ghost-trace", "ghost-denom"):
        e = [float(np.median([_neu_edge(dx, s, m) for s in seeds])) for dx in dxs]
        out[m] = (e, observed_order(dxs, e))
    return out


# ============================================================ Franke operator test
def _franke_pointwise(dx, jitter, seed, method):
    """Pointwise Laplacian approximation error of the Franke function on the interior
    of the unit box. method in {'new','denom','gfdm','fi'}."""
    field = mf.franke_field()
    pos = g2.jittered_box(dx, jitter, seed)
    h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    mask = g2.box_interior_mask(pos, h)
    num, ref = [], []
    for i in np.where(mask)[0]:
        nb = nl[i]
        if len(nb) < (6 if method in ("gfdm", "fi") else 3):
            continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        fij = field.value(pos[nb]) - field.value(pos[i])
        if method == "gfdm":
            d, _ = gfdm.gfdm_row(xij, w)
            lap = float(d @ fij)
        elif method == "fi":
            d, _ = fi.fi_row(xij, w)
            lap = float(d @ fij)
        else:
            gm = fc.geom_quantities(xij, w)
            wij = fc.correction_weights(xij, gm.B @ gm.o)
            Nn = fc.laplacian_normalization(gm, xij, w, wij,
                                            mode=("denom" if method == "denom" else "trb"))
            lap = Nn * float((w * fij * wij).sum())
        num.append(lap)
        ref.append(field.laplacian(pos[i]))
    num, ref = np.array(num), np.array(ref)
    return float(np.linalg.norm(num - ref) / np.linalg.norm(ref))


def franke_operator(dxs=DXS, seeds=None, chaos=(0.0, 0.30, 0.60, 0.90)):
    seeds = seeds if seeds is not None else list(range(12))
    out = {}
    for c in chaos:
        jit = jitter_for_chaos(c) if c > 0 else 0.0
        ss = [0] if c == 0 else seeds
        d = {}
        for m in ("new", "denom", "gfdm", "fi"):
            errs = [float(np.median([_franke_pointwise(dx, jit, s, m) for s in ss])) for dx in dxs]
            d[m] = (errs, observed_order(dxs, errs))
        out[c] = d
    return out


# ============================================================ B1 square Dirichlet
def _firm_dirichlet_box(dx, jitter, seed, field):
    pos = g2.jittered_box(dx, jitter, seed)
    A, b, info = bvp.assemble(pos, dx, field, SQUARE, "dirichlet")
    p = bvp.solve(A, b)
    mask = g2.box_interior_mask(pos, info["h"])
    return _interior_relL2(p, field.value(pos), mask)


def _gfdm_dirichlet_box(dx, jitter, seed, field):
    pos = g2.jittered_box(dx, jitter, seed)
    h = 2.5 * dx
    dirm = ~g2.box_interior_mask(pos, h)
    A, b, info = gfdm.assemble_gfdm(pos, dx, field, dir_mask=dirm)
    p = gfdm.solve(A, b)
    mask = g2.box_interior_mask(pos, h)
    return _interior_relL2(p, field.value(pos), mask)


def _fi_dirichlet_box(dx, jitter, seed, field):
    pos = g2.jittered_box(dx, jitter, seed)
    h = 2.5 * dx
    dirm = ~g2.box_interior_mask(pos, h)
    A, b, info = fi.assemble_fi(pos, dx, field, dir_mask=dirm)
    p = fi.solve(A, b)
    mask = g2.box_interior_mask(pos, h)
    return _interior_relL2(p, field.value(pos), mask)


def b1_dirichlet(dxs=DXS, seeds=None, field=None):
    field = field or mf.trig_field(np.pi)
    seeds = seeds if seeds is not None else list(range(12))
    out = {}
    for c in CHAOS:
        jit = jitter_for_chaos(c)
        ef, of = refine(lambda dx, j, s: _firm_dirichlet_box(dx, j, s, field), dxs, jit, seeds)
        eg, og = refine(lambda dx, j, s: _gfdm_dirichlet_box(dx, j, s, field), dxs, jit, seeds)
        ei, oi = refine(lambda dx, j, s: _fi_dirichlet_box(dx, j, s, field), dxs, jit, seeds)
        out[c] = dict(firm=(ef, of), gfdm=(eg, og), fi=(ei, oi))
    return out


# ====================================================== all-Neumann box (headline)
def _neumann_box(dx, jitter, seed, field, method):
    """method in {'firm-proj','firm-ghost','gfdm-constraint','gfdm-penalty'}."""
    pos = g2.jittered_box(dx, jitter, seed)
    h = 2.5 * dx
    pin = int(np.argmin(np.linalg.norm(pos - 0.5, axis=1)))
    if method.startswith("firm"):
        wc = "ghost" if method.endswith("ghost") else "projection"
        A, b, info = bvp.assemble(pos, dx, field, SQUARE, "neumann", wall_closure=wc, pin=pin)
        bnd = info["is_bnd"]
    else:
        nm = "penalty" if method.endswith("penalty") else "constraint"
        A, b, info = gfdm.assemble_gfdm(pos, dx, field, poly=SQUARE, neumann=nm, pin=pin)
        bnd = info["is_wall"]
    p = (bvp.solve if method.startswith("firm") else gfdm.solve)(A, b)
    mask = g2.box_interior_mask(pos, h)
    return _interior_relL2(p, field.value(pos), mask, mean_remove=True)


def neumann_baseline(dxs=DXS, seeds=None, field=None, chaos=(0.30, 0.60)):
    field = field or mf.complex_field(np.pi, 0.3)
    seeds = seeds if seeds is not None else list(range(12))
    methods = ["firm-proj", "firm-ghost", "gfdm-constraint", "gfdm-penalty"]
    out = {}
    for c in chaos:
        jit = jitter_for_chaos(c)
        out[c] = {m: refine(lambda dx, j, s, mm=m: _neumann_box(dx, j, s, field, mm), dxs, jit, seeds)
                  for m in methods}
    return out


# =================================================== B2 curved (star) Dirichlet/Neumann
def _star(n=720):
    return g2.star_polygon(n=n, r0=0.5, amp=0.2, k=5, center=(0.5, 0.5))


def b2_curved(dxs=DXS, seeds=None, field=None, chaos=(0.30, 0.60)):
    field = field or mf.trig_field(np.pi)
    seeds = seeds if seeds is not None else list(range(10))
    star = _star()
    seg = g2.polygon_segments(star)

    def firm_dir(dx, j, s):
        pos = g2.polygon_cloud(star, dx, j, s)
        A, b, info = bvp.assemble(pos, dx, field, star, "dirichlet")
        return _interior_relL2(bvp.solve(A, b), field.value(pos), ~info["is_bnd"])

    def gfdm_dir(dx, j, s):
        pos = g2.polygon_cloud(star, dx, j, s)
        h = 2.5 * dx
        # Dirichlet mask = particles within h of the boundary (pinned exactly in GFDM)
        dirm = np.array([g2.nearest_segment(p, *seg, h) is not None for p in pos])
        A, b, info = gfdm.assemble_gfdm(pos, dx, field, dir_mask=dirm)
        return _interior_relL2(gfdm.solve(A, b), field.value(pos), ~dirm)

    def firm_neu(dx, j, s, wc):
        pos = g2.polygon_cloud(star, dx, j, s)
        pin = int(np.argmin(np.linalg.norm(pos - 0.5, axis=1)))
        A, b, info = bvp.assemble(pos, dx, field, star, "neumann", wall_closure=wc, pin=pin)
        return _interior_relL2(bvp.solve(A, b), field.value(pos), ~info["is_bnd"], mean_remove=True)

    def gfdm_neu(dx, j, s):
        pos = g2.polygon_cloud(star, dx, j, s)
        pin = int(np.argmin(np.linalg.norm(pos - 0.5, axis=1)))
        A, b, info = gfdm.assemble_gfdm(pos, dx, field, poly=star, neumann="constraint", pin=pin)
        return _interior_relL2(gfdm.solve(A, b), field.value(pos), ~info["is_wall"], mean_remove=True)

    out = {}
    for c in chaos:
        jit = jitter_for_chaos(c)
        out[c] = dict(
            dir_firm=refine(firm_dir, dxs, jit, seeds),
            dir_gfdm=refine(gfdm_dir, dxs, jit, seeds),
            neu_proj=refine(lambda dx, j, s: firm_neu(dx, j, s, "projection"), dxs, jit, seeds),
            neu_ghost=refine(lambda dx, j, s: firm_neu(dx, j, s, "ghost"), dxs, jit, seeds),
            neu_gfdm=refine(gfdm_neu, dxs, jit, seeds),
        )
    return out


# =================================================================== B3 flower Robin
def b3_robin(dxs=DXS, seeds=None, field=None, alpha=1.0, chaos=CHAOS):
    field = field or mf.trig_field(np.pi)
    seeds = seeds if seeds is not None else list(range(10))
    flower = g2.flower_polygon(n=720, base=0.5, amp=0.08, k=8, center=(0.5, 0.5))

    def firm_robin(dx, j, s):
        pos = g2.polygon_cloud(flower, dx, j, s)
        A, b, info = bvp.assemble(pos, dx, field, flower, "robin", robin_alpha=alpha)
        return _interior_relL2(bvp.solve(A, b), field.value(pos), ~info["is_bnd"])

    return {c: refine(firm_robin, dxs, jitter_for_chaos(c), seeds) for c in chaos}


# ====================================================== B5 free-surface detection
TANK = np.array([[0.0, 1.0], [1.2, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
FILL_H = 1.8


def b5_surface_detection(dx=0.045, jitter=0.30, seed=7, field=None):
    """Detector quality at one resolution: interior vs surface sigma separation and the
    recovered-normal error against the true (vertical) free-surface normal."""
    field = field or mf.complex_field(np.pi, 0.3)
    pos = g2.tank_cloud(dx, FILL_H, TANK, jitter=jitter, seed=seed)
    h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    lam = np.full(len(pos), np.nan)
    sig = np.zeros(len(pos))
    nerr = []
    is_surf = (FILL_H - pos[:, 1] > 0) & (FILL_H - pos[:, 1] < h)
    for i in range(len(pos)):
        nb = nl[i]
        if len(nb) < 3:
            continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        # pass wall data so the detector sees the WALL-PROJECTED offset (Sec 3.2):
        # the surface direction is the deficiency the walls cannot explain.
        walls = None
        near = g2.nearby_walls(pos[i], TANK, h)
        if near:
            walls = dict(normals=np.array([n for _, n, _ in near]),
                         deltas=np.array([d for d, _, _ in near]),
                         g=np.zeros(len(near)), h_w=h)
        loc = fc.particle_operator(pos[i], xij, w, walls=walls, surface=dict(mode="natural"),
                                   dx=dx, activation="smoothstep")
        lam[i] = fc.lambda_detect(loc.r, dx)
        sig[i] = loc.sigma
        if is_surf[i] and loc.n_s is not None:
            nerr.append(abs(loc.n_s @ np.array([0.0, 1.0]) - 1.0))
    interior = ~is_surf & np.isfinite(lam)
    # the surface BAND is h thick; the genuine free-surface layer is its outermost ring
    top = is_surf & (FILL_H - pos[:, 1] < 0.5 * h)
    return dict(
        lam_interior_p90=float(np.nanpercentile(lam[interior], 90)),
        lam_surface_top_med=float(np.nanmedian(lam[top])),
        sigma_interior_max=float(np.nanmax(sig[interior])),     # false activation (want 0)
        sigma_surface_max=float(np.nanmax(sig[is_surf])),        # genuine layer activates
        surf_frac_active=float(np.mean(sig[is_surf] > 0.05)),
        normal_err_med=float(np.median(nerr)) if nerr else float("nan"),
        N=len(pos),
    )


def b5_surface_convergence(dxs=DXS_COARSE, seeds=None, field=None):
    """Natural (detected) vs exact (prescribed) free surface -- convergence of the
    surface-region error; the detected closure should match the prescribed reference."""
    field = field or mf.complex_field(np.pi, 0.3)
    seeds = seeds if seeds is not None else list(range(6))

    def run(dx, seed, surface_mode):
        pos = g2.tank_cloud(dx, FILL_H, TANK, jitter=0.30, seed=seed)
        act = "smoothstep" if surface_mode == "natural" else "rational"
        A, b, info = ps.assemble(pos, dx, field, poly=TANK, fill_h=FILL_H, wall_proj="GGP",
                                 surface_mode=surface_mode, activation=act, wall_closure="ghost")
        p = ps.solve(A, b)
        pe = field.value(pos)
        sc = max(np.max(np.abs(pe)), 1e-30)
        m = info["is_surf"]
        return float(np.sqrt(np.mean(((p - pe)[m]) ** 2)) / sc)

    out = {}
    for mode in ("exact", "natural"):
        errs = [float(np.median([run(dx, s, mode) for s in seeds])) for dx in dxs]
        out[mode] = (errs, observed_order(dxs, errs))
    return out


# ================================================================== main / report
def _print_neumann(res, dxs):
    for c, d in res.items():
        print(f"\n  all-Neumann box, chaos {int(c*100)}% (rel-L2 interior, mean-removed):")
        labs = ["firm-proj", "firm-ghost", "gfdm-constraint", "gfdm-penalty"]
        print("    dx      " + "".join(f"{l:>17}" for l in labs))
        for i, dx in enumerate(dxs):
            print(f"    {dx:6.4f}  " + "".join(f"{d[l][0][i]:>17.3e}" for l in labs))
        print("    order   " + "".join(f"{d[l][1]:>17.2f}" for l in labs))


def main(argv):
    full = "--full" in argv
    names = [a for a in argv if not a.startswith("--")] or ["all"]
    seeds = list(range(20)) if full else list(range(8))
    dxs = DXS if full else DXS_COARSE
    want = lambda k: "all" in names or k in names

    if want("b1"):
        print("\n=== B1  square Dirichlet (trig), FIRM vs GFDM ===")
        r = b1_dirichlet(dxs, seeds)
        for c in CHAOS:
            print_table(f"chaos {int(c*100)}%", dxs,
                        {"FIRM": r[c]["firm"], "GFDM": r[c]["gfdm"], "FI": r[c]["fi"]})
    if want("nb") or want("neumann"):
        print("\n=== Neumann baseline: FIRM projection/ghost vs GFDM constraint/penalty ===")
        _print_neumann(neumann_baseline(dxs, seeds), dxs)
    if want("b2"):
        print("\n=== B2  curved (star) Dirichlet & Neumann ===")
        r = b2_curved(dxs, seeds)
        for c in (0.30, 0.60):
            print_table(f"chaos {int(c*100)}%", dxs, r[c],
                        methods=["dir_firm", "dir_gfdm", "neu_proj", "neu_ghost", "neu_gfdm"])
    if want("b3"):
        print("\n=== B3  flower Robin (du/dn + u = f), FIRM ===")
        r = b3_robin(dxs, seeds)
        print_table("Robin alpha=1", dxs, {f"chaos{int(c*100)}": r[c] for c in CHAOS})
    if want("b5"):
        print("\n=== B5  free-surface detection (tuning-free) ===")
        d = b5_surface_detection()
        for k, v in d.items():
            print(f"    {k:22s} {v}")
        print("\n  natural vs exact surface convergence (surface-region RMS):")
        sc = b5_surface_convergence(dxs, seeds)
        print_table("surface", dxs, {"exact": sc["exact"], "natural": sc["natural"]})


if __name__ == "__main__":
    main(sys.argv[1:])
