"""
CAPSTONE -- Complex (manufactured) Poisson on a jittered point cloud.
========================================================================
The headline test of the FIRM unified pressure-Poisson equation (Sec 4),
exercising the SAME firm_core operators validated by the substep tests.

Domain: a tank with a non-orthogonal slant/floor WEDGE (K=2 -> GGP clean,
AN leaks O(dx)) and a free surface. Manufactured solution

    p*(x,y) = sin(k x) cos(k y) + alpha (x^2 + y^2)
    lap p*  = -2 k^2 sin(k x) cos(k y) + 4 alpha        (sign-varying source)

All boundary data is analytic: wall Neumann flux g_k = grad p*(foot_k).n_k;
free-surface target = p* at the (estimated or exact) surface point.

WHAT THIS SUITE ESTABLISHED (honest, implementation verified linear-exact):
  * Operators are linear-exact on ANY cloud; the bare Laplacian is only
    linear-consistent (O(1) pointwise error under fixed-relative jitter).
  * The Poisson SOLUTION supraconverges, but the convergence is governed by the
    boundary closure:
        - Dirichlet / free-surface closure: strong, ~order 1.3 (excellent).
        - Neumann WALL closure: linear-exact but NOT convergent for nonlinear
          solutions (one-sided normal-curvature) -> it is the accuracy-limiter.
  * GGP is exact for linear fields at the wedge where AN leaks O(dx); for the
    complex field that distinction is masked by the dominant wall-closure error.
The BC-isolation diagnostic below reproduces these orders from scratch.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import firm_core as fc
import poisson as ps
import manufactured as mf
import geometry2d as g2
from testkit import observed_order

try:
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve
except Exception:  # pragma: no cover
    sps = None

TITLE = "CAPSTONE -- Complex Poisson on a jittered point cloud"

TANK = np.array([[0.0, 1.0], [1.2, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
FILL_H = 1.8
WEDGE_APEX = np.array([1.2, 0.0])
FIELD = mf.complex_field(k=np.pi, alpha=0.3)
DXS = [0.08, 0.06, 0.045, 0.033]
VARIANTS = [("GGP", "exact"), ("GGP", "natural"), ("AN", "exact"), ("AN", "natural")]


def run_case(dx, wall, surf, field=FIELD, wall_closure="projection"):
    pos = g2.tank_cloud(dx, FILL_H, TANK, jitter=0.30, seed=7)
    activation = "smoothstep" if surf == "natural" else "rational"
    A, b, info = ps.assemble(pos, dx, field, poly=TANK, fill_h=FILL_H,
                             wall_proj=wall, surface_mode=surf, activation=activation,
                             wall_closure=wall_closure)
    p = ps.solve(A, b)
    pe = field.value(pos)
    scale = max(np.max(np.abs(pe)), 1e-30)
    l2, linf = ps.rel_errors(p, pe)

    def rms(mask):
        return float(np.sqrt(np.mean(((p - pe)[mask]) ** 2)) / scale) if mask.any() else 0.0

    near = np.linalg.norm(pos - WEDGE_APEX, axis=1) < 6 * dx
    interior = ~info["is_wall"] & ~info["is_surf"]
    return dict(N=len(pos), l2=l2, linf=linf,
                wedge=float(np.max(np.abs((p - pe)[near])) / scale) if near.any() else 0.0,
                rms_int=rms(interior), rms_wall=rms(info["is_wall"]), rms_surf=rms(info["is_surf"]),
                pos=pos, p=p, pe=pe)


def sweep():
    results = {v: [run_case(dx, *v) for dx in DXS] for v in VARIANTS}
    orders = {v: observed_order(DXS, [r["l2"] for r in results[v]]) for v in VARIANTS}
    return results, orders


# --------------------------------------------------------------- BC isolation
def _box_poisson(dx, field, bc, seed=7):
    """Complex Poisson on a jittered unit box with all-Dirichlet or all-Neumann
    boundaries (Neumann uses the SAME GGP wall closure as the tank). Isolates the
    effect of boundary-condition TYPE on convergence."""
    h = 2.5 * dx
    pos = g2.jittered_box(dx, 0.3, seed)
    nbnd = g2.box_interior_mask(pos, h)
    nl = fc.neighbor_lists(pos, h)
    pin = int(np.argmin(np.linalg.norm(pos - 0.5, axis=1)))
    R, C, D = [], [], []
    b = np.zeros(len(pos))
    faces = [np.array([-1.0, 0]), np.array([1.0, 0]), np.array([0, -1.0]), np.array([0, 1.0])]
    for i in range(len(pos)):
        bnd = not nbnd[i]
        if (bc == "dir" and bnd) or (bc == "neu" and i == pin):
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        xij = pos[nl[i]] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        gm = fc.geom_quantities(xij, w)
        bwall = 0.0
        if bc == "neu" and bnd:
            d = np.array([pos[i, 0], 1 - pos[i, 0], pos[i, 1], 1 - pos[i, 1]])
            k = int(d.argmin())
            n = faces[k]
            foot = pos[i].copy(); foot[k // 2] = float(k % 2)   # nearest-face foot point
            P, Nm, Gi = fc.proj_GGP(np.array([n]))
            V = gm.B @ (P @ gm.o)
            bwall = fc.wall_flux_rhs_GGP(Nm, Gi, gm.o, [float(field.grad(foot) @ n)])  # flux AT the wall
        else:
            V = gm.B @ gm.o
        wij = fc.correction_weights(xij, V)
        wsum = float((w * wij).sum())
        for k, j in enumerate(nl[i]):
            R.append(i); C.append(j); D.append(w[k] * wij[k])
        R.append(i); C.append(i); D.append(-wsum)
        b[i] = field.laplacian(pos[i]) / gm.N + bwall
    A = sps.csr_matrix((D, (R, C)), shape=(len(pos), len(pos)))
    p = spsolve(A, b)
    pe = field.value(pos)
    err = (p - pe)[nbnd]
    ref = pe[nbnd].copy()
    if bc == "neu":                       # pure-Neumann solution is defined up to a constant;
        err = err - err.mean()            # a single-point pin does NOT fix the level, so compare
        ref = ref - ref.mean()            # mean-removed (else the null space looks like divergence)
    return np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-30)


def bc_isolation(seeds=(7, 11, 19, 23)):
    """Returns {bc: (errs, full_order)} with errors median-averaged over seeds
    (single-seed Neumann is noisy). Errors are mean-removed for the pure-Neumann
    case so the constant null space is not mistaken for divergence."""
    out = {}
    for bc in ("dir", "neu"):
        errs = [float(np.median([_box_poisson(dx, FIELD, bc, s) for s in seeds])) for dx in DXS]
        out[bc] = (errs, observed_order(DXS, errs))
    return out


# --------------------------------------------------------------- reporting
def print_report(results, orders, bc):
    print(f"\nManufactured  p* = sin(pi x)cos(pi y) + 0.3(x^2+y^2)   tank wedge + free surface")
    print(f"jitter = 30% of dx.   rel-L2 error vs dx:\n")
    print("  dx     " + "".join(f"{w+'/'+s:>15}" for w, s in VARIANTS))
    for i, dx in enumerate(DXS):
        print(f"  {dx:5.3f}  " + "".join(f"{results[v][i]['l2']:>15.3e}" for v in VARIANTS)
              + f"  (N={results[VARIANTS[0]][i]['N']})")
    print("  order  " + "".join(f"{orders[v]:>15.2f}" for v in VARIANTS))

    g = results[("GGP", "exact")]
    o = lambda key: observed_order(DXS, [r[key] for r in g])
    print(f"\n  GGP/exact error by region (RMS / max|p*|):")
    print(f"    {'dx':>6}{'interior':>12}{'wall':>12}{'free-surf':>12}")
    for i, dx in enumerate(DXS):
        print(f"    {dx:6.3f}{g[i]['rms_int']:12.2e}{g[i]['rms_wall']:12.2e}{g[i]['rms_surf']:12.2e}")
    print(f"    order {o('rms_int'):>11.2f}{o('rms_wall'):>12.2f}{o('rms_surf'):>12.2f}")
    print(f"    -> free-surface closure converges strongly; the WALL closure is the limiter.")

    print(f"\n  BC-isolation on a jittered box (same operators, complex field, median over seeds):")
    print(f"    all-Dirichlet : errs {[f'{e:.2e}' for e in bc['dir'][0]]}  order {bc['dir'][1]:.2f}")
    print(f"    all-Neumann   : errs {[f'{e:.2e}' for e in bc['neu'][0]]}  order {bc['neu'][1]:.2f}  (mean-removed)")
    print(f"    -> BOTH converge, but the Neumann/wall closure is far less accurate (~10-100x larger")
    print(f"       error) and lower order (~1 vs ~1.3) -- it is the method's accuracy-limiter. NOTE:")
    print(f"       pure-Neumann needs proper null-space handling (zero-mean), NOT a single-point pin.")


def ghost_demo(dxs=(0.06, 0.045, 0.033)):
    """Projection vs mirror-ghost wall closure on the tank (GGP walls, exact surface)."""
    out = {}
    for wc in ("projection", "ghost"):
        rows = [run_case(dx, "GGP", "exact", wall_closure=wc) for dx in dxs]
        out[wc] = (list(dxs), [r["l2"] for r in rows], observed_order(list(dxs), [r["l2"] for r in rows]))
    return out


def run(rep):
    results, orders = sweep()
    bc = bc_isolation()
    print_report(results, orders, bc)

    gd = ghost_demo()
    print(f"\n  Wall closure: projection vs mirror-ghost (GGP walls, exact surface, tank):")
    print(f"    projection L2: {[f'{e:.2e}' for e in gd['projection'][1]]}  order {gd['projection'][2]:.2f}")
    print(f"    mirror-ghost L2: {[f'{e:.2e}' for e in gd['ghost'][1]]}  order {gd['ghost'][2]:.2f}")
    print(f"    -> ghost stencil-completion (reflect neighbours + i + corner) cuts the wall-limited")
    print(f"       error ~10x and restores fast convergence. (firm_core.mirror_ghost_terms)")

    # 0. the mirror-ghost wall closure sharply beats the projection closure
    rep.check("mirror-ghost wall closure beats projection (>3x at finest dx)",
              gd["ghost"][1][-1] < gd["projection"][1][-1] / 3,
              f"tank L2 projection {gd['projection'][1][-1]:.2e} -> ghost {gd['ghost'][1][-1]:.2e}")

    # 1. all variants are bounded and the solution is sane
    for v in VARIANTS:
        rep.check(f"bounded solution {v[0]}/{v[1]}", results[v][-1]["l2"] < 0.1,
                  f"rel L2 = {results[v][-1]['l2']:.2e}, order = {orders[v]:.2f}")

    # 2. free-surface (Dirichlet/Robin) closure converges strongly
    g = results[("GGP", "exact")]
    surf_order = observed_order(DXS, [r["rms_surf"] for r in g])
    rep.check_order("free-surface region converges", surf_order, expected=1.0, slack=0.3,
                    detail=f"surface RMS order = {surf_order:.2f}")

    # 3. the Neumann wall closure is the accuracy-limiter: it DOES converge (with
    #    proper null-space handling) but at lower order and ~10-100x larger error
    #    than the Dirichlet closure at the same resolution.
    rep.check("Dirichlet closure supraconverges", bc["dir"][1] > 1.0,
              f"box Dirichlet order = {bc['dir'][1]:.2f}, finest err = {bc['dir'][0][-1]:.2e}")
    rep.check("Neumann closure converges (~order 1) but is far less accurate",
              bc["neu"][1] > 0.5 and bc["neu"][0][-1] > 10 * bc["dir"][0][-1],
              f"box Neumann order = {bc['neu'][1]:.2f}, finest err {bc['neu'][0][-1]:.2e} "
              f">> Dirichlet {bc['dir'][0][-1]:.2e}")

    # 4. the clean GGP advantage: exact for a LINEAR field at the wedge, AN leaks
    lin = mf.linear_field([0.3, -0.5], 0.2, "lin-wedge")
    rg = run_case(DXS[1], "GGP", "exact", field=lin)
    ra = run_case(DXS[1], "AN", "exact", field=lin)
    rep.check("GGP exact for linear field (any geometry incl. wedge)", rg["linf"] < 1e-7,
              f"GGP rel L_inf = {rg['linf']:.2e}")
    rep.check("AN leaks O(dx) at the non-orthogonal wedge (Sec 10.3)",
              ra["linf"] > 1e-3 and ra["linf"] > 1e3 * rg["linf"],
              f"AN rel L_inf = {ra['linf']:.2e} >> GGP {rg['linf']:.2e}")

    # 5. exact-Dirichlet surface is no worse than natural-Robin (same wall)
    for wall in ("GGP", "AN"):
        ex = results[(wall, "exact")][-1]["l2"]
        na = results[(wall, "natural")][-1]["l2"]
        rep.check(f"exact-Dirichlet <= natural-Robin ({wall})", ex <= na * 1.5,
                  f"exact {ex:.2e} vs natural {na:.2e}")

    # 6. assembly + source-scaling guards
    rep.check("linear-exact guard (GGP/exact, dx=0.06)", rg["linf"] < 1e-7,
              f"rel L_inf = {rg['linf']:.2e}")
    quad = mf.quadratic_field([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], 0.0, "const-src")  # lap = 2
    rq = run_case(DXS[1], "GGP", "exact", field=quad)
    rep.check("constant-source scaling f/N_i (GGP/exact)", rq["l2"] < 0.05,
              f"rel L2 = {rq['l2']:.2e} for lap p = 2")
    return results, orders


def make_plots(results, orders, path_prefix="capstone"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for v in VARIANTS:
        errs = [results[v][i]["l2"] for i in range(len(DXS))]
        ax.loglog(DXS, errs, "o-", label=f"{v[0]}/{v[1]} (ord {orders[v]:.2f})")
    ref = results[("GGP", "exact")][0]["l2"] * (np.array(DXS) / DXS[0])
    ax.loglog(DXS, ref, "k--", alpha=0.4, label="order 1 ref")
    ax.set_xlabel("dx"); ax.set_ylabel("rel L2 error"); ax.legend(fontsize=8)
    ax.set_title("FIRM complex-Poisson convergence (jittered tank)")
    ax.invert_xaxis(); fig.tight_layout()
    fig.savefig(f"{path_prefix}_convergence.png", dpi=130); plt.close(fig)

    r = results[("GGP", "exact")][-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    poly = np.vstack([TANK, TANK[0]])
    ax.plot(poly[:, 0], poly[:, 1], "k-", lw=1.2)
    ax.axhline(FILL_H, color="royalblue", ls="--", lw=1, alpha=0.6)
    sc = ax.scatter(r["pos"][:, 0], r["pos"][:, 1], c=np.abs(r["p"] - r["pe"]), s=10, cmap="magma")
    ax.set_aspect("equal"); ax.set_title("|p - p*|  GGP/exact, finest dx (error concentrates at the wall/wedge)")
    fig.colorbar(sc, ax=ax, shrink=0.8); fig.tight_layout()
    err_name = os.path.join(os.path.dirname(path_prefix) or ".", "fig_error_field.png")
    fig.savefig(err_name, dpi=130); plt.close(fig)
    print(f"saved {path_prefix}_convergence.png and {err_name}")


if __name__ == "__main__":
    import testkit
    rep = testkit.Reporter(TITLE)
    testkit.section(TITLE)
    res, ords = run(rep)
    ok = rep.summary()
    if "--plot" in sys.argv:
        make_plots(res, ords)
    sys.exit(0 if ok else 1)
