"""Substep 5 -- Wall Neumann Laplacian + projection ordering (Sec 2.3, 2.4, 2.6).

Normalized form (Sec 4.1): o_w = P_tan o, V = B o_w, w_ij = 1 - x_ij . V,
LHS = sum_j W_ij f_ij w_ij. For a linear field f = a.x with analytic flux
g_k = a.n_k, wall linear consistency requires LHS == b_wall (flux RHS).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
from testkit import RND

TITLE = "Substep 5 -- Wall Neumann Laplacian + projection ordering"


def _local(pos, i, h):
    nb = fc.neighbor_lists(pos, h)[i]
    xij = pos[nb] - pos[i]
    w = fc.kernel(np.linalg.norm(xij, axis=1), h)
    return xij, w, fc.geom_quantities(xij, w)


def _wall_residual(xij, w, gm, a, normals, proj):
    """|LHS - b_wall| for a linear field f = a.x at a wall particle."""
    fij = xij @ a
    g = normals @ a  # g_k = grad f . n_k = a . n_k
    if proj == "AN":
        betas = np.ones(len(normals)) / len(normals)
        P, neff = fc.proj_AN(normals, betas)
        b_wall = fc.wall_flux_rhs_AN(float((betas * g).sum()), gm.o, neff)
    else:
        P, Nmat, Ginv = fc.proj_GGP(normals, eps=0.0)
        b_wall = fc.wall_flux_rhs_GGP(Nmat, Ginv, gm.o, g)
    V = gm.B @ (P @ gm.o)
    wij = fc.correction_weights(xij, V)
    LHS = float((w * fij * wij).sum())
    return abs(LHS - b_wall), P


def _wedge_two_wall_idx(pos, poly, h, h_w):
    out = []
    nl = fc.neighbor_lists(pos, h)
    for i in range(len(pos)):
        if len(nl[i]) < 4:
            continue
        ws = g2.nearby_walls(pos[i], poly, h_w)
        if len(ws) == 2:
            out.append((i, ws))
    return out, nl


def run(rep):
    rng = np.random.default_rng(9)
    a = rng.normal(size=2)

    # --- single flat wall: AN and GGP both exact
    dx, h = 0.06, 2.5 * 0.06
    posh, normal, cut = g2.half_plane_cloud(dx, 0.3, 7, axis=1, cut=0.7, keep="below")
    cand = np.where((posh[:, 1] > cut - h) & (posh[:, 1] < cut) & (np.abs(posh[:, 0] - 0.5) < 0.15))[0]
    i_wall = int(cand[np.argmax(posh[cand, 1])])  # closest to the wall -> strongest truncation
    xij, w, gm = _local(posh, i_wall, h)
    for proj in ("AN", "GGP"):
        res, _ = _wall_residual(xij, w, gm, a, np.array([normal]), proj)
        rep.check_below(f"single wall consistency ({proj})", res, 1e-9, f"|LHS - b_wall| = {res:.2e}")

    # --- non-orthogonal wedge (K=2): GGP exact, AN leaks O(dx)
    an_res = {}
    ggp_res = {}
    for dxw in (0.05, 0.025):
        hw = 2.5 * dxw
        pos, poly = g2.wedge_cloud(130, dxw, 0.3, 7)
        idxs, nl = _wedge_two_wall_idx(pos, poly, hw, hw)
        worst_an = worst_ggp = 0.0
        for i, ws in idxs:
            normals = np.array([n for _, n, _ in ws])
            xij = pos[nl[i]] - pos[i]
            wk = fc.kernel(np.linalg.norm(xij, axis=1), hw)
            gm = fc.geom_quantities(xij, wk)
            worst_ggp = max(worst_ggp, _wall_residual(xij, wk, gm, a, normals, "GGP")[0])
            worst_an = max(worst_an, _wall_residual(xij, wk, gm, a, normals, "AN")[0])
        an_res[dxw], ggp_res[dxw] = worst_an, worst_ggp
    rep.check_below("wedge K=2 consistency (GGP exact)", max(ggp_res.values()), 1e-8,
                    f"GGP residuals {[f'{v:.1e}' for v in ggp_res.values()]}")
    rep.check("wedge K=2 AN leaks (>> round-off)", an_res[0.05] > 1e-4,
              f"AN residual @dx=0.05 = {an_res[0.05]:.2e}")
    ratio = an_res[0.025] / max(an_res[0.05], 1e-30)
    rep.check("wedge AN leak shrinks ~O(dx)", an_res[0.025] < an_res[0.05] and ratio < 0.85,
              f"residual ratio (dx/2) = {ratio:.2f} (O(dx) => ~0.5)")

    # --- projection ordering (Sec 2.4): B(P o) vs P(B o)
    xij, w, gm = _local(posh, i_wall, h)
    fij = xij @ a
    g = np.array([float(normal @ a)])
    P, Nmat, Ginv = fc.proj_GGP(np.array([normal]))
    b_wall = fc.wall_flux_rhs_GGP(Nmat, Ginv, gm.o, g)
    V_correct = gm.B @ (P @ gm.o)            # project, THEN renormalize
    V_wrong = P @ (gm.B @ gm.o)              # renormalize, then project
    rep.check("project-order: correct != wrong", np.linalg.norm(V_correct - V_wrong) > 1e-6,
              f"||V_correct - V_wrong|| = {np.linalg.norm(V_correct - V_wrong):.3e}")
    res_c = abs(float((w * fij * fc.correction_weights(xij, V_correct)).sum()) - b_wall)
    res_w = abs(float((w * fij * fc.correction_weights(xij, V_wrong)).sum()) - b_wall)
    rep.check_below("project-then-renormalize preserves consistency", res_c, 1e-9, f"res = {res_c:.2e}")
    rep.check("renormalize-then-project breaks consistency", res_w > 1e-4, f"res = {res_w:.2e}")


if __name__ == "__main__":
    import testkit
    rep = testkit.Reporter(TITLE)
    testkit.section(TITLE)
    run(rep)
    sys.exit(0 if rep.summary() else 1)


def test_substep():
    """pytest entry point: every check in this substep must pass."""
    import testkit
    rep = testkit.Reporter(TITLE)
    run(rep)
    assert rep.failed == 0, f"{rep.failed} failed: {rep.fails}"
