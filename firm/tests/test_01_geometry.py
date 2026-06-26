"""Substep 1 -- Geometric quantities o, S, M, B, N (docs/spec.md Sec 1.2)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
from testkit import RND

TITLE = "Substep 1 -- Geometry (o, S, M, B, N)"


def _geom_at(pos, i, h):
    nb = fc.neighbor_lists(pos, h)[i]
    xij = pos[nb] - pos[i]
    w = fc.kernel(np.linalg.norm(xij, axis=1), h)
    return fc.geom_quantities(xij, w), nb


def run(rep):
    dx, h = 0.08, 2.5 * 0.08

    # --- B M = I, symmetry, SPD on interior AND boundary rows of a jittered box
    pos = g2.jittered_box(dx, jitter=0.3, seed=7)
    nl = fc.neighbor_lists(pos, h)
    worst_inv = worst_sym = 0.0
    min_eig = np.inf
    too_few = 0
    for i in range(len(pos)):
        if len(nl[i]) < 3:
            too_few += 1
            continue
        xij = pos[nl[i]] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        gm = fc.geom_quantities(xij, w)
        worst_inv = max(worst_inv, np.linalg.norm(gm.B @ gm.M - np.eye(2)))
        worst_sym = max(worst_sym, np.linalg.norm(gm.B - gm.B.T))
        min_eig = min(min_eig, np.linalg.eigvalsh(gm.B).min())
        if abs(gm.N - (2.0 / gm.d) * np.trace(gm.B)) > RND:
            rep.check("N == (2/d)tr(B)", False, "mismatch")
            return
    rep.check_below("B@M = I (all rows incl. boundary)", worst_inv, 1e-9, f"max ||B M - I|| = {worst_inv:.2e}")
    rep.check_below("B symmetric", worst_sym, RND, f"max ||B - B^T|| = {worst_sym:.2e}")
    rep.check("B SPD (min eig > 0)", min_eig > 0, f"min eig(B) = {min_eig:.3e}")
    rep.check("N = (2/d) tr(B)", True, "verified on every row")
    rep.check("all rows have >= d+1 neighbours", too_few == 0, f"{too_few} under-supported")

    # --- o -> 0 for a symmetric full-support particle (zero jitter)
    posg = g2.jittered_box(dx, jitter=0.0, seed=1)
    centre = int(np.argmin(np.linalg.norm(posg - 0.48, axis=1)))
    gm_c, _ = _geom_at(posg, centre, h)
    rep.check_below("o = 0 on symmetric lattice (full support)", np.linalg.norm(gm_c.o), 1e-12,
                    f"|o| = {np.linalg.norm(gm_c.o):.2e}")

    # --- o points inward at a truncated boundary (o . outward_normal < 0)
    posh, normal, cut = g2.half_plane_cloud(dx, 0.3, 7, axis=1, cut=0.7, keep="below")
    cand = np.where((posh[:, 1] > cut - h) & (posh[:, 1] < cut) & (np.abs(posh[:, 0] - 0.5) < 0.12))[0]
    i_near = int(cand[np.argmax(posh[cand, 1])])  # closest to the wall
    gm_b, _ = _geom_at(posh, i_near, h)
    odn = float(gm_b.o @ normal)
    rep.check("o points inward at wall (o . n_out < 0)", odn < 0, f"o . n_out = {odn:.3e}")
    rep.check("|o| at boundary >> 0 (not round-off)", np.linalg.norm(gm_b.o) > 1e-3,
              f"|o| = {np.linalg.norm(gm_b.o):.3e}")

    # --- under-supported detection flag (Sec 10.2)
    xij = np.array([[dx, 0.0]])  # a single neighbour -> rank-deficient
    gm_bad = fc.geom_quantities(xij, np.array([1.0]), ridge=1e-6 * dx * dx)
    rep.check("under-support flagged (K < d+1)", gm_bad.ok is False, "ok flag = False")


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
