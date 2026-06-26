"""Substep 7 -- Unified correction vector V_i = B_i o_*,i (Sec 3.5, 3.6).

Checks the unified operator reduces correctly to the interior, wall-only, and
surface-only limits, and that at a contact line the wall and surface projections
do not interfere (o_* stays in the wall-tangential subspace; Sec 3.5).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
from testkit import RND

TITLE = "Substep 7 -- Unified correction vector"


def _nbr(pos, i, h):
    nb = fc.neighbor_lists(pos, h)[i]
    xij = pos[nb] - pos[i]
    w = fc.kernel(np.linalg.norm(xij, axis=1), h)
    return xij, w


def run(rep):
    dx, h = 0.06, 2.5 * 0.06
    pos = g2.jittered_box(dx, 0.3, 7)

    # --- interior reduction: no walls, no surface  =>  V == B o
    i_int = int(np.argmin(np.linalg.norm(pos - 0.5, axis=1)))
    xij, w = _nbr(pos, i_int, h)
    loc = fc.particle_operator(pos[i_int], xij, w, dx=dx)
    rep.check_below("interior: V == B o", np.linalg.norm(loc.V - loc.geom.B @ loc.geom.o), RND,
                    f"diff = {np.linalg.norm(loc.V - loc.geom.B @ loc.geom.o):.2e}")
    rep.check("interior: robin_diag == 0", loc.robin_diag == 0.0)

    # --- wall-only reduction:  V == B (P_tan o)
    posh, normal, cut = g2.half_plane_cloud(dx, 0.3, 7, axis=1, cut=0.7)
    cand = np.where((posh[:, 1] > cut - h) & (posh[:, 1] < cut) & (np.abs(posh[:, 0] - 0.5) < 0.15))[0]
    iw = cand[0]
    xij, w = _nbr(posh, iw, h)
    walls = dict(normals=np.array([normal]), deltas=np.array([cut - posh[iw, 1]]), g=np.array([0.3]), h_w=h)
    loc = fc.particle_operator(posh[iw], xij, w, walls=walls, dx=dx, wall_proj="GGP")
    rep.check_below("wall-only: V == B (P_tan o)", np.linalg.norm(loc.V - loc.geom.B @ (loc.P_tan @ loc.geom.o)),
                    RND, "matches manual projection")
    rep.check("wall-only: robin_diag == 0 (Neumann)", loc.robin_diag == 0.0)

    # --- surface-only (exact, sigma=1): normal component strongly reduced
    i_surf = int(np.argmax(pos[:, 1]))
    xij, w = _nbr(pos, i_surf, h)
    surf = dict(mode="exact", n_s=np.array([0.0, 1.0]), delta=1.0 - pos[i_surf, 1], sigma=1.0)
    loc = fc.particle_operator(pos[i_surf], xij, w, surface=surf, dx=dx)
    before = abs(float(loc.o_w @ surf["n_s"]))
    after = abs(float(loc.o_star @ surf["n_s"]))
    rep.check("surface-only: normal component removed", after < 1e-9 and before > 1e-3,
              f"|o_w.n_s| = {before:.3e} -> |o_*.n_s| = {after:.2e}")
    rep.check("surface-only: robin_diag = -(o_w.n_s)/delta > 0", loc.robin_diag > 0,
              f"robin_diag = {loc.robin_diag:.3f}")

    # --- contact line: single wall + exact surface with n_s in tangential subspace
    # box top-left region: left wall normal (-1,0); surface normal (0,1) (orthogonal)
    corner = np.where((pos[:, 0] < h) & (pos[:, 1] > 1 - h))[0]
    ic = corner[int(np.argmax(pos[corner, 1]))]
    xij, w = _nbr(pos, ic, h)
    walls = dict(normals=np.array([[-1.0, 0.0]]), deltas=np.array([pos[ic, 0]]), g=np.array([0.1]), h_w=h)
    surf = dict(mode="exact", n_s=np.array([0.0, 1.0]), delta=1.0 - pos[ic, 1], sigma=1.0)
    loc = fc.particle_operator(pos[ic], xij, w, walls=walls, surface=surf, dx=dx, wall_proj="GGP")
    leak = np.linalg.norm((np.eye(2) - loc.P_tan) @ loc.o_star)
    rep.check_below("contact line: o_* stays in wall-tangential subspace", leak, 1e-9,
                    f"||(I - P_tan) o_*|| = {leak:.2e}")
    rep.check("contact line: |V| -> ~0 (both directions constrained)", np.linalg.norm(loc.V) < 0.5,
              f"|V| = {np.linalg.norm(loc.V):.3e}")


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
