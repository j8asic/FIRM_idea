"""Substep 16 -- Full-Inverse operator with the algebraic-ghost Neumann closure.

The algebraic-ghost closure is operator-agnostic stencil completion
(``firm_core.ghost_complete``): it reflects the neighbours and the node across the
boundary, assigns Neumann-consistent ghost values, and lets the ordinary interior
operator run on the completed support. Pairing it with the second-order Full-Inverse
operator (``fi.assemble_fi`` with ``poly=``) is therefore expected to give a
genuinely second-order Neumann boundary on disordered clouds, with no extra unknowns
-- the "future" cell of the operator/closure matrix in the paper. This test measures it.

Unlike the renormalised single-sum closures, no cancellation identity is invoked: FI
inherits linear-exactness at the boundary purely because the ghost completion is
linear-reproducing (a ghost of a linear field is the true field at the mirror point).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import fi
import paper_benchmarks as pb
from testkit import observed_order

try:
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve
    _HAVE_SPARSE = True
except Exception:  # pragma: no cover
    _HAVE_SPARSE = False

TITLE = "Substep 16 -- Full-Inverse operator + algebraic-ghost Neumann closure"
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def _fi_straight_neumann(dx, seed, field):
    """FI interior + FI-ghost on a straight Neumann edge at y=0, Dirichlet on the
    other three edges (well-posed; no non-orthogonal-corner cap). Returns the
    Neumann-strip relative L2 error."""
    pos = g2.jittered_box(dx, 0.30, seed)
    n = len(pos); h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    isN = (pos[:, 1] < h) & (pos[:, 0] > h) & (pos[:, 0] < 1 - h)
    isD = (~isN) & ((pos[:, 0] < h) | (pos[:, 0] > 1 - h) | (pos[:, 1] > 1 - h) | (pos[:, 1] < h))
    nrm = np.array([0.0, -1.0])
    R, C, D = [], [], []; b = np.zeros(n)
    for i in range(n):
        if isD[i] or len(nl[i]) < 3:
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        nb = nl[i]; xij = pos[nb] - pos[i]; w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        if isN[i]:
            foot = pos[i].copy(); foot[1] = 0.0; gf = float(field.grad(foot) @ nrm)
            offs, wts, src, inc = fc.ghost_complete(
                xij, w, np.array([nrm]), np.array([foot - pos[i]]), np.array([gf]), h)
            d, _ = fi.fi_row(offs, wts)
            diag = 0.0
            for m in range(len(d)):
                col = i if src[m] < 0 else int(nb[src[m]])
                R.append(i); C.append(col); D.append(d[m]); diag -= d[m]; b[i] -= d[m] * inc[m]
            R.append(i); C.append(i); D.append(diag); b[i] += field.laplacian(pos[i]); continue
        d, _ = fi.fi_row(xij, w)
        diag = 0.0
        for k, j in enumerate(nb):
            R.append(i); C.append(int(j)); D.append(d[k]); diag -= d[k]
        R.append(i); C.append(i); D.append(diag); b[i] = field.laplacian(pos[i])
    A = sps.csr_matrix((D, (R, C)), shape=(n, n)); p = spsolve(A, b); pe = field.value(pos)
    return float(np.linalg.norm((p - pe)[isN]) / np.linalg.norm(pe[isN]))


def run(rep):
    field = mf.complex_field(np.pi, 0.3)

    # (1) linear-exactness at a wall: a ghost of a linear field is the true field at
    #     the mirror point, so FI on the completed support reproduces lap(linear)=0.
    rng = np.random.default_rng(1)
    xij = rng.uniform(-1, 1, (24, 2)) * 0.3
    w = fc.kernel(np.linalg.norm(xij, axis=1), 0.5)
    xij, w = xij[w > 0], w[w > 0]
    lin = mf.linear_field([0.7, -0.4], 0.2)
    nrm = np.array([0.0, -1.0]); foot_rel = np.array([0.0, -0.15])
    gf = float(lin.grad(np.zeros(2)) @ nrm)
    offs, wts, src, inc = fc.ghost_complete(xij, w, np.array([nrm]), np.array([foot_rel]),
                                            np.array([gf]), 0.5)
    d, _ = fi.fi_row(offs, wts, ridge=0.0)
    # value at each completed term: real -> f(x_src), ghost -> f(x_src)+inc
    fsrc = np.array([lin.value(np.zeros(2)) if s < 0 else lin.value(xij[s]) for s in src])
    fterm = fsrc + inc
    lap = float(d @ (fterm - lin.value(np.zeros(2))))
    rep.check_below("FI+ghost linear-exact at a Neumann wall (no identity needed)",
                    abs(lap), 1e-9, f"lap(linear)|wall = {abs(lap):.2e}")

    # (2) headline: straight Neumann edge convergence vs the FIRM ghost/projection.
    dxs = [0.05, 0.037, 0.027, 0.020]
    seeds = list(range(6))
    e_fi = [float(np.median([_fi_straight_neumann(dx, s, field) for s in seeds])) for dx in dxs]
    o_fi = observed_order(dxs, e_fi)
    e_fg = [float(np.median([pb._neu_edge(dx, s, "ghost-denom") for s in seeds])) for dx in dxs]
    o_fg = observed_order(dxs, e_fg)
    e_pr = [float(np.median([pb._neu_edge(dx, s, "projection") for s in seeds])) for dx in dxs]
    o_pr = observed_order(dxs, e_pr)
    rep.check_order("FI+ghost approaches 2nd order at a straight Neumann edge", o_fi,
                    expected=2.0, slack=0.45,
                    detail=f"FI+ghost errs={['%.2e' % e for e in e_fi]} order={o_fi:.2f}")
    rep.check("FI+ghost beats the flux-only projection closure (order)", o_fi > o_pr + 0.5,
              f"FI+ghost order {o_fi:.2f} vs FIRM projection {o_pr:.2f}")
    rep.check("FI+ghost is at least as accurate as the renormalised ghost (finest dx)",
              e_fi[-1] <= 1.5 * e_fg[-1],
              f"FI+ghost {e_fi[-1]:.2e} vs FIRM ghost-denom {e_fg[-1]:.2e} "
              f"(orders {o_fi:.2f} vs {o_fg:.2f})")

    # (3) all-Neumann box through assemble_fi(poly=): runs, bounded, mean-removed error
    #     decreases. (Order is null-space noise on a singular box, so not asserted.)
    def fi_box(dx, seed):
        pos = g2.jittered_box(dx, 0.30, seed)
        pin = int(np.argmin(np.linalg.norm(pos - 0.5, axis=1)))
        A, b, info = fi.assemble_fi(pos, dx, field, poly=SQUARE, pin=pin)
        p = fi.solve(A, b); m = g2.box_interior_mask(pos, 2.5 * dx)
        e = (p - field.value(pos))[m]; r = field.value(pos)[m]
        return float(np.linalg.norm(e - e.mean()) / np.linalg.norm(r - r.mean()))
    bx = [0.05, 0.035, 0.025]
    eb = [float(np.median([fi_box(dx, s) for s in range(4)])) for dx in bx]
    rep.check("FI+ghost all-Neumann box assembles and is bounded", eb[-1] < 5e-2,
              f"mean-removed rel-L2 errs={['%.2e' % e for e in eb]} (finest {eb[-1]:.2e})")
    rep.check("FI+ghost all-Neumann box error decreases under refinement", eb[-1] < eb[0],
              f"{eb[0]:.2e} -> {eb[-1]:.2e}")


if __name__ == "__main__":
    import testkit
    rep = testkit.Reporter(TITLE)
    testkit.section(TITLE)
    run(rep)
    sys.exit(0 if rep.summary() else 1)


def test_substep():
    import testkit
    rep = testkit.Reporter(TITLE)
    run(rep)
    assert rep.failed == 0, f"{rep.failed} failed: {rep.fails}"
