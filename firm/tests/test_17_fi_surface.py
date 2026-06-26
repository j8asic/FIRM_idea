"""Substep 17 -- Full-Inverse operator with a Dirichlet (odd-reflection) free-surface ghost.

A free surface is a Dirichlet condition, so its image construction is the ODD
(anti-symmetric) reflection, the mirror of the EVEN (value-symmetric) Neumann
ghost of Section 16:

    Neumann  (even):  p_ghost = p_src - 2 sigma g        (enforces  dp/dn = g)
    Dirichlet (odd):  p_ghost = 2 p_target - p_src        (enforces  p = p_target)

Like the Neumann ghost it is operator-agnostic stencil completion, so the
Full-Inverse operator inherits it with no single-sum cancellation identity. This
prototype isolates the mechanism on a flat surface of known geometry (the fluid
box truncated at y=1, outward normal [0,1]), exactly as test_16 isolated the wall
on a straight edge. In assembly an odd-ghost term couples to its source with the
opposite sign and sends the constant 2 p_target to the right-hand side.

Detection of the surface (n_s, delta, sigma from r_i = B o_w) is geometric and
already transfers from the FIRM detector; it is not exercised here so that the
operator+closure order is measured without detection error.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import fi
from testkit import observed_order

try:
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve
    _HAVE_SPARSE = True
except Exception:  # pragma: no cover
    _HAVE_SPARSE = False

TITLE = "Substep 17 -- Full-Inverse operator + Dirichlet free-surface ghost (odd reflection)"
YSURF = 1.0


def _surface_ghost(pos_i, xij, w, field, h, ysurf=YSURF):
    """Odd (Dirichlet) reflection of neighbours + node i across the flat surface
    y=ysurf (outward normal [0,1]). Returns aligned (offsets, weights, src_local,
    mult, const): a completed term has field value  mult*p_src + const, with
    (mult, const) = (1, 0) for a real neighbour and (-1, 2 p_target(foot)) for a
    Dirichlet ghost, p_target evaluated at the source's perpendicular surface foot."""
    n = np.array([0.0, 1.0])
    offs = list(np.asarray(xij, float))
    wts = list(np.asarray(w, float))
    src = list(range(len(w)))
    mult = [1.0] * len(w)
    const = [0.0] * len(w)
    sources = [(k, xij[k]) for k in range(len(w))] + [(-1, np.zeros(2))]
    for sl, x_src in sources:
        Xsrc = pos_i + x_src
        sig = float(Xsrc[1] - ysurf)               # signed distance to plane (<0 in fluid)
        Xg = Xsrc - 2.0 * sig * n                   # mirror image (above surface)
        xg = Xg - pos_i
        rr = float(np.linalg.norm(xg))
        if 1e-12 < rr < h:
            p_target = float(field.value(np.array([Xsrc[0], ysurf])))
            offs.append(xg); wts.append(fc.kernel(rr, h))
            src.append(sl); mult.append(-1.0); const.append(2.0 * p_target)
    return np.array(offs), np.array(wts), np.array(src), np.array(mult), np.array(const)


def _solve_free_surface(dx, jitter, seed, field):
    """Fluid box [0,1]^2: free surface (Dirichlet via odd ghost) at the top y=1,
    prescribed Dirichlet on the other three edges, FI interior. Returns
    (surface-region rel-L2, interior rel-L2)."""
    pos = g2.jittered_box(dx, jitter, seed)
    n = len(pos); h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    isD = (pos[:, 1] < h) | (pos[:, 0] < h) | (pos[:, 0] > 1 - h)
    isS = (~isD) & (pos[:, 1] > 1 - h)
    R, C, D = [], [], []; b = np.zeros(n)
    for i in range(n):
        if isD[i] or len(nl[i]) < 3:
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        nb = nl[i]; xij = pos[nb] - pos[i]; w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        if isS[i]:
            offs, wts, src, mult, const = _surface_ghost(pos[i], xij, w, field, h)
            d, _ = fi.fi_row(offs, wts)
            diag = 0.0
            for m in range(len(d)):
                col = i if src[m] < 0 else int(nb[src[m]])
                R.append(i); C.append(col); D.append(d[m] * mult[m])
                diag -= d[m]; b[i] -= d[m] * const[m]
            R.append(i); C.append(i); D.append(diag); b[i] += field.laplacian(pos[i]); continue
        d, _ = fi.fi_row(xij, w)
        diag = 0.0
        for k, j in enumerate(nb):
            R.append(i); C.append(int(j)); D.append(d[k]); diag -= d[k]
        R.append(i); C.append(i); D.append(diag); b[i] = field.laplacian(pos[i])
    A = sps.csr_matrix((D, (R, C)), shape=(n, n)); p = spsolve(A, b); pe = field.value(pos)
    surf = float(np.linalg.norm((p - pe)[isS]) / np.linalg.norm(pe[isS]))
    intr_mask = (~isD) & (~isS)
    intr = float(np.linalg.norm((p - pe)[intr_mask]) / np.linalg.norm(pe[intr_mask]))
    return surf, intr


def run(rep):
    # (1) exact for a field satisfying the surface BC: p = 1 - y is linear and zero on
    #     y=1, so the odd ghost (p_ghost = -p_src) is exact and the solve recovers it.
    bc_field = mf.linear_field([0.0, -1.0], 1.0)          # p = 1 - y, lap = 0, p|_{y=1}=0
    s_lin, i_lin = _solve_free_surface(0.05, 0.30, 3, bc_field)
    rep.check_below("FI+Dirichlet-ghost exact for a BC-satisfying linear field", max(s_lin, i_lin),
                    1e-9, f"surface rel-L2 = {s_lin:.2e}, interior = {i_lin:.2e}")

    # (2) convergence: manufactured field with a varying surface value.
    field = mf.complex_field(np.pi, 0.3)
    dxs = [0.05, 0.037, 0.027, 0.020]
    seeds = list(range(6))
    es, ei = [], []
    for dx in dxs:
        sv = [_solve_free_surface(dx, 0.30, s, field) for s in seeds]
        es.append(float(np.median([a for a, _ in sv])))
        ei.append(float(np.median([c for _, c in sv])))
    os_, oi = observed_order(dxs, es), observed_order(dxs, ei)
    rep.check_order("FI+Dirichlet-ghost free-surface region converges (approaching 2nd order)",
                    os_, expected=2.0, slack=0.6,
                    detail=f"surface errs={['%.2e' % e for e in es]} order={os_:.2f}")
    rep.check("free-surface order is close to the FI interior order (not boundary-limited)",
              os_ > 0.7 * oi,
              f"surface order {os_:.2f} vs interior order {oi:.2f} "
              f"(a flux-only/limited closure would sit near ~0.9)")


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
