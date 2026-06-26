"""Substep 18 -- Full-Inverse + Dirichlet odd-ghost on a CURVED boundary.

The flat free surface (Substep 17) is the easy case; curvature is the one that
matters, and is where the original linear-consistent value closure is capped
(the star Dirichlet value closure supraconverges at ~1.6-1.85). This test pairs
the second-order Full-Inverse operator with the Dirichlet odd-reflection ghost on
the curved star r=0.5+0.2 sin(5 theta), reflecting each source across the nearest
boundary facet (local-plane / faceted-boundary treatment). It confirms the
combination reaches second order on the curve and is at least as accurate as the
FIRM value closure on identical clouds.

Caveat: this is a prescribed curved Dirichlet boundary (the apples-to-apples
comparison with the FIRM value closure). The geometry-detected curved free
surface, and contact lines (mixed even/odd reflections), remain future work.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import fi
import bvp
from testkit import observed_order

try:
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve
    _HAVE_SPARSE = True
except Exception:  # pragma: no cover
    _HAVE_SPARSE = False

TITLE = "Substep 18 -- Full-Inverse + Dirichlet odd-ghost on a curved boundary"
STAR = g2.star_polygon(n=720, r0=0.5, amp=0.2, k=5, center=(0.5, 0.5))
SEG = g2.polygon_segments(STAR)


def _relL2(p, pe, mask):
    e = (p - pe)[mask]; r = pe[mask]
    return float(np.linalg.norm(e) / max(np.linalg.norm(r), 1e-30))


def _fi_dir_ghost_star(dx, jitter, seed, field):
    pos = g2.polygon_cloud(STAR, dx, jitter, seed); n = len(pos); h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    R, C, D = [], [], []; b = np.zeros(n); bnd = np.zeros(n, bool)
    for i in range(n):
        nb = nl[i]
        if len(nb) < 3:
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        xij = pos[nb] - pos[i]; w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        hit = g2.nearest_segment(pos[i], SEG[0], SEG[1], SEG[2], h)
        if hit is not None:                    # FI + Dirichlet odd-ghost across nearest facet
            bnd[i] = True; _, nrm, foot = hit
            offs = list(xij); wts = list(w); src = list(range(len(w)))
            mult = [1.0] * len(w); cst = [0.0] * len(w)
            srcs = [(k, xij[k]) for k in range(len(w))] + [(-1, np.zeros(2))]
            for sl, x0 in srcs:
                X = pos[i] + x0; sig = float((X - foot) @ nrm)
                xg = (X - 2 * sig * nrm) - pos[i]; rr = float(np.linalg.norm(xg))
                if 1e-12 < rr < h:
                    pt = float(field.value(X - sig * nrm))     # Dirichlet value at the foot
                    offs.append(xg); wts.append(fc.kernel(rr, h)); src.append(sl)
                    mult.append(-1.0); cst.append(2.0 * pt)
            d, _ = fi.fi_row(np.array(offs), np.array(wts)); diag = 0.0
            for m in range(len(d)):
                col = i if src[m] < 0 else int(nb[src[m]])
                R.append(i); C.append(col); D.append(d[m] * mult[m])
                diag -= d[m]; b[i] -= d[m] * cst[m]
            R.append(i); C.append(i); D.append(diag); b[i] += field.laplacian(pos[i]); continue
        d, _ = fi.fi_row(xij, w); diag = 0.0
        for k, j in enumerate(nb):
            R.append(i); C.append(int(j)); D.append(d[k]); diag -= d[k]
        R.append(i); C.append(i); D.append(diag); b[i] = field.laplacian(pos[i])
    A = sps.csr_matrix((D, (R, C)), shape=(n, n)); p = spsolve(A, b)
    return _relL2(p, field.value(pos), ~bnd)


def _firm_dir_star(dx, jitter, seed, field):
    pos = g2.polygon_cloud(STAR, dx, jitter, seed)
    A, b, info = bvp.assemble(pos, dx, field, STAR, "dirichlet")
    return _relL2(bvp.solve(A, b), field.value(pos), ~info["is_bnd"])


def run(rep):
    field = mf.trig_field(np.pi)
    dxs = [0.045, 0.032, 0.022]
    seeds = list(range(4))
    e_fi = [float(np.median([_fi_dir_ghost_star(dx, 0.30, s, field) for s in seeds])) for dx in dxs]
    o_fi = observed_order(dxs, e_fi)
    e_fm = [float(np.median([_firm_dir_star(dx, 0.30, s, field) for s in seeds])) for dx in dxs]
    o_fm = observed_order(dxs, e_fm)
    rep.check_order("FI+Dirichlet-ghost reaches 2nd order on a CURVED boundary", o_fi,
                    expected=2.0, slack=0.5,
                    detail=f"FI errs={['%.2e' % e for e in e_fi]} order={o_fi:.2f}")
    rep.check("FI+Dirichlet-ghost order beats the FIRM value closure on the curve",
              o_fi > o_fm,
              f"FI order {o_fi:.2f} vs FIRM value-closure {o_fm:.2f} "
              f"(both on identical curved clouds)")
    rep.check("FI+Dirichlet-ghost is at least as accurate as FIRM value closure (finest dx)",
              e_fi[-1] <= 1.2 * e_fm[-1],
              f"FI {e_fi[-1]:.2e} vs FIRM {e_fm[-1]:.2e}")


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
