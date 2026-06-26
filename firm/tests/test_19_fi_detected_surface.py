"""Substep 19 -- Full-Inverse + Dirichlet odd-ghost on a DETECTED curved free surface.

Substep 18 used the prescribed star polygon for the reflection facet. This test
closes the real gap: the surface plane is taken purely from the cloud by the FIRM
geometric detector (n_s = -r_i/|r_i| with r_i = B o, delta = |o.n_s|/S, smoothstep
activation sigma) -- no knowledge of the boundary polygon -- and the FI Dirichlet
odd-ghost reflects across that detected plane. It confirms the detection error does
not spoil the second order: the detected curved free surface converges at the same
rate as the prescribed boundary (Substep 18).

Remaining future work: contact-line nodes (a node that is both wall and surface,
needing mixed even/odd reflections in one group).
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

TITLE = "Substep 19 -- Full-Inverse + Dirichlet odd-ghost, surface DETECTED from the cloud"
STAR = g2.star_polygon(n=720, r0=0.5, amp=0.2, k=5, center=(0.5, 0.5))


def _fi_detected(dx, jitter, seed, field):
    pos = g2.polygon_cloud(STAR, dx, jitter, seed); n = len(pos); h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    R, C, D = [], [], []; b = np.zeros(n); bnd = np.zeros(n, bool)
    for i in range(n):
        nb = nl[i]
        if len(nb) < 3:
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        xij = pos[nb] - pos[i]; w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        loc = fc.particle_operator(pos[i], xij, w, surface=dict(mode="natural"),
                                   dx=dx, activation="smoothstep")
        if loc.sigma > 0.0:                    # surface DETECTED purely from the cloud
            bnd[i] = True; ns = loc.n_s; Frel = loc.delta_est * ns
            offs = list(xij); wts = list(w); src = list(range(len(w)))
            mult = [1.0] * len(w); cst = [0.0] * len(w)
            for sl, x0 in [(k, xij[k]) for k in range(len(w))] + [(-1, np.zeros(2))]:
                sig = float((x0 - Frel) @ ns); xg = x0 - 2 * sig * ns; rr = float(np.linalg.norm(xg))
                if 1e-12 < rr < h:
                    pt = float(field.value(pos[i] + x0 - sig * ns))    # Dirichlet value at detected foot
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
    A = sps.csr_matrix((D, (R, C)), shape=(n, n)); p = spsolve(A, b); pe = field.value(pos)
    return float(np.linalg.norm((p - pe)[~bnd]) / np.linalg.norm(pe[~bnd]))


def run(rep):
    field = mf.trig_field(np.pi)
    dxs = [0.045, 0.032, 0.022]
    seeds = list(range(4))
    errs = [float(np.median([_fi_detected(dx, 0.30, s, field) for s in seeds])) for dx in dxs]
    o = observed_order(dxs, errs)
    rep.check_order("FI + DETECTED curved free surface converges at ~2nd order", o,
                    expected=2.0, slack=0.5,
                    detail=f"errs={['%.2e' % e for e in errs]} order={o:.2f} "
                           f"(surface plane from the cloud detector, not the polygon)")
    rep.check("detected-surface accuracy is sound at the finest resolution", errs[-1] < 3e-3,
              f"finest interior rel-L2 = {errs[-1]:.2e}")


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
