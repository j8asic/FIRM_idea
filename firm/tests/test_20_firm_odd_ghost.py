"""Substep 20 -- the Dirichlet odd-ghost free surface for the ORIGINAL LDD/FIRM operator.

The reflection-ghost closure is operator-agnostic (Section 19 / firm_core.reflect_complete),
so the odd (Dirichlet) reflection that gives the Full-Inverse operator a second-order free
surface applies equally to the renormalised single-sum (LDD/FIRM) operator. This test runs
the FIRM operator on the odd-ghost-completed support, with the surface DETECTED from the cloud,
on the curved star, and checks that it converges at least as well as -- in fact better than --
the FIRM diagonal-Robin value closure of Section 3.4: the completed two-sided stencil lets the
linear-consistent operator supraconverge cleanly.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import bvp
from testkit import observed_order

try:
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve
    _HAVE_SPARSE = True
except Exception:  # pragma: no cover
    _HAVE_SPARSE = False

TITLE = "Substep 20 -- LDD/FIRM operator + Dirichlet odd-ghost free surface (detected, curved)"
STAR = g2.star_polygon(n=720, r0=0.5, amp=0.2, k=5, center=(0.5, 0.5))


def _relL2(p, pe, m):
    e = (p - pe)[m]; r = pe[m]; return float(np.linalg.norm(e) / max(np.linalg.norm(r), 1e-30))


def _firm_odd_ghost(dx, jitter, seed, field):
    pos = g2.polygon_cloud(STAR, dx, jitter, seed); n = len(pos); h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h); R, C, D = [], [], []; b = np.zeros(n); bnd = np.zeros(n, bool)
    for i in range(n):
        nb = nl[i]
        if len(nb) < 3:
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        xij = pos[nb] - pos[i]; w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        loc = fc.particle_operator(pos[i], xij, w, surface=dict(mode="natural"),
                                   dx=dx, activation="smoothstep")
        if loc.sigma > 0.0:                        # detected free surface -> odd ghost
            bnd[i] = True
            planes = [dict(n=loc.n_s, foot=loc.delta_est * loc.n_s, kind="dirichlet", data=field.value)]
            offs, wts, src, mult, cst = fc.reflect_complete(pos[i], xij, w, planes, h)
            gm = fc.geom_quantities(offs, wts)
            coeff = wts * fc.correction_weights(offs, gm.B @ gm.o) * gm.N   # FIRM coeff * N
            diag = 0.0
            for m in range(len(coeff)):
                col = i if src[m] < 0 else int(nb[src[m]])
                R.append(i); C.append(col); D.append(coeff[m] * mult[m])
                diag -= coeff[m]; b[i] -= coeff[m] * cst[m]
            R.append(i); C.append(i); D.append(diag); b[i] += field.laplacian(pos[i]); continue
        gm = fc.geom_quantities(xij, w)
        d = w * fc.correction_weights(xij, gm.B @ gm.o) * gm.N
        diag = 0.0
        for k, j in enumerate(nb):
            R.append(i); C.append(int(j)); D.append(d[k]); diag -= d[k]
        R.append(i); C.append(i); D.append(diag); b[i] = field.laplacian(pos[i])
    A = sps.csr_matrix((D, (R, C)), shape=(n, n)); p = spsolve(A, b)
    return _relL2(p, field.value(pos), ~bnd)


def _firm_diag_robin(dx, jitter, seed, field):
    pos = g2.polygon_cloud(STAR, dx, jitter, seed)
    A, b, info = bvp.assemble(pos, dx, field, STAR, "dirichlet")
    return _relL2(bvp.solve(A, b), field.value(pos), ~info["is_bnd"])


def run(rep):
    field = mf.trig_field(np.pi)
    dxs = [0.045, 0.032, 0.022]
    seeds = list(range(3))
    e_g = [float(np.median([_firm_odd_ghost(dx, 0.30, s, field) for s in seeds])) for dx in dxs]
    o_g = observed_order(dxs, e_g)
    e_r = [float(np.median([_firm_diag_robin(dx, 0.30, s, field) for s in seeds])) for dx in dxs]
    o_r = observed_order(dxs, e_r)
    rep.check_order("LDD/FIRM + odd-ghost free surface converges at ~2nd order", o_g,
                    expected=2.0, slack=0.5,
                    detail=f"odd-ghost errs={['%.2e' % e for e in e_g]} order={o_g:.2f}")
    rep.check("odd-ghost order is at least the FIRM diagonal-Robin value closure",
              o_g > o_r - 0.1,
              f"odd-ghost order {o_g:.2f} vs diagonal-Robin {o_r:.2f} (same detected clouds)")


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
