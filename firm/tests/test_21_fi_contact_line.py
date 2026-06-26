"""Substep 21 -- contact lines: Full-Inverse with composed even+odd reflection ghosts.

A contact-line node is both a Neumann wall node and a free-surface node. With
firm_core.reflect_complete the two are simply composed: the reflection group contains
the wall (even) plane, the surface (odd) plane, and their double reflection. This test
solves a fluid box with Neumann walls on the bottom and sides and a DETECTED free surface
on top, so the two top corners are Neumann+Dirichlet contact lines, with the Full-Inverse
operator throughout, and checks that the interior reaches second order and the
wall/contact-line region tracks it -- i.e. the composed ghost does not spoil convergence.
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

TITLE = "Substep 21 -- Full-Inverse contact lines (composed even+odd reflection ghosts)"


def _relL2(p, pe, m):
    e = (p - pe)[m]; r = pe[m]; return float(np.linalg.norm(e) / max(np.linalg.norm(r), 1e-30))


def _solve(dx, jitter, seed, field):
    pos = g2.jittered_box(dx, jitter, seed); n = len(pos); h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    R, C, D = [], [], []; b = np.zeros(n); free = np.zeros(n, bool); wall = np.zeros(n, bool)
    for i in range(n):
        nb = nl[i]
        if len(nb) < 3:
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        xij = pos[nb] - pos[i]; w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        wl = []                                    # prescribed Neumann walls (not the top)
        if pos[i, 1] < h:     wl.append((np.array([0., -1.]), np.array([pos[i, 0], 0.])))
        if pos[i, 0] < h:     wl.append((np.array([-1., 0.]), np.array([0., pos[i, 1]])))
        if pos[i, 0] > 1 - h: wl.append((np.array([1., 0.]),  np.array([1., pos[i, 1]])))
        wdict = None
        if wl:
            wall[i] = True
            wdict = dict(normals=np.array([nn for nn, _ in wl]),
                         deltas=np.array([abs(float((ff - pos[i]) @ nn)) for nn, ff in wl]),
                         g=np.zeros(len(wl)), h_w=h)
        loc = fc.particle_operator(pos[i], xij, w, walls=wdict, surface=dict(mode="natural"),
                                   dx=dx, activation="smoothstep")
        planes = [dict(n=nn, foot=ff - pos[i], kind="neumann", data=field.wall_flux(ff, nn))
                  for nn, ff in wl]
        if loc.sigma > 0.0:
            free[i] = True
            planes.append(dict(n=loc.n_s, foot=loc.delta_est * loc.n_s,
                               kind="dirichlet", data=field.value))
        if planes:
            offs, wts, src, mult, cst = fc.reflect_complete(pos[i], xij, w, planes, h)
            d = fi.fi_row(offs, wts)[0]; diag = 0.0
            for m in range(len(d)):
                col = i if src[m] < 0 else int(nb[src[m]])
                R.append(i); C.append(col); D.append(d[m] * mult[m])
                diag -= d[m]; b[i] -= d[m] * cst[m]
            R.append(i); C.append(i); D.append(diag); b[i] += field.laplacian(pos[i])
        else:
            d = fi.fi_row(xij, w)[0]; diag = 0.0
            for k, j in enumerate(nb):
                R.append(i); C.append(int(j)); D.append(d[k]); diag -= d[k]
            R.append(i); C.append(i); D.append(diag); b[i] = field.laplacian(pos[i])
    A = sps.csr_matrix((D, (R, C)), shape=(n, n)); p = spsolve(A, b); pe = field.value(pos)
    interior = (~free) & (~wall)
    n_cl = int((free & wall).sum())
    return _relL2(p, pe, interior), _relL2(p, pe, wall), n_cl


def run(rep):
    field = mf.complex_field(np.pi, 0.3)
    dxs = [0.05, 0.035, 0.025]
    seeds = list(range(4))
    ei, ew, ncl = [], [], []
    for dx in dxs:
        res = [_solve(dx, 0.30, s, field) for s in seeds]
        ei.append(float(np.median([a for a, _, _ in res])))
        ew.append(float(np.median([c for _, c, _ in res])))
        ncl.append(int(np.median([k for _, _, k in res])))
    oi, ow = observed_order(dxs, ei), observed_order(dxs, ew)
    rep.check("contact-line nodes are actually present (Neumann wall meets free surface)",
              min(ncl) > 0, f"contact-line nodes per cloud: {ncl}")
    rep.check_order("FI interior reaches 2nd order with walls + detected surface + contact lines",
                    oi, expected=2.0, slack=0.5,
                    detail=f"interior errs={['%.2e' % e for e in ei]} order={oi:.2f}")
    rep.check("Neumann/contact-line region tracks the interior (not boundary-limited)",
              ow > 1.4,
              f"wall+contact order {ow:.2f}, errs={['%.2e' % e for e in ew]}")


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
