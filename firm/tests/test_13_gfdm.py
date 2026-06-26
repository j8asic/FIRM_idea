"""Substep 13 -- GFDM / 2nd-order WLSQ baseline and its constraint-row Neumann.

The paper compares FIRM against a generalised finite-difference (GFDM) baseline: a
full-quadratic-basis weighted-least-squares Laplacian (the "Asai-type" 2nd-order
operator) with an exact-flux KKT constraint row (Tiwari--Kuhnert) for Neumann walls.
This test pins the baseline's properties so the comparison is trustworthy:
  * interior operator is linear-exact and 2nd-order (recovers a quadratic Laplacian);
  * the constraint row enforces the flux exactly and stays linear-exact at the wall;
  * the soft penalty form is a genuinely different (weaker) operator;
  * a Dirichlet Poisson solve converges, and on an all-Neumann box the FIRM
    mirror-ghost closure beats both flux-only closures (FIRM projection and GFDM
    constraint-row), which are limited by one-sided normal curvature.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import gfdm
import bvp
from testkit import observed_order

TITLE = "Substep 13 -- GFDM 2nd-order baseline + constraint-row Neumann"
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def run(rep):
    rng = np.random.default_rng(0)
    xij = rng.uniform(-1, 1, (24, 2)) * 0.3
    w = fc.kernel(np.linalg.norm(xij, axis=1), 0.5)
    xij, w = xij[w > 0], w[w > 0]

    # --- interior operator: linear-exact and 2nd-order (exact for any quadratic) ---
    lin = mf.linear_field([0.7, -0.4], 0.2)
    fij = lin.value(xij) - lin.value(np.zeros(2))
    d, bcv = gfdm.gfdm_row(xij, w, ridge=0.0)
    rep.check_below("GFDM interior linear-exact", abs(d @ fij + bcv), 1e-10,
                    f"lap(linear) = {abs(d @ fij + bcv):.2e}")
    Q = mf.quadratic_field([[1.3, 0.4], [0.4, -0.7]], [0.2, -0.1], 0.05)
    fij = Q.value(xij) - Q.value(np.zeros(2))
    d, bcv = gfdm.gfdm_row(xij, w, ridge=0.0)
    rep.check_close("GFDM exact for quadratic (2nd-order)", d @ fij + bcv, 1.3 - 0.7, tol=1e-9,
                    detail=f"lap(quad) = {d @ fij + bcv:.6f} (expect {1.3 - 0.7:.3f})")

    # --- constraint-row Neumann: enforces flux exactly, stays linear-exact -----
    n = np.array([0.3, -0.95]); n = n / np.linalg.norm(n)
    g = float(lin.grad(np.zeros(2)) @ n)
    fij = lin.value(xij) - lin.value(np.zeros(2))
    dc, bc = gfdm.gfdm_row(xij, w, [(n, g)], ridge=0.0, constraint="constraint")
    rep.check_below("GFDM constraint-row linear-exact at wall", abs(dc @ fij + bc), 1e-9,
                    f"lap(linear)|wall = {abs(dc @ fij + bc):.2e}")
    dp, bp = gfdm.gfdm_row(xij, w, [(n, g)], ridge=0.0, constraint="penalty")
    rep.check("penalty != constraint (distinct operators)",
              np.linalg.norm(dc - dp) > 1e-6, f"||d_con - d_pen|| = {np.linalg.norm(dc - dp):.2e}")

    # --- Dirichlet Poisson convergence (square, trig) -------------------------
    trig = mf.trig_field(np.pi)
    dxs = [0.05, 0.035, 0.025]

    def gfdm_box(dx, seed):
        pos = g2.jittered_box(dx, 0.30, seed)
        dirm = ~g2.box_interior_mask(pos, 2.5 * dx)
        A, b, _ = gfdm.assemble_gfdm(pos, dx, trig, dir_mask=dirm)
        p = gfdm.solve(A, b)
        m = g2.box_interior_mask(pos, 2.5 * dx)
        return float(np.linalg.norm((p - trig.value(pos))[m]) / np.linalg.norm(trig.value(pos)[m]))

    errs = [float(np.median([gfdm_box(dx, s) for s in range(6)])) for dx in dxs]
    o = observed_order(dxs, errs)
    rep.check_order("GFDM Dirichlet Poisson converges (2nd-order)", o, expected=1.6, slack=0.3,
                    detail=f"errs {[f'{e:.2e}' for e in errs]} order {o:.2f}")

    # --- headline: FIRM ghost beats both flux-only closures on all-Neumann box -
    cf = mf.complex_field(np.pi, 0.3)
    dx = 0.04

    def neu(method, seed):
        pos = g2.jittered_box(dx, 0.30, seed)
        pin = int(np.argmin(np.linalg.norm(pos - 0.5, axis=1)))
        if method.startswith("firm"):
            wc = "ghost" if method.endswith("ghost") else "projection"
            A, b, info = bvp.assemble(pos, dx, cf, SQUARE, "neumann", wall_closure=wc, pin=pin)
            solve = bvp.solve
        else:
            A, b, info = gfdm.assemble_gfdm(pos, dx, cf, poly=SQUARE, neumann="constraint", pin=pin)
            solve = gfdm.solve
        p = solve(A, b)
        m = g2.box_interior_mask(pos, 2.5 * dx)
        e = (p - cf.value(pos))[m]; r = cf.value(pos)[m]
        return float(np.linalg.norm(e - e.mean()) / np.linalg.norm(r - r.mean()))

    seeds = [7, 11, 19, 23, 31]
    e_proj = float(np.median([neu("firm-proj", s) for s in seeds]))
    e_ghost = float(np.median([neu("firm-ghost", s) for s in seeds]))
    e_gfdm = float(np.median([neu("gfdm", s) for s in seeds]))
    rep.check("FIRM ghost beats FIRM projection on Neumann box", e_ghost < 0.5 * e_proj,
              f"proj {e_proj:.2e} -> ghost {e_ghost:.2e}")
    rep.check("FIRM ghost beats GFDM constraint-row on Neumann box", e_ghost < 0.5 * e_gfdm,
              f"gfdm-constraint {e_gfdm:.2e} -> ghost {e_ghost:.2e} (both flux-only closures are "
              f"limited by one-sided normal curvature)")


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
