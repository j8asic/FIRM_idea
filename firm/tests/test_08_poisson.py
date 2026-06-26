"""Substep 8 -- Hydrostatic Poisson: linear-exact recovery (docs/spec.md Sec 4).

The renormalized operators reproduce linear fields exactly on ANY cloud, and for
a linear field the Taylor-to-surface relation is exact, so the unified system
recovers p_exact to round-off with the EXACT surface treatment -- isolating the
wall (AN/GGP) closure and the assembly. The natural-Robin treatment instead
leaves an O(eta) error here (its surface distance/normal are estimated), which
the capstone then measures as finite-order convergence.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import poisson as ps
import manufactured as mf

TITLE = "Substep 8 -- Hydrostatic Poisson (linear-exact)"

TANK = np.array([[0.0, 1.0], [1.2, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
FILL_H = 1.8
RHO, G = 1000.0, 9.81


def run(rep):
    import geometry2d as g2
    dx = 0.10
    # hydrostatic pressure p = rho g (fill_h - y): linear, grad = (0,-rho g), lap = 0
    field = mf.linear_field([0.0, -RHO * G], RHO * G * FILL_H, "hydrostatic")
    pos = g2.tank_cloud(dx, FILL_H, TANK, jitter=0.30, seed=7)
    p_exact = field.value(pos)

    errs = {}
    for wall_proj in ("AN", "GGP"):
        A, b, info = ps.assemble(pos, dx, field, poly=TANK, fill_h=FILL_H,
                                 wall_proj=wall_proj, surface_mode="exact")
        p = ps.solve(A, b)
        errs[wall_proj] = ps.rel_errors(p, p_exact)
    # GGP is exact on linear fields for ANY geometry (round-off recovery)
    rep.check("linear-exact recovery, GGP walls", errs["GGP"][1] < 1e-7,
              f"rel L_inf = {errs['GGP'][1]:.2e}, rel L2 = {errs['GGP'][0]:.2e} (N={len(pos)})")
    # AN leaks O(dx) at the tank's non-orthogonal slant/floor wedge (Sec 2.2/10.3)
    rep.check("AN walls bounded but leak at wedge (Sec 10.3)",
              1e-4 < errs["AN"][1] < 0.2 and errs["AN"][1] > 10 * errs["GGP"][1],
              f"AN rel L_inf = {errs['AN'][1]:.2e} >> GGP {errs['GGP'][1]:.2e} (the documented AN corner error)")

    # --- natural-Robin: NOT round-off here (documents the estimate's O(eta) error)
    A, b, info = ps.assemble(pos, dx, field, poly=TANK, fill_h=FILL_H,
                             wall_proj="GGP", surface_mode="natural")
    p = ps.solve(A, b)
    l2n, linfn = ps.rel_errors(p, p_exact)
    rep.check("natural-Robin bounded (expected > round-off)", linfn < 0.5,
              f"rel L_inf = {linfn:.2e} (natural surface estimate; capstone measures its order)")

    # --- constant-field guard: interior rows annihilate a constant
    const = mf.linear_field([0.0, 0.0], 5.0, "const")
    A, b, info = ps.assemble(pos, dx, const, poly=TANK, fill_h=FILL_H,
                             wall_proj="GGP", surface_mode="exact")
    pc = ps.solve(A, b)
    l2c, linfc = ps.rel_errors(pc, const.value(pos))
    rep.check("constant field recovered (assembly guard)", linfc < 1e-7,
              f"rel L_inf = {linfc:.2e}")


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
