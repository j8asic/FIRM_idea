"""Substep 10 -- Mirror-ghost wall closure (the Neumann accuracy fix).

The flux-only projection wall closure converges but with a large constant (the
normal 2nd-derivative is one-sided). Completing the boundary stencil with mirror
ghosts -- reflecting every neighbour AND particle i across the near wall(s), plus
the corner double-reflection -- supplies the missing normal curvature (LeVeque
ghost-node cure). This test checks it stays linear-exact and sharply reduces the
wall-region error on the manufactured complex field.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import poisson as ps
import manufactured as mf
import geometry2d as g2
from testkit import observed_order

TITLE = "Substep 10 -- Mirror-ghost wall closure"

TANK = np.array([[0.0, 1.0], [1.2, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]])
FILL_H = 1.8


def _solve(dx, field, wall_closure):
    pos = g2.tank_cloud(dx, FILL_H, TANK, jitter=0.30, seed=7)
    A, b, info = ps.assemble(pos, dx, field, poly=TANK, fill_h=FILL_H, wall_proj="GGP",
                             surface_mode="exact", wall_closure=wall_closure)
    p = ps.solve(A, b)
    pe = field.value(pos)
    scale = max(np.max(np.abs(pe)), 1e-30)
    wall = info["is_wall"]
    wall_rms = float(np.sqrt(np.mean(((p - pe)[wall]) ** 2)) / scale)
    l2 = float(np.linalg.norm(p - pe) / np.linalg.norm(pe))
    return l2, wall_rms


def run(rep):
    # --- linear-exact preserved by the ghost closure (ghost value of a linear
    #     field equals the true field at the mirrored point)
    lin = mf.linear_field([0.3, -0.5], 0.2, "lin")
    l2, _ = _solve(0.08, lin, "ghost")
    rep.check("ghost wall closure stays linear-exact", l2 < 1e-7, f"rel L2 = {l2:.2e}")

    # --- complex field: ghost sharply reduces wall-region AND overall error
    cf = mf.complex_field(np.pi, 0.3)
    dxs = [0.06, 0.045, 0.033]
    proj = [_solve(dx, cf, "projection") for dx in dxs]
    ghost = [_solve(dx, cf, "ghost") for dx in dxs]
    for k, dx in enumerate(dxs):
        rep.check(f"ghost wall error << projection (dx={dx})",
                  ghost[k][1] < 0.5 * proj[k][1],
                  f"wall RMS: projection {proj[k][1]:.2e} -> ghost {ghost[k][1]:.2e}")
    rep.check("ghost overall L2 << projection at finest dx",
              ghost[-1][0] < 0.5 * proj[-1][0],
              f"L2: projection {proj[-1][0]:.2e} -> ghost {ghost[-1][0]:.2e}")
    o_proj = observed_order(dxs, [e[0] for e in proj])
    o_ghost = observed_order(dxs, [e[0] for e in ghost])
    rep.check("ghost converges at least as fast as projection", o_ghost > o_proj - 0.2,
              f"order: projection {o_proj:.2f}, ghost {o_ghost:.2f}")


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
