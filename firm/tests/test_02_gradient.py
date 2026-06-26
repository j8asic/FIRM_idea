"""Substep 2 -- Renormalized gradient operator (docs/spec.md Sec 1.3)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
from testkit import RND, observed_order

TITLE = "Substep 2 -- Gradient operator"


def _grad_field(pos, h, field):
    nl = fc.neighbor_lists(pos, h)
    f = field.value(pos)
    out = np.full((len(pos), 2), np.nan)
    for i in range(len(pos)):
        if len(nl[i]) < 3:
            continue
        xij = pos[nl[i]] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        gm = fc.geom_quantities(xij, w)
        out[i] = fc.grad_op(gm.B, xij, w, f[nl[i]] - f[i])
    return out


def run(rep):
    h_of = lambda dx: 2.5 * dx

    # --- linear-exact on EVERY particle (interior, boundary, truncated, wedge)
    rng = np.random.default_rng(3)
    a = rng.normal(size=2)
    lin = mf.linear_field(a, 0.21)
    worst = 0.0
    for label, pos in [("box", g2.jittered_box(0.07, 0.3, 7)),
                       ("half-plane", g2.half_plane_cloud(0.07, 0.3, 7, axis=1, cut=0.7)[0]),
                       ("wedge-130", g2.wedge_cloud(130, 0.06, 0.3, 7)[0])]:
        gr = _grad_field(pos, h_of(0.07 if label != "wedge-130" else 0.06), lin)
        err = np.nanmax(np.linalg.norm(gr - a, axis=1))
        worst = max(worst, err)
        rep.check_below(f"linear-exact grad ({label})", err, 1e-9, f"max |grad - a| = {err:.2e}")

    # --- convergence on smooth nonlinear fields (interior only)
    dxs = [0.06, 0.045, 0.033, 0.025]
    for field in (mf.QUAD2D, mf.TRIG2D):
        errs = []
        for dx in dxs:
            pos = g2.jittered_box(dx, 0.3, 7)
            mask = g2.box_interior_mask(pos, margin=h_of(dx))
            gr = _grad_field(pos, h_of(dx), field)
            ex = field.grad(pos)
            num = np.linalg.norm((gr - ex)[mask])
            den = np.linalg.norm(ex[mask])
            errs.append(num / den)
        order = observed_order(dxs, errs)
        rep.check_order(f"convergence grad ({field.name})", order, expected=1.5, slack=0.5,
                        detail=f"errs={['%.2e' % e for e in errs]} order={order:.2f}")


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
