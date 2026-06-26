"""Substep 3 -- Interior Laplacian operator (docs/spec.md Sec 1.4).

Includes a 3D constant-Laplacian check that PINS the N_i = (2/d) tr(B)
convention: in 2D the 2/d factor equals 1, so only a 3D test distinguishes it
from the bare tr(B) used by the old (untrusted) firm_* scripts.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
from testkit import RND, observed_order

TITLE = "Substep 3 -- Interior Laplacian"


def _lap_field(pos, h, field, form="boxed"):
    nl = fc.neighbor_lists(pos, h)
    f = field.value(pos)
    out = np.full(len(pos), np.nan)
    for i in range(len(pos)):
        if len(nl[i]) < pos.shape[1] + 1:
            continue
        xij = pos[nl[i]] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        gm = fc.geom_quantities(xij, w)
        fij = f[nl[i]] - f[i]
        out[i] = (fc.laplacian_interior(gm, xij, w, fij) if form == "boxed"
                  else fc.laplacian_naive(gm, xij, w, fij))
    return out


def run(rep):
    h_of = lambda dx: 2.5 * dx

    # --- (a) linear consistency: Lap(linear) = 0 on ANY cloud incl. truncated rows
    rng = np.random.default_rng(5)
    lin = mf.linear_field(rng.normal(size=2), 0.4)
    for label, pos in [("box", g2.jittered_box(0.07, 0.3, 7)),
                       ("half-plane", g2.half_plane_cloud(0.07, 0.3, 7, axis=1, cut=0.7)[0]),
                       ("wedge-130", g2.wedge_cloud(130, 0.06, 0.3, 7)[0])]:
        lap = _lap_field(pos, h_of(0.07 if label != "wedge-130" else 0.06), lin)
        err = np.nanmax(np.abs(lap))
        rep.check_below(f"linear consistency Lap=0 ({label})", err, 1e-8, f"max |Lap(linear)| = {err:.2e}")

    # --- (b) boxed form == naive two-term form
    pos = g2.jittered_box(0.06, 0.3, 7)
    lb = _lap_field(pos, h_of(0.06), mf.TRIG2D, "boxed")
    ln = _lap_field(pos, h_of(0.06), mf.TRIG2D, "naive")
    diff = np.nanmax(np.abs(lb - ln))
    rep.check_below("boxed form == naive 2-term form", diff, 1e-9, f"max diff = {diff:.2e}")

    # --- (c) convergence (interior). IMPORTANT METHOD PROPERTY:
    # The renormalized Laplacian is only LINEAR-consistent. On a regular lattice
    # it is 2nd-order; under FIXED-RELATIVE jitter its pointwise truncation error
    # is O(1) (does not converge). The Poisson SOLUTION nevertheless supraconverges
    # (~order 1.3) -- that is measured in the capstone, not here.
    dxs = [0.06, 0.045, 0.033, 0.025]

    def lap_conv(field, jitter):
        errs = []
        for dx in dxs:
            pos = g2.jittered_box(dx, jitter, 7)
            mask = g2.box_interior_mask(pos, margin=h_of(dx))
            lap = _lap_field(pos, h_of(dx), field)
            ex = field.laplacian(pos)
            errs.append(np.linalg.norm((lap - ex)[mask]) / max(np.linalg.norm(ex[mask]), 1e-30))
        return errs, observed_order(dxs, errs)

    errs0, order0 = lap_conv(mf.TRIG2D, 0.0)
    rep.check_order("pointwise convergence on REGULAR lattice (trig)", order0, expected=2.0, slack=0.4,
                    detail=f"errs={['%.2e' % e for e in errs0]} order={order0:.2f}")
    errsj, orderj = lap_conv(mf.TRIG2D, 0.3)
    rep.check("jittered pointwise truncation is O(1) (linear-consistent only)", abs(orderj) < 0.4,
              f"order={orderj:.2f}, errs={['%.2e' % e for e in errsj]} -> solution supraconverges (see capstone)")

    # --- (c/pin) 3D constant Laplacian pins N_i = (2/d) tr(B)
    Q3 = np.array([[1.2, 0.3, -0.2], [0.3, -0.8, 0.4], [-0.2, 0.4, 0.6]])
    quad3 = mf.quadratic_field(Q3, [0.1, -0.2, 0.3], 0.0, "quad3d")
    trQ = float(np.trace(Q3))
    dx3, h3 = 0.13, 2.5 * 0.13
    pos3 = g2.jittered_box(dx3, 0.3, 7, d=3)
    nl3 = fc.neighbor_lists(pos3, h3)
    f3 = quad3.value(pos3)
    mask3 = np.all((pos3 > h3) & (pos3 < 1 - h3), axis=1)
    vals = []
    for i in np.where(mask3)[0]:
        if len(nl3[i]) < 4:
            continue
        xij = pos3[nl3[i]] - pos3[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h3)
        gm = fc.geom_quantities(xij, w)
        vals.append(fc.laplacian_interior(gm, xij, w, f3[nl3[i]] - f3[i]))
    mean_lap = float(np.mean(vals))
    rel = abs(mean_lap - trQ) / abs(trQ)
    rep.check("3D const-Laplacian pins N=(2/d)tr(B)", rel < 0.10,
              f"mean Lap = {mean_lap:.4f} vs tr(Q) = {trQ:.4f} (rel {rel:.1%}); bare tr(B) would give ~{1.5*trQ:.3f}")


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
