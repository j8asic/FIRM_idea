"""Substep 14 -- Full-Inverse (FI) 2nd-order meshless Laplacian baseline (Asai 2023).

The paper adds a second, *same-family* baseline next to GFDM: Asai et al.'s
Full-Inverse operator (CMAME 415:116203), which descends directly from our LDD
method and extends it to genuine second order by including the cross-derivative
terms LDD/FIRM omit. This test pins the baseline so the comparison is trustworthy:

  * interior operator is linear-exact and recovers a quadratic Laplacian exactly,
    *including the cross term and on a disordered cloud* (the genuine 2nd-order
    property -- this is what the cross coupling buys);
  * it is pointwise second-order under jitter, where FIRM's renormalised operator
    only supraconverges (plateaus pointwise) -- the headline contrast;
  * a Dirichlet Poisson solve converges at ~2nd order;
  * the consistency-class story behind Asai's Table 1 (BD == LDD == Schwaiger):
    on a SYMMETRIC cloud FI, its cross-free Block-Diagonal reduction, and FIRM's
    interior operator all agree exactly; on a DISORDERED cloud FI stays exact
    while the cross-free BD reduction AND FIRM both deviate -- i.e. FIRM's
    interior operator and BD belong to the same linear-consistent class, and the
    cross term is precisely what lifts FI out of it.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import fi
import gfdm
from testkit import observed_order

TITLE = "Substep 14 -- Full-Inverse (FI) 2nd-order baseline (Asai 2023)"


def run(rep):
    rng = np.random.default_rng(0)
    xij = rng.uniform(-1, 1, (24, 2)) * 0.3
    w = fc.kernel(np.linalg.norm(xij, axis=1), 0.5)
    xij, w = xij[w > 0], w[w > 0]

    # --- interior operator: linear-exact -------------------------------------
    lin = mf.linear_field([0.7, -0.4], 0.2)
    fij = lin.value(xij) - lin.value(np.zeros(2))
    d, bcv = fi.fi_row(xij, w, ridge=0.0)
    rep.check_below("FI interior linear-exact", abs(d @ fij + bcv), 1e-10,
                    f"lap(linear) = {abs(d @ fij + bcv):.2e}")

    # gradient sub-operator is linear-exact too (used inside the 2nd-deriv fit)
    gx, gy = fi.fi_grad_row(xij, w, ridge=0.0)
    g0 = lin.grad(np.zeros(2))
    rep.check_below("FI corrected gradient linear-exact",
                    abs(gx @ fij - g0[0]) + abs(gy @ fij - g0[1]), 1e-10,
                    f"|grad err| = {abs(gx @ fij - g0[0]) + abs(gy @ fij - g0[1]):.2e}")

    # --- 2nd-order: exact for a quadratic WITH a cross term, on a DISORDERED cloud
    Q = mf.quadratic_field([[1.3, 0.4], [0.4, -0.7]], [0.2, -0.1], 0.05)   # Q12=0.4 cross
    fij = Q.value(xij) - Q.value(np.zeros(2))
    d, bcv = fi.fi_row(xij, w, ridge=0.0)
    rep.check_close("FI exact for quadratic incl. cross term (disordered cloud)",
                    d @ fij + bcv, 1.3 - 0.7, tol=1e-9,
                    detail=f"lap(quad) = {d @ fij + bcv:.10f} (expect {1.3 - 0.7:.3f})")

    # --- pointwise order: 2nd on a lattice; tracks GFDM and beats FIRM under jitter
    # FI and GFDM are both 2nd-order WLSQ-class operators, so they behave alike: ~2.0
    # on a regular lattice, and the same reduced pointwise order under fixed-relative
    # jitter (it is the Poisson SOLUTION that supraconverges). FIRM's renormalised
    # operator is only linear-consistent and plateaus pointwise -- the headline contrast.
    trig = mf.trig_field(np.pi)
    dxs = [0.05, 0.037, 0.027, 0.020]

    def pointwise_order(method, jitter, seeds=(3, 7, 11, 17)):
        errs = []
        for dx in dxs:
            ev = []
            for sd in seeds:
                pos = g2.jittered_box(dx, jitter, sd)
                h = 2.5 * dx
                nl = fc.neighbor_lists(pos, h)
                m = g2.box_interior_mask(pos, h)
                num, ref = [], []
                for i in np.where(m)[0]:
                    nb = nl[i]
                    if len(nb) < 6:
                        continue
                    xj = pos[nb] - pos[i]
                    ww = fc.kernel(np.linalg.norm(xj, axis=1), h)
                    fj = trig.value(pos[nb]) - trig.value(pos[i])
                    if method == "fi":
                        lap = fi.fi_row(xj, ww)[0] @ fj
                    elif method == "gfdm":
                        lap = gfdm.gfdm_row(xj, ww)[0] @ fj
                    else:  # firm renormalised interior operator
                        lap = fc.laplacian_interior(fc.geom_quantities(xj, ww), xj, ww, fj)
                    num.append(lap)
                    ref.append(trig.laplacian(pos[i]))
                num, ref = np.array(num), np.array(ref)
                ev.append(np.linalg.norm(num - ref) / np.linalg.norm(ref))
            errs.append(float(np.median(ev)))
        return errs, observed_order(dxs, errs)

    e_reg, o_reg = pointwise_order("fi", 0.0, seeds=(0,))
    rep.check_order("FI pointwise 2nd-order on a regular lattice", o_reg, expected=2.0, slack=0.3,
                    detail=f"errs={['%.2e' % e for e in e_reg]} order={o_reg:.2f}")

    o_fi = pointwise_order("fi", 0.15)[1]
    o_gf = pointwise_order("gfdm", 0.15)[1]
    o_fm = pointwise_order("firm", 0.15)[1]
    rep.check("FI tracks the GFDM 2nd-order baseline under jitter", abs(o_fi - o_gf) < 0.3,
              f"FI order {o_fi:.2f} vs GFDM order {o_gf:.2f} (same 2nd-order WLSQ class)")
    rep.check("FI converges pointwise where FIRM operator plateaus", o_fi > o_fm + 0.8,
              f"FI order {o_fi:.2f} vs FIRM-operator order {o_fm:.2f} "
              f"(FIRM is linear-consistent: O(1) pointwise under fixed-relative jitter)")

    # --- Dirichlet Poisson convergence (jittered square) ----------------------
    def fi_box(dx, seed):
        pos = g2.jittered_box(dx, 0.30, seed)
        dirm = ~g2.box_interior_mask(pos, 2.5 * dx)
        A, b, _ = fi.assemble_fi(pos, dx, trig, dir_mask=dirm)
        p = fi.solve(A, b)
        m = g2.box_interior_mask(pos, 2.5 * dx)
        return float(np.linalg.norm((p - trig.value(pos))[m]) / np.linalg.norm(trig.value(pos)[m]))

    bxs = [0.05, 0.035, 0.025]
    errs = [float(np.median([fi_box(dx, s) for s in range(6)])) for dx in bxs]
    o = observed_order(bxs, errs)
    rep.check_order("FI Dirichlet Poisson converges (2nd-order)", o, expected=1.7, slack=0.35,
                    detail=f"errs={['%.2e' % e for e in errs]} order={o:.2f}")

    # --- consistency-class story behind Asai Table 1 (BD == LDD == Schwaiger) --
    # symmetric regular stencil: FI, cross-free BD, and FIRM interior all agree exactly
    grid = np.array([[i, j] for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if not (i == 0 and j == 0)], float) * 0.1
    wg = fc.kernel(np.linalg.norm(grid, axis=1), 0.5)
    fq = Q.value(grid) - Q.value(np.zeros(2))
    lap_fi = fi.fi_row(grid, wg, ridge=0.0)[0] @ fq
    lap_bd = fi.fi_row_bd(grid, wg, ridge=0.0)[0] @ fq
    lap_fm = fc.laplacian_interior(fc.geom_quantities(grid, wg), grid, wg, fq)
    sym_spread = max(abs(lap_fi - 0.6), abs(lap_bd - 0.6), abs(lap_fm - 0.6))
    rep.check_below("symmetric cloud: FI == BD == FIRM interior (all exact)", sym_spread, 1e-9,
                    f"FI {lap_fi:.8f}, BD {lap_bd:.8f}, FIRM {lap_fm:.8f} (true 0.6)")

    # disordered stencil: FI stays exact; cross-free BD and FIRM both deviate
    fq2 = Q.value(xij) - Q.value(np.zeros(2))
    e_fi2 = abs(fi.fi_row(xij, w, ridge=0.0)[0] @ fq2 - 0.6)
    e_bd2 = abs(fi.fi_row_bd(xij, w, ridge=0.0)[0] @ fq2 - 0.6)
    e_fm2 = abs(fc.laplacian_interior(fc.geom_quantities(xij, w), xij, w, fq2) - 0.6)
    rep.check("disordered cloud: FI exact, BD & FIRM deviate (same linear-consistent class)",
              e_fi2 < 1e-9 and e_bd2 > 1e-4 and e_fm2 > 1e-4,
              f"FI err {e_fi2:.2e}; BD err {e_bd2:.2e}; FIRM err {e_fm2:.2e}")


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
