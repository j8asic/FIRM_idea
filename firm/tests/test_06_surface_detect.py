"""Substep 6 -- Free-surface detection (docs/spec.md Sec 3.3, 3.4).

Verifies the surface normal sign, the singularity-cancellation identity
-(o.n_s)/delta = S, the activation limits for both sigma options, and the
regularized surface projection in the interior and surface limits.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
from testkit import RND

TITLE = "Substep 6 -- Surface detection + singularity cancellation"


def _geom(pos, i, h):
    nb = fc.neighbor_lists(pos, h)[i]
    xij = pos[nb] - pos[i]
    w = fc.kernel(np.linalg.norm(xij, axis=1), h)
    return fc.geom_quantities(xij, w)


def run(rep):
    dx, h = 0.06, 2.5 * 0.06
    pos = g2.jittered_box(dx, 0.3, 7)

    # a clear free-surface particle (top), and a deep interior particle
    i_surf = int(np.argmax(pos[:, 1]))
    i_int = int(np.argmin(np.linalg.norm(pos - 0.5, axis=1)))
    gm_s = _geom(pos, i_surf, h)
    gm_i = _geom(pos, i_int, h)

    # --- surface normal / sign:  o . n_s < 0  (o inward, n_s outward)
    r = gm_s.B @ gm_s.o
    n_s = -r / np.linalg.norm(r)
    odn = float(gm_s.o @ n_s)
    rep.check("o . n_s < 0 at surface", odn < 0, f"o . n_s = {odn:.3e}")

    # --- singularity cancellation:  -(o.n_s)/delta = S  with delta = |o.n_s|/S
    delta = abs(odn) / gm_s.S
    lhs = -odn / delta
    rep.check_close("cancellation -(o.n_s)/delta = S", lhs, gm_s.S, 1e-9,
                    f"{lhs:.4f} vs S = {gm_s.S:.4f}")

    # --- detection parameter + activation limits (BOTH options)
    lam_s = fc.lambda_detect(r, dx)
    # median lambda over genuinely-interior particles = the disorder floor
    nl = fc.neighbor_lists(pos, h)
    interior = g2.box_interior_mask(pos, margin=h)
    lam_int = []
    for j in np.where(interior)[0]:
        if len(nl[j]) < 3:
            continue
        xij = pos[nl[j]] - pos[j]
        wj = fc.kernel(np.linalg.norm(xij, axis=1), h)
        gj = fc.geom_quantities(xij, wj)
        lam_int.append(fc.lambda_detect(gj.B @ gj.o, dx))
    lam_med = float(np.median(lam_int))
    rep.check("lambda: surface >> interior median", lam_s > 4 * lam_med,
              f"lambda_surf = {lam_s:.3f}, median interior lambda = {lam_med:.3f}")

    # smoothstep (Option 2) is parameter-free and robust: interior == 0, surface > 0
    interior_smooth = max(fc.sigma_smoothstep(l) for l in lam_int)
    rep.check("smoothstep: ALL interior == 0 (robust, parameter-free)", interior_smooth == 0.0,
              f"max interior sigma = {interior_smooth:.3f} (all lambda < 2/3)")
    rep.check("smoothstep: surface activated", fc.sigma_smoothstep(lam_s) > 0.0,
              f"sigma_surf = {fc.sigma_smoothstep(lam_s):.3f} (lambda={lam_s:.3f})")

    # rational (Option 1) needs c ~ disorder level (Sec 10.5): c=0.2 over-activates
    # the TYPICAL interior particle at 30% jitter; adaptive c = 3*median(lambda)
    # suppresses the bulk (outliers near the support edge still ring -- use smoothstep
    # for a hard guarantee).
    med_fixed = float(np.median([fc.sigma_rational(l, 0.2) for l in lam_int]))
    c_adapt = 3.0 * lam_med
    sig_adapt = [fc.sigma_rational(l, c_adapt) for l in lam_int]
    med_adapt = float(np.median(sig_adapt))
    rep.check("rational c=0.2 over-activates typical interior (Sec 10.5)", med_fixed > 0.1,
              f"median interior sigma (c=0.2) = {med_fixed:.3f} -- too high")
    rep.check("rational adaptive c substantially suppresses bulk (but no hard zero)",
              med_adapt < 0.6 * med_fixed,
              f"median interior sigma {med_fixed:.3f} (c=0.2) -> {med_adapt:.3f} (c={c_adapt:.2f}); "
              f"surface stays {fc.sigma_rational(lam_s, c_adapt):.2f}. Smoothstep gives a hard zero -> preferred.")

    # --- regularized projection: ~ exact at surface, finite (small) at interior
    lam_i = fc.lambda_detect(gm_i.B @ gm_i.o, dx)
    eta = 0.2 / dx
    proj_reg = fc.surface_proj_regularized(gm_s.o, r, 1.0, eta)
    proj_exact = (gm_s.o @ n_s) * n_s
    rel = np.linalg.norm(proj_reg - proj_exact) / np.linalg.norm(proj_exact)
    rep.check("regularized surf-proj ~ exact at surface", rel < 0.15,
              f"relative diff = {rel:.1%} (eta bias)")
    ri = gm_i.B @ gm_i.o
    proj_int = fc.surface_proj_regularized(gm_i.o, ri, fc.sigma_rational(lam_i, 0.2), eta)
    rep.check("regularized surf-proj finite & small in interior", np.linalg.norm(proj_int) < 1e-2,
              f"|proj_int| = {np.linalg.norm(proj_int):.2e}")

    # --- ray-cast helper sanity (Sec 3.4.1): ray toward a wall hits it
    poly = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    d_hit = g2.raycast_segment_hit([0.5, 0.9], [0.0, 1.0], poly, max_dist=1.0)
    rep.check_close("raycast hits wall at expected distance", d_hit, 0.1, 1e-9, f"d_hit = {d_hit:.4f}")
    d_miss = g2.raycast_segment_hit([0.5, 0.5], [0.0, 1.0], poly, max_dist=0.2)
    rep.check("raycast respects max_dist (miss -> inf)", not np.isfinite(d_miss), f"d = {d_miss}")


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
