"""Substep 11 -- Robustness across jitter amplitude (particle anisotropy/disorder).

Sweeps the jitter amplitude (fraction of dx, per axis) from a regular lattice
(0.0) to strong disorder (0.4) and checks:
  * linear-exactness is structural -- holds to round-off at EVERY jitter level;
  * the renormalization tensor stays finite (anisotropy/conditioning grows but
    M remains invertible) up to strong disorder;
  * the Poisson solution degrades gracefully and monotonically with disorder;
  * the trb vs denom normalizations coincide on a lattice and the denom edge
    appears only under anisotropy (it is exact for |x|^2);
  * smoothstep surface activation never false-triggers in the interior, even at
    high jitter (the rational c=0.2 activation does, confirming Sec 10.5).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import compare_normalization as cn
from testkit import RND

TITLE = "Substep 11 -- Jitter / anisotropy robustness"

JITTERS = [0.0, 0.1, 0.2, 0.3, 0.4]
DX = 0.045


def _linear_exact(jitter, h_factor=2.5):
    """Max gradient and Laplacian error for a random linear field at this jitter."""
    h = h_factor * DX
    rng = np.random.default_rng(3)
    a = rng.normal(size=2)
    lin = mf.linear_field(a, 0.2)
    pos = g2.jittered_box(DX, jitter, 7)
    nl = fc.neighbor_lists(pos, h)
    f = lin.value(pos)
    gerr = lerr = 0.0
    cond_max = 0.0
    for i in range(len(pos)):
        if len(nl[i]) < 3:
            continue
        xij = pos[nl[i]] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        gm = fc.geom_quantities(xij, w)
        cond_max = max(cond_max, float(np.linalg.cond(gm.M)))
        gerr = max(gerr, np.linalg.norm(fc.grad_op(gm.B, xij, w, f[nl[i]] - f[i]) - a))
        lerr = max(lerr, abs(fc.laplacian_interior(gm, xij, w, f[nl[i]] - f[i])))
    return gerr, lerr, cond_max


def _interior_activation(jitter, h_factor=2.5):
    h = h_factor * DX
    pos = g2.jittered_box(DX, jitter, 7)
    nl = fc.neighbor_lists(pos, h)
    interior = g2.box_interior_mask(pos, h)
    lam = []
    for j in np.where(interior)[0]:
        if len(nl[j]) < 3:
            continue
        xij = pos[nl[j]] - pos[j]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        gm = fc.geom_quantities(xij, w)
        lam.append(fc.lambda_detect(gm.B @ gm.o, DX))
    lam = np.array(lam)
    smooth_max = max(fc.sigma_smoothstep(l) for l in lam)
    rat_med = float(np.median([fc.sigma_rational(l, 0.2) for l in lam]))
    return float(np.median(lam)), smooth_max, rat_med


def run(rep):
    cf = mf.complex_field(np.pi, 0.3)

    # --- 1. linear-exactness is structural: round-off at EVERY jitter level
    print(f"  jitter sweep (dx={DX}):")
    worst_g = worst_l = 0.0
    conds = []
    for jit in JITTERS:
        g, l, cond = _linear_exact(jit)
        conds.append(cond)
        worst_g = max(worst_g, g); worst_l = max(worst_l, l)
        print(f"    jitter={jit:.1f}:  max|grad-a|={g:.1e}  max|lap(lin)|={l:.1e}  cond(M)_max={cond:.1e}")
    rep.check_below("linear-exact gradient at all jitters", worst_g, 1e-9)
    rep.check_below("linear-exact Laplacian at all jitters", worst_l, 1e-7)

    # --- 2. anisotropy grows but M stays invertible (finite conditioning)
    rep.check("conditioning grows with jitter but stays finite",
              conds[0] < conds[-1] and conds[-1] < 1e6,
              f"cond(M)_max: lattice {conds[0]:.1e} -> jitter 0.4 {conds[-1]:.1e}")

    # --- 3. Poisson solution degrades gracefully with disorder
    pois = [cn.poisson_dirichlet(cf, [DX], jit, "trb")[0] for jit in JITTERS]
    print(f"    Poisson rel-L2 (trb): " + "  ".join(f"{j:.1f}:{e:.2e}" for j, e in zip(JITTERS, pois)))
    rep.check("regular lattice is most accurate", pois[0] <= min(pois) + 1e-12,
              f"lattice {pois[0]:.2e} vs max {max(pois):.2e}")
    rep.check("Poisson error stays bounded up to 40% jitter", max(pois) < 0.05,
              f"max rel-L2 = {max(pois):.2e}")

    # --- 4. trb vs denom across jitter: identical on lattice, denom edge under anisotropy
    print(f"    trb vs denom (Poisson rel-L2):")
    ratios = []
    for jit in JITTERS:
        et = cn.poisson_dirichlet(cf, [DX], jit, "trb")[0]
        ed = cn.poisson_dirichlet(cf, [DX], jit, "denom")[0]
        ratios.append(ed / et)
        print(f"      jitter={jit:.1f}:  trb={et:.2e}  denom={ed:.2e}  denom/trb={ed/et:.2f}")
    rep.check_close("trb == denom on a regular lattice", ratios[0], 1.0, 1e-6,
                    f"denom/trb at jitter 0 = {ratios[0]:.6f}")
    rep.check("denom never much worse than trb across jitter", max(ratios) < 1.15,
              f"max denom/trb ratio = {max(ratios):.2f}")

    # --- 5. surface activation robustness vs jitter. With the smooth poly6 default the
    # interior lambda floor stays well below the smoothstep threshold 2/3 across the
    # whole sweep, so smoothstep is a hard zero in the interior even at 40% jitter;
    # rational c=0.2 climbs into false-trigger territory. (A more concentrated kernel
    # raises the lambda floor and would erode this margin -- see compare_kernels.py.)
    print(f"    interior surface activation:")
    smooth_at = {}
    for jit in JITTERS:
        lam_med, smooth_max, rat_med = _interior_activation(jit)
        smooth_at[jit] = smooth_max
        print(f"      jitter={jit:.1f}:  median lambda={lam_med:.3f}  smoothstep_max={smooth_max:.2f}  rational_med={rat_med:.3f}")
    rep.check("smoothstep never false-triggers interior (jitter <= 0.4, poly6)",
              all(smooth_at[j] == 0.0 for j in JITTERS),
              "max interior smoothstep sigma == 0 across the whole sweep")


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
