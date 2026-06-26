"""Substep 12 -- Robustness to kernel shape and support radius.

The FIRM operators consume the kernel weights as input and renormalize by B_i, so
they are (in principle) insensitive to the kernel's normalization and fairly robust
to its shape and to the support radius h = h_factor * dx. This test sweeps:
  * support radius h_factor in {1.8, 2.0, 2.2, 2.6}
  * kernel shape: poly6 (1-q^2)^3 (smooth, the default), spiky (1-q)^3, and the
    truncated "spiky" kernel (1-0.4q)^3 (nonzero/discontinuous at the support edge),
and checks linear-exactness (structural, must hold for every combo), neighbour
counts, pointwise Laplacian error, and Poisson solution convergence.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from testkit import observed_order

TITLE = "Substep 12 -- Kernel shape & support radius robustness"

KERNELS = {
    # chosen DEFAULT (kernel/support study, compare_kernels.py): poly6 wins at small support
    "poly6 (1-q^2)^3": lambda q: np.where(q < 1.0, (1.0 - q * q) ** 3, 0.0),
    # spiky cubic (1-q)^3 == (1-0.4 r/dx)^3 at h=2.5dx -- competitive only at large support
    "spiky (1-q)^3": lambda q: np.where(q < 1.0, (1.0 - q) ** 3, 0.0),
    # edge-truncated variant (1-0.4q)^3 with q=r/h: nonzero (0.216) at the cutoff
    "trunc (1-0.4q)^3": lambda q: np.where(q < 1.0, np.maximum(0.0, 1.0 - 0.4 * q) ** 3, 0.0),
}
RADII = [1.8, 2.0, 2.2, 2.6]
JIT = 0.3


def _weights(xij, h, kfn):
    return kfn(np.linalg.norm(xij, axis=1) / h)


def _linear_and_pointwise(dx, hf, kfn, field):
    """Returns (max grad err, max lap(linear) err, pointwise lap rel-L2, min #neighbours)."""
    h = hf * dx
    pos = g2.jittered_box(dx, JIT, 7)
    nl = fc.neighbor_lists(pos, h)
    rng = np.random.default_rng(3)
    a = rng.normal(size=2)
    lin = mf.linear_field(a, 0.2)
    fl = lin.value(pos)
    ff = field.value(pos)
    ex = field.laplacian(pos)
    mask = g2.box_interior_mask(pos, h)
    gerr = lerr = 0.0
    minnb = 999
    num, den = [], []
    for i in range(len(pos)):
        nb = nl[i]
        minnb = min(minnb, len(nb))
        if len(nb) < 3:
            continue
        xij = pos[nb] - pos[i]
        w = _weights(xij, h, kfn)
        gm = fc.geom_quantities(xij, w)
        gerr = max(gerr, np.linalg.norm(fc.grad_op(gm.B, xij, w, fl[nb] - fl[i]) - a))
        lerr = max(lerr, abs(fc.laplacian_interior(gm, xij, w, fl[nb] - fl[i])))
        if mask[i]:
            num.append(fc.laplacian_interior(gm, xij, w, ff[nb] - ff[i]))
            den.append(ex[i])
    rel = np.linalg.norm(np.array(num) - np.array(den)) / max(np.linalg.norm(den), 1e-30)
    return gerr, lerr, rel, minnb


def _poisson_dir(dx, hf, kfn, field, seed=7):
    h = hf * dx
    pos = g2.jittered_box(dx, JIT, seed)
    N = len(pos)
    nl = fc.neighbor_lists(pos, h)
    inter = g2.box_interior_mask(pos, h)
    R, C, D = [], [], []
    b = np.zeros(N)
    for i in range(N):
        if not inter[i]:
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        xij = pos[nl[i]] - pos[i]
        w = _weights(xij, h, kfn)
        gm = fc.geom_quantities(xij, w)
        wij = fc.correction_weights(xij, gm.B @ gm.o)
        for k, j in enumerate(nl[i]):
            R += [i, i]; C += [j, i]; D += [w[k] * wij[k], -w[k] * wij[k]]
        b[i] = field.laplacian(pos[i]) / gm.N
    A = sps.csr_matrix((D, (R, C)), shape=(N, N))
    p = spsolve(A, b)
    pe = field.value(pos)
    return np.linalg.norm((p - pe)[inter]) / np.linalg.norm(pe[inter])


def run(rep):
    cf = mf.complex_field(np.pi, 0.3)
    dxs = [0.06, 0.045, 0.033]
    seeds = [7, 11, 19]
    worst_g = worst_l = 0.0
    worst_err = 0.0
    print("  (Poisson errors are median over seeds; min nb = smallest neighbour count seen)")

    for kname, kfn in KERNELS.items():
        print(f"  kernel {kname}:")
        for hf in RADII:
            g, l, ptw, minnb = _linear_and_pointwise(0.045, hf, kfn, cf)
            worst_g = max(worst_g, g); worst_l = max(worst_l, l)
            errs = [float(np.median([_poisson_dir(dx, hf, kfn, cf, s) for s in seeds])) for dx in dxs]
            worst_err = max(worst_err, errs[-1])
            o = observed_order(dxs, errs)
            tag = "" if minnb >= 5 else "  <- under-supported (min nb < 5)"
            print(f"    h={hf}dx (min nb {minnb:2d}): lin grad {g:.0e} lap {l:.0e} | "
                  f"ptw lap rel {ptw:.2e} | Poisson rel-L2 {errs[-1]:.2e} (order {o:.2f}){tag}")
            if abs(hf - 2.6) < 1e-9:   # well-supported radius: convergence must be clean
                rep.check(f"clean convergence at h=2.6dx [{kname.split()[0]}]", o > 0.9,
                          f"order {o:.2f}, finest rel-L2 {errs[-1]:.2e}")

    # structural / robust facts that must hold for EVERY kernel and radius
    rep.check_below("linear-exact gradient (all kernels & radii)", worst_g, 1e-9)
    rep.check_below("linear-exact Laplacian (all kernels & radii)", worst_l, 1e-7)
    rep.check("Poisson error bounded for all kernels & radii", worst_err < 0.05,
              f"max finest rel-L2 = {worst_err:.2e}")
    print("  Finding: operators are linear-exact for any kernel/radius; clean CONVERGENCE")
    print("  needs adequate support (h>=~2.5dx at 30% jitter -> min ~6 neighbours). Small")
    print("  radii (1.8-2.2dx) hit the 2D rank floor (3 nbrs) under jitter and get noisy.")
    print("  poly6 (flat-topped) is the best small-support kernel (see compare_kernels.py);")
    print("  the spiky (1-q)^3 is competitive only at larger support.")


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
