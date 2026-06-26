"""
Kernel & support-radius study: pick the best weight at the SMALLEST robust compact
support, for the disorder levels relevant to incompressible flow.

Kernels are compared at MATCHED compact support h (= kappa*dx), shape written in the
support-normalized coordinate s = r/h in [0,1):
    spiky3    (1-s)^3                     (= the chosen (1-0.4 r/dx)^3 at h=2.5dx)
    poly6     (1-s^2)^3                   (smooth, flat-topped; "scaled to h")
    wendlandC2(1-s)^4 (1+4s)              (standard robust SPH kernel)

Metric: all-Dirichlet manufactured complex Poisson on a jittered box (clean interior
measure of solution accuracy), multi-seed median rel-L2, at two dx (convergence
indicator) and a high-jitter stress point, plus the minimum neighbour count (cost).
"""
import numpy as np
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve

CF = mf.complex_field(np.pi, 0.3)

KERNELS = {
    "spiky3   (1-s)^3":      lambda s: (1.0 - s) ** 3,
    "poly6    (1-s^2)^3":    lambda s: (1.0 - s * s) ** 3,
    "wendC2 (1-s)^4(1+4s)":  lambda s: (1.0 - s) ** 4 * (1.0 + 4.0 * s),
}
SUPPORTS = [2.0, 2.25, 2.5, 3.0]


def solve_dirichlet(dx, kappa, kfn, jitter, seed):
    h = kappa * dx
    pos = g2.jittered_box(dx, jitter, seed)
    N = len(pos)
    nl = fc.neighbor_lists(pos, h)
    inter = g2.box_interior_mask(pos, h)
    R, C, D = [], [], []
    b = np.zeros(N)
    min_nb = 10 ** 9
    for i in range(N):
        if not inter[i]:
            R.append(i); C.append(i); D.append(1.0); b[i] = CF.value(pos[i]); continue
        nb = nl[i]
        min_nb = min(min_nb, len(nb))
        xij = pos[nb] - pos[i]
        s = np.linalg.norm(xij, axis=1) / h
        w = np.where(s < 1.0, kfn(s), 0.0)
        gm = fc.geom_quantities(xij, w)
        wij = fc.correction_weights(xij, gm.B @ gm.o)
        for k, j in enumerate(nb):
            R += [i, i]; C += [j, i]; D += [w[k] * wij[k], -w[k] * wij[k]]
        b[i] = CF.laplacian(pos[i]) / gm.N
    A = sps.csr_matrix((D, (R, C)), shape=(N, N))
    p = spsolve(A, b)
    pe = CF.value(pos)
    return np.linalg.norm((p - pe)[inter]) / np.linalg.norm(pe[inter]), min_nb


def med(dx, kappa, kfn, jitter, seeds):
    vals = [solve_dirichlet(dx, kappa, kfn, jitter, s) for s in seeds]
    return float(np.median([v[0] for v in vals])), min(v[1] for v in vals)


if __name__ == "__main__":
    seeds = [7, 11, 19, 23, 31]
    dxc, dxf = 0.05, 0.033
    print("All-Dirichlet complex Poisson, median over %d seeds." % len(seeds))
    print("err30c/f = rel-L2 at jitter 0.30 (dx=%.3f / %.3f);  err45 = jitter 0.45 (dx=%.3f).\n" % (dxc, dxf, dxf))
    print(f"  {'kernel':22s}{'h/dx':>6}{'minNb':>7}{'err30_c':>11}{'err30_f':>11}{'conv?':>7}{'err45':>11}")
    best = None
    for kname, kfn in KERNELS.items():
        for kap in SUPPORTS:
            ec, nb = med(dxc, kap, kfn, 0.30, seeds)
            ef, _ = med(dxf, kap, kfn, 0.30, seeds)
            e45, nb45 = med(dxf, kap, kfn, 0.45, seeds)
            conv = "yes" if ef < ec else "NO"
            print(f"  {kname:22s}{kap:6.2f}{nb:7d}{ec:11.2e}{ef:11.2e}{conv:>7}{e45:11.2e}")
            # candidate score: accurate & converging & enough neighbours, smallest support
            if ef < ec and nb >= 6:
                score = (kap, ef)  # prefer small support, then low error
                if best is None or score < best[0]:
                    best = (score, kname, kap, ef, nb)
        print()
    if best:
        _, kname, kap, ef, nb = best
        print(f"  -> smallest robust support (converging, >=6 nbrs): {kname} at h={kap}dx "
              f"(min {nb} nbrs, err30_f={ef:.2e})")
