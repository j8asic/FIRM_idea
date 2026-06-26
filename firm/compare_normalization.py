"""
Compare two Laplacian normalizations for FIRM:
  trb   :  N = (2/d) tr(B)                              (current v3 choice)
  denom :  N = 2d / sum_j W_ij |x_ij|^2 w_ij            (the 'old' denominator form,
                                                         w_ij = 1 - x_ij . B o)

Both share the SAME numerator  sum_j W_ij f_ij w_ij.  'denom' is exact for the
isotropic quadratic |x|^2 by construction; they coincide for an isotropic moment
matrix (regular lattice) and differ under anisotropy/jitter.

Tests: (A) pointwise Laplacian truncation error (regular + jittered, 2D + 3D);
       (B) all-Dirichlet manufactured Poisson solution convergence (the normalization
           only rescales the per-row source f_i/N_i, so this isolates its effect).
"""
import numpy as np

import firm_core as fc
import geometry2d as g2
import manufactured as mf
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve


def N_denom(xij, w, V, d):
    wij = fc.correction_weights(xij, V)
    den = float((w * (xij * xij).sum(1) * wij).sum())
    return (2.0 * d) / den if abs(den) > 1e-30 else np.nan


def lap_at(xij, w, fij, norm):
    gm = fc.geom_quantities(xij, w)
    V = gm.B @ gm.o
    wij = fc.correction_weights(xij, V)
    num = float((w * fij * wij).sum())
    N = gm.N if norm == "trb" else N_denom(xij, w, V, gm.d)
    return N * num


# ----------------------------------------------------------------- (A) pointwise
def pointwise(field, dxs, jitter, d=2, h_factor=2.5):
    errs = {"trb": [], "denom": []}
    for dx in dxs:
        h = h_factor * dx
        pos = g2.jittered_box(dx, jitter, 7, d=d)
        nl = fc.neighbor_lists(pos, h)
        f = field.value(pos)
        ex = field.laplacian(pos)
        mask = np.all((pos > h) & (pos < 1 - h), axis=1)
        acc = {"trb": [], "denom": []}
        idx = []
        for i in np.where(mask)[0]:
            if len(nl[i]) < d + 1:
                continue
            xij = pos[nl[i]] - pos[i]
            w = fc.kernel(np.linalg.norm(xij, axis=1), h)
            fij = f[nl[i]] - f[i]
            for nm in ("trb", "denom"):
                acc[nm].append(lap_at(xij, w, fij, nm))
            idx.append(i)
        exm = ex[idx]
        for nm in ("trb", "denom"):
            errs[nm].append(np.linalg.norm(np.array(acc[nm]) - exm) / max(np.linalg.norm(exm), 1e-30))
    return errs


# ----------------------------------------------------------------- (B) Poisson
def poisson_dirichlet(field, dxs, jitter, norm, d=2, h_factor=2.5):
    out = []
    for dx in dxs:
        h = h_factor * dx
        pos = g2.jittered_box(dx, jitter, 7, d=d)
        N = len(pos)
        nl = fc.neighbor_lists(pos, h)
        inter = g2.box_interior_mask(pos, h)
        R, C, D = [], [], []
        b = np.zeros(N)
        for i in range(N):
            if not inter[i]:
                R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
            xij = pos[nl[i]] - pos[i]
            w = fc.kernel(np.linalg.norm(xij, axis=1), h)
            gm = fc.geom_quantities(xij, w)
            V = gm.B @ gm.o
            wij = fc.correction_weights(xij, V)
            Nrm = gm.N if norm == "trb" else N_denom(xij, w, V, gm.d)
            wsum = float((w * wij).sum())
            for k, j in enumerate(nl[i]):
                R += [i, i]; C += [j, i]; D += [w[k] * wij[k], -w[k] * wij[k]]
            b[i] = field.laplacian(pos[i]) / Nrm
        A = sps.csr_matrix((D, (R, C)), shape=(N, N))
        p = spsolve(A, b)
        pe = field.value(pos)
        out.append(np.linalg.norm((p - pe)[inter]) / np.linalg.norm(pe[inter]))
    return out


def order(dxs, e):
    return float(np.polyfit(np.log(dxs), np.log(e), 1)[0])


if __name__ == "__main__":
    dxs2 = [0.06, 0.045, 0.033, 0.025]
    print("=== (A) pointwise Laplacian truncation error (rel L2, interior) ===")
    for fld in (mf.TRIG2D, mf.QUAD2D):
        for jit in (0.0, 0.3):
            e = pointwise(fld, dxs2, jit)
            print(f"  {fld.name:5s} jitter={jit:>3}:  "
                  f"trb {[f'{v:.2e}' for v in e['trb']]} (o {order(dxs2, e['trb']):.2f})   "
                  f"denom {[f'{v:.2e}' for v in e['denom']]} (o {order(dxs2, e['denom']):.2f})")
    # 3D quadratic (constant Laplacian) -- magnitude check
    Q3 = np.array([[1.2, 0.3, -0.2], [0.3, -0.8, 0.4], [-0.2, 0.4, 0.6]])
    q3 = mf.quadratic_field(Q3, [0.1, -0.2, 0.3], 0.0, "quad3d")
    e3 = pointwise(q3, [0.13, 0.10, 0.08], 0.3, d=3)
    print(f"  quad3d jitter=0.3: trb {[f'{v:.2e}' for v in e3['trb']]}   denom {[f'{v:.2e}' for v in e3['denom']]}")

    print("\n=== (B) all-Dirichlet Poisson solution (rel L2, interior) ===")
    for fld in (mf.TRIG2D, mf.complex_field(np.pi, 0.3)):
        for jit in (0.0, 0.3):
            et = poisson_dirichlet(fld, dxs2, jit, "trb")
            ed = poisson_dirichlet(fld, dxs2, jit, "denom")
            print(f"  {fld.name:7s} jitter={jit:>3}:  "
                  f"trb {[f'{v:.2e}' for v in et]} (o {order(dxs2, et):.2f})   "
                  f"denom {[f'{v:.2e}' for v in ed]} (o {order(dxs2, ed):.2f})")
