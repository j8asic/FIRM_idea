"""
Revision tooling (paper_rev_*): ADJOINTNESS DEFECT of the discrete FIRM
divergence/gradient pair and the Laplacian composition defect, quantifying the
paper's "non-adjointness" vocabulary.

Unit box, one resolution, 30% disorder (jitter 0.15). Operators exactly as in
hodge_projection.py: G stacks the renormalised component gradients (Gx; Gy)
(2N x N), D = [Gx, Gy] (N x 2N, the trace of the component-wise gradient), and
L is the FIRM trace-normalised Laplacian assembled on interior rows only (the
non-interior rows of both L and DG are zeroed, matching the interior-restricted
residual identity used in the Hodge study).

Reported (median over 3 seeds):
    ||G + D^T||_2 / ||G||_2      (continuous adjoint: div = -grad^T)
    ||L - D G||_2 / ||L||_2      (interior rows; L != DG defect)
2-norms estimated by power iteration on A^T A; Frobenius norms given as a
cross-check.

Writes figures/paper_extra_numbers.json  key 'adjointness_defect'.
Run:  python3 paper_rev_adjointness.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import firm_core as fc
import geometry2d as g2

import scipy.sparse as sps

OUT = os.path.join(HERE, "figures", "paper_extra_numbers.json")
DX = 0.025
JITTER = 0.15          # 30% disorder (chaos c = 0.30, jitter = c/2)
SEEDS = [7, 11, 19]
POWER_ITERS = 60


def build_ops(pos, h):
    """Sparse Gx, Gy (renormalised gradient rows, hodge_projection.grad_scalar)
    and the interior-row FIRM Laplacian L (trace normalisation, physical units)."""
    n = len(pos)
    nl = fc.neighbor_lists(pos, h)
    interior = g2.box_interior_mask(pos, h)
    rows_x, rows_y, rows_l = ([], [], []), ([], [], []), ([], [], [])

    def add(tr, i, j, v):
        tr[0].append(i); tr[1].append(j); tr[2].append(v)

    for i in range(n):
        nb = nl[i]
        if len(nb) < 3:
            continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        gm = fc.geom_quantities(xij, w)
        # gradient rows: grad f = B @ sum_j w f_ij x_ij  ->  row over (f_j - f_i)
        Grows = gm.B @ (w[:, None] * xij).T          # (2, K)
        for k, j in enumerate(nb):
            add(rows_x, i, int(j), Grows[0, k])
            add(rows_y, i, int(j), Grows[1, k])
        add(rows_x, i, i, -float(Grows[0].sum()))
        add(rows_y, i, i, -float(Grows[1].sum()))
        if interior[i]:
            wij = fc.correction_weights(xij, gm.B @ gm.o)
            d = gm.N * w * wij                       # physical-units Laplacian stencil
            for k, j in enumerate(nb):
                add(rows_l, i, int(j), float(d[k]))
            add(rows_l, i, i, -float(d.sum()))

    mk = lambda tr: sps.csr_matrix((tr[2], (tr[0], tr[1])), shape=(n, n))
    return mk(rows_x), mk(rows_y), mk(rows_l), interior


def norm2(A, iters=POWER_ITERS, seed=0):
    """2-norm estimate by power iteration on A^T A."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(iters):
        u = A @ v
        v = A.T @ u
        nv = np.linalg.norm(v)
        if nv == 0:
            return 0.0
        v /= nv
    u = A @ v
    return float(np.linalg.norm(u))


def one_seed(seed):
    pos = g2.jittered_box(DX, JITTER, seed)
    n = len(pos)
    h = 2.5 * DX
    Gx, Gy, L, interior = build_ops(pos, h)
    G = sps.vstack([Gx, Gy]).tocsr()                 # (2n, n)
    D = sps.hstack([Gx, Gy]).tocsr()                 # (n, 2n)
    S = (G + D.T).tocsr()                            # adjointness defect
    DG = (D @ G).tocsr()
    mask = sps.diags(interior.astype(float))
    E = (L - mask @ DG).tocsr()                      # interior-row composition defect

    fro = lambda A: float(sps.linalg.norm(A))
    out = dict(
        N=n, N_interior=int(interior.sum()),
        adjoint_defect_2norm=norm2(S) / norm2(G),
        adjoint_defect_fro=fro(S) / fro(G),
        composition_defect_2norm=norm2(E) / norm2(L),
        composition_defect_fro=fro(E) / fro(L),
        norm_G_2=norm2(G), norm_L_2=norm2(L),
    )
    return out


def run():
    per_seed = [one_seed(s) for s in SEEDS]
    med = {k: float(np.median([r[k] for r in per_seed])) for k in per_seed[0]}
    med["N"] = int(med["N"]); med["N_interior"] = int(med["N_interior"])
    for s, r in zip(SEEDS, per_seed):
        print(f"seed {s}: ||G+D^T||/||G|| = {r['adjoint_defect_2norm']:.3f} (fro {r['adjoint_defect_fro']:.3f})   "
              f"||L-DG||/||L|| = {r['composition_defect_2norm']:.3f} (fro {r['composition_defect_fro']:.3f})")
    print(f"\nmedian: ||G+D^T||/||G|| = {med['adjoint_defect_2norm']:.3f}   "
          f"||L-DG||/||L|| = {med['composition_defect_2norm']:.3f}")
    return dict(per_seed={str(s): r for s, r in zip(SEEDS, per_seed)}, median=med)


def save(key, payload):
    d = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            d = json.load(f)
    d[key] = payload
    with open(OUT, "w") as f:
        json.dump(d, f, indent=2)
    print(f"\nsaved '{key}' -> {OUT}")


if __name__ == "__main__":
    meta = dict(domain="unit box", dx=DX, jitter=JITTER, disorder_pct=30, seeds=SEEDS,
                operators=("G = stacked renormalised gradients (2N x N); D = [Gx, Gy]; "
                           "L = FIRM trace-normalised Laplacian, interior rows only "
                           "(DG restricted to the same rows)"),
                norms=f"2-norm via {POWER_ITERS} power iterations on A^T A; Frobenius cross-check")
    res = run()
    save("adjointness_defect", dict(meta=meta, results=res))
