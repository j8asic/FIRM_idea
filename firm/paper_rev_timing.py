"""
Revision tooling (paper_rev_*): TIMINGS for the five schemes of the paper's
Table `tab:schemes`, measured on the all-Neumann unit box (the Table-1 benchmark;
the harness provides an assemble path for every scheme there, unlike the tank,
which GFDM/FI cannot run).

  a  trace-normalised (smooth-naive, norm='trb') + projection Neumann closure
  b  trace-normalised + algebraic-ghost closure
  c  sum-normalised   (norm='denom')  + algebraic-ghost closure
  d  GFDM + KKT constraint-row Neumann
  e  Full Inverse (Asai FI) + algebraic-ghost closure

Note: row (a) of tab:schemes also lists the Robin *surface* closure; the box has
no free surface, so these timings cover the interior operator + the Neumann wall
closure only (the Robin surface diagonal is O(1) extra work per surface node).

Measured per scheme x resolution (median of 3 repetitions, fixed seed 0, 30%
disorder = jitter 0.15):
  * wall-clock assembly time            (bvp/gfdm/fi assemble)
  * wall-clock direct solve time        (scipy spsolve, the suite's default)
  * ILU(0)-preconditioned BiCGSTAB      (scipy spilu + bicgstab, rtol 1e-8),
    iteration count via callback, ILU setup + Krylov times
Times are also reported per node (microseconds/node).

Writes figures/paper_extra_numbers.json  key 'timings'.
Run:  python3 paper_rev_timing.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bvp
import fi
import gfdm
import geometry2d as g2
import manufactured as mf

import scipy.sparse as sps
from scipy.sparse.linalg import spsolve, spilu, bicgstab, LinearOperator

OUT = os.path.join(HERE, "figures", "paper_extra_numbers.json")
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
FIELD = mf.complex_field(np.pi, 0.3)
DXS = [0.05, 0.025, 0.0125]       # ~400 / ~1600 / ~6400 nodes
JITTER = 0.15                     # chaos/disorder 30% (jitter = c/2)
SEED = 0
REPS = 3
RTOL = 1e-8
MAXITER = 2000

SCHEMES = [
    ("a_trace_projection", "trace-normalised + projection"),
    ("b_trace_ghost", "trace-normalised + ghost"),
    ("c_sum_ghost", "sum-normalised + ghost"),
    ("d_gfdm_constraint", "GFDM + constraint row"),
    ("e_fi_ghost", "Full Inverse + ghost"),
]


def _assemble(scheme, pos, dx, pin):
    if scheme == "a_trace_projection":
        return bvp.assemble(pos, dx, FIELD, SQUARE, "neumann",
                            wall_closure="projection", norm="trb", pin=pin)
    if scheme == "b_trace_ghost":
        return bvp.assemble(pos, dx, FIELD, SQUARE, "neumann",
                            wall_closure="ghost", norm="trb", pin=pin)
    if scheme == "c_sum_ghost":
        return bvp.assemble(pos, dx, FIELD, SQUARE, "neumann",
                            wall_closure="ghost", norm="denom", pin=pin)
    if scheme == "d_gfdm_constraint":
        return gfdm.assemble_gfdm(pos, dx, FIELD, poly=SQUARE,
                                  neumann="constraint", pin=pin)
    if scheme == "e_fi_ghost":
        return fi.assemble_fi(pos, dx, FIELD, poly=SQUARE, pin=pin)
    raise ValueError(scheme)


def _krylov(A, b):
    """ILU-preconditioned BiCGSTAB; falls back to Jacobi if spilu fails.
    Returns (precond_name, t_setup, t_krylov, iters, flag, relres)."""
    Acsc = A.tocsc()
    t0 = time.perf_counter()
    try:
        ilu = spilu(Acsc, drop_tol=1e-5, fill_factor=10.0)
        M = LinearOperator(A.shape, ilu.solve)
        precond = "ILU(spilu)"
    except Exception as exc:  # singular pivot etc. -> Jacobi
        diag = Acsc.diagonal()
        diag = np.where(np.abs(diag) > 1e-300, diag, 1.0)
        M = LinearOperator(A.shape, lambda x: x / diag)
        precond = f"Jacobi (spilu failed: {type(exc).__name__})"
    t_setup = time.perf_counter() - t0

    it = [0]

    def cb(xk):
        it[0] += 1

    t0 = time.perf_counter()
    x, flag = bicgstab(A.tocsr(), b, rtol=RTOL, atol=0.0, maxiter=MAXITER,
                       M=M, callback=cb)
    t_krylov = time.perf_counter() - t0
    relres = float(np.linalg.norm(A @ x - b) / max(np.linalg.norm(b), 1e-300))
    return precond, t_setup, t_krylov, it[0], int(flag), relres


def run():
    results = {}
    for dx in DXS:
        pos = g2.jittered_box(dx, JITTER, SEED)
        n = len(pos)
        pin = int(np.argmin(np.linalg.norm(pos - 0.5, axis=1)))
        interior = g2.box_interior_mask(pos, 2.5 * dx)
        pe = FIELD.value(pos)
        print(f"\ndx = {dx}  (N = {n})")
        for scheme, label in SCHEMES:
            t_asm, t_dir, kry = [], [], []
            err = None
            nnz = 0
            for rep in range(REPS):
                t0 = time.perf_counter()
                A, b, info = _assemble(scheme, pos, dx, pin)
                t_asm.append(time.perf_counter() - t0)
                nnz = A.nnz
                t0 = time.perf_counter()
                p = spsolve(A.tocsr(), b)
                t_dir.append(time.perf_counter() - t0)
                kry.append(_krylov(A, b))
                if err is None:  # sanity: mean-removed interior rel-L2
                    e = (p - pe)[interior]
                    r = pe[interior]
                    e = e - e.mean(); r = r - r.mean()
                    err = float(np.linalg.norm(e) / np.linalg.norm(r))
            med = lambda v: float(np.median(v))
            iters = med([k[3] for k in kry])
            row = dict(
                label=label, N=n, nnz=int(nnz),
                assembly_s=med(t_asm), solve_direct_s=med(t_dir),
                assembly_us_per_node=med(t_asm) / n * 1e6,
                solve_direct_us_per_node=med(t_dir) / n * 1e6,
                ilu_setup_s=med([k[1] for k in kry]),
                krylov_s=med([k[2] for k in kry]),
                krylov_us_per_node=(med([k[1] for k in kry]) + med([k[2] for k in kry])) / n * 1e6,
                bicgstab_iters=iters,
                bicgstab_flag=int(kry[-1][4]), bicgstab_relres=float(kry[-1][5]),
                preconditioner=kry[-1][0],
                interior_relL2_mean_removed=err,
            )
            results.setdefault(scheme, {})[f"dx{dx}"] = row
            print(f"  {label:34s} asm {row['assembly_us_per_node']:8.1f} us/node   "
                  f"direct {row['solve_direct_us_per_node']:7.2f} us/node   "
                  f"bicgstab {iters:5.0f} it (flag {row['bicgstab_flag']})   "
                  f"err {err:.2e}")
    return results


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
    meta = dict(benchmark="all-Neumann unit box (Table-1 setup), complex field",
                dxs=DXS, jitter=JITTER, disorder_pct=30, seed=SEED,
                repetitions=REPS, statistic="median of repetitions",
                krylov=f"BiCGSTAB rtol={RTOL} + spilu(drop_tol=1e-5, fill_factor=10)",
                note=("Row (a) of tab:schemes includes the Robin surface closure; the box "
                      "has no free surface, so timings cover interior + Neumann closure."))
    res = run()
    save("timings", dict(meta=meta, results=res))
