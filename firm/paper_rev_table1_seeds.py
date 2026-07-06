"""
Revision tooling (paper_rev_*): SEED SPREAD for the paper's Table `tab:neumann`
(all-Neumann unit box, 30% disorder, Delta = 0.050/0.035/0.025/0.018, closures:
FIRM projection, FIRM ghost, GFDM constraint, GFDM penalty).

Reruns exactly paper_benchmarks._neumann_box (the Table-1 computation) with 24
random seeds per resolution and reports the median, 25th and 75th percentile of
the interior mean-removed relative L2 error per cell, to quantify the seed
scatter behind the non-monotone FIRM-ghost column.

Writes figures/paper_extra_numbers.json  key 'table1_seed_spread'.
Run:  python3 paper_rev_table1_seeds.py
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import manufactured as mf
import paper_benchmarks as pb
from convergence import jitter_for_chaos

OUT = os.path.join(HERE, "figures", "paper_extra_numbers.json")
FIELD = mf.complex_field(np.pi, 0.3)
DXS = pb.DXS                          # [0.05, 0.035, 0.025, 0.018]
CHAOS = 0.30                          # 30% disorder
JITTER = jitter_for_chaos(CHAOS)      # = 0.15
SEEDS = list(range(24))               # >= 20 seeds
METHODS = ["firm-proj", "firm-ghost", "gfdm-constraint", "gfdm-penalty"]


def run():
    res = {}
    for m in METHODS:
        res[m] = {}
        for dx in DXS:
            t0 = time.perf_counter()
            errs = [pb._neumann_box(dx, JITTER, s, FIELD, m) for s in SEEDS]
            errs = np.asarray(errs, float)
            res[m][f"dx{dx}"] = dict(
                median=float(np.median(errs)),
                q25=float(np.percentile(errs, 25)),
                q75=float(np.percentile(errs, 75)),
                min=float(errs.min()), max=float(errs.max()),
                n_seeds=len(SEEDS),
                errs=[float(e) for e in errs],
            )
            print(f"{m:16s} dx={dx:6.3f}  median {np.median(errs):.3e}  "
                  f"IQR [{np.percentile(errs, 25):.3e}, {np.percentile(errs, 75):.3e}]  "
                  f"({time.perf_counter() - t0:.1f}s)")
    return res


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
    meta = dict(benchmark="all-Neumann unit box, interior mean-removed rel-L2 "
                          "(paper_benchmarks._neumann_box, identical to tab:neumann)",
                dxs=DXS, chaos=CHAOS, jitter=JITTER, seeds=SEEDS,
                statistic="median / q25 / q75 over seeds")
    res = run()
    save("table1_seed_spread", dict(meta=meta, results=res))
