"""
Recompute the benchmark numbers behind the paper figures and dump them to
figures/paper_numbers.json (the cache read by paper_figures.py). Separated from the
plotting so that restyling figures does not require re-solving the sweeps.

Run:  python3 paper_figures_compute.py [--quick]
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_benchmarks as pb
from convergence import CHAOS

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)


def main(argv):
    quick = "--quick" in argv
    seeds = list(range(6)) if quick else list(range(16))
    dxs = pb.DXS_COARSE if quick else pb.DXS
    num = {}

    ns = pb.neumann_straight(pb.DXS_NEU, seeds)
    num["neumann_straight"] = {m: dict(errs=ns[m][0], order=ns[m][1]) for m in ns}

    fr = pb.franke_operator(pb.DXS_FRANKE if hasattr(pb, "DXS_FRANKE") else dxs, seeds)
    num["franke_operator"] = {f"chaos{int(c*100)}": {
        m: dict(errs=fr[c][m][0], order=fr[c][m][1]) for m in fr[c]} for c in fr}

    r = pb.b1_dirichlet(dxs, seeds)
    num["B1_square_dirichlet"] = {f"chaos{int(c*100)}": {
        "firm": dict(errs=r[c]["firm"][0], order=r[c]["firm"][1]),
        "gfdm": dict(errs=r[c]["gfdm"][0], order=r[c]["gfdm"][1]),
        "fi": dict(errs=r[c]["fi"][0], order=r[c]["fi"][1])} for c in CHAOS}

    nb = pb.neumann_baseline(dxs, seeds, chaos=(0.30, 0.60))
    num["neumann_baseline"] = {f"chaos{int(c*100)}": {
        m: dict(errs=nb[c][m][0], order=nb[c][m][1]) for m in nb[c]} for c in nb}

    b2 = pb.b2_curved(dxs, seeds, chaos=(0.30, 0.60))
    num["B2_star"] = {f"chaos{int(c*100)}": {
        k: dict(errs=b2[c][k][0], order=b2[c][k][1]) for k in b2[c]} for c in b2}

    b3 = pb.b3_robin(dxs, seeds)
    num["B3_flower_robin"] = {f"chaos{int(c*100)}": dict(errs=b3[c][0], order=b3[c][1]) for c in CHAOS}

    num["B5_surface"] = dict(
        detection=pb.b5_surface_detection(dx=0.045, jitter=0.30, seed=7),
        convergence={m: dict(errs=v[0], order=v[1]) for m, v in pb.b5_surface_convergence(dxs, seeds).items()})

    with open(os.path.join(OUT, "paper_numbers.json"), "w") as f:
        json.dump(num, f, indent=2)
    print("saved paper_numbers.json  (dxs =", dxs, ")")


if __name__ == "__main__":
    main(sys.argv[1:])
