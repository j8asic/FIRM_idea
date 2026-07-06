"""
Revision tooling (paper_rev_*): MARRONE-STYLE lambda_min free-surface detector vs
the paper's smoothstep activation sigma_i, on the Section-5.5 jittered wedge-tank
clouds (paper_benchmarks B5 geometry).

Marrone et al. (CMAME 199, 2010) flag free-surface particles by the minimum
eigenvalue of the renormalisation matrix; the analogue in the FIRM machinery is
the (already-computed) weighted moment matrix M_i = sum_j W_ij x_ij x_ij^T,
normalised so a full interior support gives eigenvalues ~1:

    m0 = (1/dx^2) * int_{|x|<h} W(|x|) x^2 dA = pi h^4 / (40 dx^2)   (poly6)
    lambda_n = eigmin(M_i) / m0 ,   flag surface when lambda_n < 0.75.

The paper's detector: fc.particle_operator(walls=..., surface='natural',
activation='smoothstep') -> flag when sigma_i > 0 (wall-projected offset, so
walls do not trigger it).

Ground truth is geometric: d_surf = FILL_H - y (distance to the known free
surface), d_wall = distance to the tank polygon.
  * outermost surface ring : d_surf < dx          (must be flagged; miss = FN)
  * strict interior        : d_surf > h, d_wall > h (flag = FP)
  * wall band              : d_wall < h, d_surf > h (near a wall only; Marrone
    has no wall information, so flags here are reported separately)

Writes figures/paper_extra_numbers.json  key 'marrone_detector'.
Run:  python3 paper_rev_marrone.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import firm_core as fc
import geometry2d as g2
import paper_benchmarks as pb

OUT = os.path.join(HERE, "figures", "paper_extra_numbers.json")
TANK, FILL_H = pb.TANK, pb.FILL_H
DXS = [0.06, 0.045, 0.033]
DISORDER = {30: 0.15, 60: 0.30, 90: 0.45}   # chaos % -> jitter (= c/2)
SEEDS = [7, 11, 19]
LAMBDA_THRESHOLD = 0.75


def one_cloud(dx, jitter, seed):
    h = 2.5 * dx
    m0 = np.pi * h ** 4 / (40.0 * dx * dx)    # full-support moment scale (poly6)
    pos = g2.tank_cloud(dx, FILL_H, TANK, jitter=jitter, seed=seed)
    n = len(pos)
    nl = fc.neighbor_lists(pos, h)

    lam_n = np.full(n, np.nan)
    flag_marrone = np.zeros(n, bool)
    flag_sigma = np.zeros(n, bool)
    for i in range(n):
        nb = nl[i]
        if len(nb) < 3:
            continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        gm = fc.geom_quantities(xij, w)
        lam_n[i] = float(np.linalg.eigvalsh(gm.M).min()) / m0
        flag_marrone[i] = lam_n[i] < LAMBDA_THRESHOLD
        # the paper's detector (identical to paper_benchmarks.b5_surface_detection)
        walls = None
        near = g2.nearby_walls(pos[i], TANK, h)
        if near:
            walls = dict(normals=np.array([nn for _, nn, _ in near]),
                         deltas=np.array([d for d, _, _ in near]),
                         g=np.zeros(len(near)), h_w=h)
        loc = fc.particle_operator(pos[i], xij, w, walls=walls,
                                   surface=dict(mode="natural"), dx=dx,
                                   activation="smoothstep")
        flag_sigma[i] = loc.sigma is not None and loc.sigma > 0.0

    d_surf = FILL_H - pos[:, 1]
    d_wall = np.array([g2.nearest_wall(p, TANK)[0] for p in pos])
    ring = d_surf < dx
    interior = (d_surf > h) & (d_wall > h)
    wall_band = (d_wall < h) & (d_surf > h)
    lam_int_med = float(np.nanmedian(lam_n[interior])) if interior.any() else float("nan")

    def counts(flag):
        return dict(
            fp_interior=int(np.sum(flag & interior)), n_interior=int(interior.sum()),
            fp_wall_band=int(np.sum(flag & wall_band)), n_wall_band=int(wall_band.sum()),
            fn_ring=int(np.sum(~flag & ring)), n_ring=int(ring.sum()),
        )

    return dict(N=n, lam_interior_median=lam_int_med,
                marrone=counts(flag_marrone), sigma=counts(flag_sigma))


def _agg(cs, det):
    """Sum counts over seeds and derive rates."""
    keys = ("fp_interior", "n_interior", "fp_wall_band", "n_wall_band", "fn_ring", "n_ring")
    tot = {k: int(sum(c[det][k] for c in cs)) for k in keys}
    tot["fp_interior_rate"] = tot["fp_interior"] / max(tot["n_interior"], 1)
    tot["fp_wall_band_rate"] = tot["fp_wall_band"] / max(tot["n_wall_band"], 1)
    tot["fn_ring_rate"] = tot["fn_ring"] / max(tot["n_ring"], 1)
    return tot


def run():
    res = {}
    for cpct, jit in DISORDER.items():
        for dx in DXS:
            cs = [one_cloud(dx, jit, s) for s in SEEDS]
            row = dict(
                N=int(np.median([c["N"] for c in cs])), n_seeds=len(SEEDS),
                lam_interior_median=float(np.median([c["lam_interior_median"] for c in cs])),
                marrone=_agg(cs, "marrone"), sigma=_agg(cs, "sigma"),
            )
            res[f"disorder{cpct}_dx{dx}"] = row
            m, s = row["marrone"], row["sigma"]
            print(f"disorder {cpct:2d}%  dx={dx:5.3f}  N~{row['N']:5d}  "
                  f"lam_int~{row['lam_interior_median']:.2f} | "
                  f"Marrone FP_int {m['fp_interior']}/{m['n_interior']} "
                  f"FP_wall {m['fp_wall_band']}/{m['n_wall_band']} "
                  f"FN_ring {m['fn_ring']}/{m['n_ring']} | "
                  f"sigma FP_int {s['fp_interior']}/{s['n_interior']} "
                  f"FP_wall {s['fp_wall_band']}/{s['n_wall_band']} "
                  f"FN_ring {s['fn_ring']}/{s['n_ring']}")
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
    meta = dict(
        benchmark="B5 jittered wedge tank (paper Sec 5.5 geometry)",
        dxs=DXS, disorder_to_jitter=DISORDER, seeds=SEEDS,
        marrone=("eigmin of moment matrix M_i normalised by full-support value "
                 "m0 = pi h^4/(40 dx^2); flag if < 0.75 (Marrone et al. 2010 threshold)"),
        sigma="fc.particle_operator smoothstep activation, flag if sigma_i > 0",
        ground_truth=("ring: d_surf < dx (must flag); interior: d_surf > h & d_wall > h "
                      "(must not flag); wall band: d_wall < h & d_surf > h, reported "
                      "separately because Marrone has no wall information"),
        counts="summed over seeds",
    )
    res = run()
    save("marrone_detector", dict(meta=meta, results=res))
