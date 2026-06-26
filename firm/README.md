# FIRM validation suite

Bottom-up tests for the **FIRM** (Flux-Integrated Renormalized Meshless) method in
[`../docs/spec.md`](../docs/spec.md), Sections 1–5 (the Poisson pipeline).
Every substep is checked against an **independent analytic ground truth**, so the
capstone (a complex manufactured Poisson on a jittered cloud) exercises only code
that the substeps have already validated.

[`../docs/spec.md`](../docs/spec.md) (the v3 specification) is the single source of
truth; the section numbers cited throughout the suite refer to it.

## Run

```bash
python3 run_all.py            # whole suite: 110 checks, per-check PASS/FAIL + summary
python3 run_all.py --plot     # + capstone figures (convergence, error field)
python3 tests/test_03_laplacian.py    # any substep standalone
python3 capstone_poisson.py --plot    # capstone only
pytest tests/                 # the substeps are pytest-discoverable too

# JCP paper artefacts (boundary-closure manuscript):
python3 paper_benchmarks.py [b1 nb b2 b3 b5] [--full]   # benchmark battery
python3 paper_figures.py [--quick]                       # paper figures -> figures/
python3 make_figures.py                                  # robustness/kernel figures
```

Dependencies: numpy (required), scipy (neighbour search + sparse solve, optional —
falls back to numpy), matplotlib (plots only).

## Layout

| file | role |
|------|------|
| `firm_core.py` | the operators (kernel, geometry, gradient, Laplacian, AN/GGP projectors, surface detection, unified `particle_operator`, `mirror_ghost_terms`). **Trust anchor.** |
| `geometry2d.py` | polygon helpers (incl. `nearby_walls` returning *all* segments < `h_w`, needed for the K=2 wedge) + jittered cloud builders |
| `manufactured.py` | analytic fields (value/grad/laplacian) + analytic BC data |
| `poisson.py` | unified Poisson assembly + solve; `wall_closure="projection"` or `"ghost"` (shared by substeps 8/10 and the capstone) |
| `testkit.py` | tiny PASS/FAIL reporter + log-log order helper |
| `tests/test_01..13` | the substep tests (10 = mirror-ghost wall closure; 11 = jitter/anisotropy robustness; 12 = kernel shape & support radius; 13 = GFDM baseline + constraint-row Neumann) |
| `capstone_poisson.py` | complex Poisson on the jittered tank + convergence study + projection-vs-ghost wall demo |
| `prototype_neumann.py` | standalone all-Neumann-box study behind the ghost fix (baseline vs ghost vs GFD) |
| `compare_normalization.py` | trb vs denom Laplacian normalization sweep (pointwise + Poisson) |
| `make_figures.py` | summary plots → `figures/` (convergence, jitter robustness, kernel/radius) |
| **`gfdm.py`** | **JCP-paper baseline:** 2nd-order GFDM/WLSQ Laplacian + exact-flux KKT constraint-row Neumann (Tiwari–Kuhnert) and a penalty variant |
| **`bvp.py`** | **JCP-paper driver:** FIRM curved-domain BVP assembly (Dirichlet value closure, Neumann projection/ghost, Robin) on a general polygon |
| **`convergence.py`** | chaos-level (30/60/90 %) + ≥20-seed refinement harness matching the 2017 protocol |
| **`paper_benchmarks.py`** | benchmark battery B1/NB/B2/B3/B5 (FIRM vs GFDM) feeding the manuscript |
| **`paper_figures.py`** | new paper figures + `figures/paper_numbers.json` (B1, Neumann baseline, B2, B3, B5, closure schematic) |

The manuscript and its bibliography live in `../paper/` (`firm_boundary_closure.tex`, `firm.bib`).

## Locked conventions

* kernel: poly6 `W = max(0, 1 − q²)³`, `q = r/h`, default support `h = 2.5·dx`. Operators
  are kernel-agnostic; a kernel/support study (`compare_kernels.py`) chose poly6 because the
  flat top uses neighbours efficiently — it beats the spiky `(1−q)³`/`(1−0.4·r/dx)³` and
  Wendland at **small** support and gives the cleanest ghost-wall convergence. `2.0–2.25·dx`
  is viable for quasi-uniform (incompressible) clouds.
* `N_i = (2/d) tr(B_i)` (Sec 1.2; pinned by a 3D constant-Laplacian test — in 2D the
  `2/d` factor is invisible, which is why the old scripts' bare `tr(B)` went unnoticed).
  An optional denominator form `N = 2d / Σ Wᵢⱼ|xᵢⱼ|² wᵢⱼ` (exact for `|x|²`) is ~8–15%
  more accurate (growing with jitter) — `norm="denom"`; see
  `compare_normalization.py`. Default stays `trb`.
* **project in physical space, then renormalize**: `V = B (P_tan o)`, never `P_tan (B o)`

## What the suite establishes (verified, honest)

**Works to round-off (any cloud, incl. truncated boundary rows):**
- gradient and Laplacian are **linear-exact** (~1e-14)
- AN/GGP projector algebra: idempotent, symmetric, `P n_k = 0`, K=1 / orthogonal ⇒ AN=GGP, 3 walls ⇒ P=0
- wall-Neumann **linear consistency** `LHS = b_wall`; **project-then-renormalize** is required (the other order breaks it)
- free-surface **singularity cancellation** `−(o·n_s)/δ = S` (exact)
- hydrostatic (linear) Poisson recovered to ~1e-14 with **GGP** walls

**Convergence behaviour (the important, non-obvious part):**
- the bare Laplacian is *only* linear-consistent → its pointwise truncation error is
  **O(1) under fixed-relative jitter** (2nd-order only on a regular lattice). The
  Poisson **solution** nevertheless supraconverges.
- **The boundary closure governs the order:**
  - **Dirichlet / free-surface** closure: strong (~order 1.2–1.7). The free-surface
    region is the *most* accurate part of the capstone (~1e-3, order ~1.7).
  - **Neumann wall** closure: converges (~order 1.3 asymptotically) but is the
    **accuracy-limiter** — error constant ~25x larger than Dirichlet at the same dx,
    because the flux-only boundary row leaves the normal 2nd-derivative one-sided
    (LeVeque §2.12: the O(1) boundary column does not damp the boundary truncation error).
    The capstone error map concentrates on the walls/wedge for this reason. NOTE: a
    pure-Neumann solve needs zero-mean null-space handling, not a single-point pin (pin +
    raw error makes it *look* divergent — an earlier diagnostic artifact, since corrected).
- **GGP vs AN:** GGP is exact for linear fields at the non-orthogonal wedge; AN leaks
  O(Δ) (Sec 10.3). For the complex field both give comparable *total* error because the
  wall-closure error dominates and masks the projector difference.

**Robustness sweeps** (test 11 jitter, test 12 kernel/radius):
- **Jitter 0→0.4·dx:** linear-exact at every level; cond(M) rises smoothly (1.9→5.7) but
  stays invertible; Poisson error grows gracefully (lattice most accurate). The denom-vs-trb
  edge **widens with anisotropy** (error ratio 1.00→0.78). Smoothstep never false-triggers
  interior even at 40% jitter; rational c=0.2 climbs to ~0.34.
- **Kernel/radius:** operators are kernel-agnostic (linear-exact for poly6, spiky `(1-q)³`,
  and spiky `(1-0.4q)³` at h ∈ {1.8,2.0,2.2,2.6}·dx). Clean convergence needs adequate
  support — h≈2.6·dx (≈6 nbrs) is clean; h≤2.2·dx hits the 2D rank floor (3 nbrs) under
  jitter and gets noisy. The spiky `(1-0.4q)³` kernel has the lowest pointwise error and is
  most robust at small radii.

The capstone prints a variant × dx table, a per-region (interior/wall/surface) error
breakdown, and a from-scratch Dirichlet-vs-Neumann BC-isolation diagnostic that
reproduces the order story.

### The mirror-ghost wall closure (the Neumann fix — now integrated)

The Neumann limiter is fixed by **completing the boundary stencil with mirror ghosts**:
reflect every fluid neighbour AND particle `i` itself across the near wall(s), plus the
corner double-reflection `R_B(R_A(·))` for wedges, assigning each ghost the
Neumann-consistent value `p_src − 2σg`. This is the LeVeque/ghost-node cure — keep a
two-sided stencil so the prescribed flux supplies the missing normal 2nd-derivative.

Implemented in `firm_core.mirror_ghost_terms` and selectable as
`poisson.assemble(..., wall_closure="ghost")`; regression-tested in
`tests/test_10_ghost_wall.py`. It stays **linear-exact** and on the manufactured tank
problem cuts the overall L2 error **~10–17×** and the wall-region error **~14×** vs the
projection closure, restoring fast convergence (capstone: projection order ≈0.45 →
ghost ≈3.2). It is applied to pure-Neumann-wall particles; surface/contact-line and
interior particles keep the projection path.

A soft penalty-flux GFD does *not* help (the flux must be an exact constraint or a
completed stencil — see `prototype_neumann.py`). The corner double-reflection and the
self-reflection of `i` were both essential to remove jitter/instability. Pushing to a
guaranteed high order (Neumann = Dirichlet rate) would use a 2nd-order operator with an
exact flux constraint (GMLS/KKT — Trask, Perego, Bochev, SIAM J. Sci. Comput.
39(2):A479, 2017).
