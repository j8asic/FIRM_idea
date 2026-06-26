# FIRM — Flux-Integrated Renormalized Meshless

*A boundary-integrated closure for renormalized, least-squares-corrected meshless
operators on elliptic boundary-value problems.*

FIRM is a meshless discretization framework for incompressible flow that folds the
**wall (Neumann)** and **free-surface (Robin/Dirichlet)** boundary conditions **directly
into the differential operators** — algebraically, in strong form, on a single code path —
instead of padding the domain with auxiliary particles, embedding it in a level set, or
enforcing the condition through a weak form. The boundary is detected from the point cloud
itself, with no tuning parameters.

The method rests on three principles:

1. **Trust the neighbors for what they can see; impose physics for what they cannot.**
2. **All projections are performed in physical space on the raw offset vector `oᵢ`,
   *before* renormalization by `Bᵢ`.**
3. **The boundary-condition *type* — not the physical quantity — decides where the known
   datum enters: a Neumann flux goes to the right-hand side; a Dirichlet/Robin value goes
   to the diagonal.**

This repository collects the **idea**, a bottom-up **numerical validation suite**, and the
**manuscript** describing the closure.

---

## Repository layout

| Path | Contents |
|------|----------|
| [`docs/spec.md`](docs/spec.md) | The FIRM method specification (v3) — the single source of truth: operators, boundary closure, free-surface detection, the unified pressure-Poisson equation, validation summary, known limitations, and future directions. |
| [`firm/`](firm/) | The validation suite: the operators (`firm_core.py`), Poisson assembly/solve, the GFDM baseline, the benchmark battery, and ~20 substep tests each checked against an **independent analytic ground truth**, plus a manufactured-Poisson capstone. See [`firm/README.md`](firm/README.md). |
| [`firm/figures/`](firm/figures/) | Figures and `paper_numbers.json` consumed by the manuscript (`\graphicspath` points here). |
| [`paper/`](paper/) | The manuscript `firm_boundary_closure.tex`, its bibliography `firm.bib`, and a compiled PDF. |

---

## The method, briefly

**Renormalized operators.** Each particle uses a least-squares-corrected (renormalization
tensor `Bᵢ = Mᵢ⁻¹`) gradient and Laplacian that are **linear-exact on any point cloud** —
regular or highly irregular — to round-off.

**Boundary closure, organized by condition type:**

- A **Neumann flux** is imposed by a **tangential projection performed in physical space
  before renormalization**, entering the right-hand side. A general **Gram projector**
  treats non-orthogonal corners exactly.
- A complementary **algebraic-ghost closure** completes the truncated boundary stencil by
  reflecting the neighbours *and the node itself* across the local reflection group — no
  extra unknowns. It is the LeVeque/ghost-node cure cast algebraically.
- A **Dirichlet/Robin value** enters the diagonal (self-interaction) term.
- The **free surface** is detected from the support-deficiency geometry; its enforcement
  strength cancels to the total kernel weight, removing any penalty or threshold parameter.

**What the validation establishes (honest summary):**

- Gradient and Laplacian are **linear-exact (~1e-14)** on any cloud, including truncated
  boundary rows; the projector algebra (idempotent, symmetric, `P·nₖ = 0`) holds exactly.
- The **value/free-surface closure supraconverges**; the free-surface region is the most
  accurate part of the capstone.
- The flux-only Neumann wall is the **accuracy limiter**; the **algebraic-ghost closure
  cuts the wall-limited error by ~an order of magnitude** and, with the sum normalization,
  **approaches second-order** at a *straight* Neumann boundary. Non-orthogonal corners cap
  the attainable order (~1.3).
- Used in a **Helmholtz–Hodge decomposition**, the renormalized operators are consistent
  for a *single* projection — the residual divergence equals the operators'
  non-adjointness defect and vanishes under refinement.

See [`docs/spec.md`](docs/spec.md) §9 and the manuscript for the quantified results.

---

## Running the validation suite

Requires **Python 3** and **NumPy**. SciPy (neighbour search + sparse solve) and Matplotlib
(plots) are optional — the suite falls back to NumPy without them.

```bash
cd firm
python3 run_all.py            # full suite: per-check PASS/FAIL + summary (exits nonzero on any fail)
python3 run_all.py --plot     # + capstone figures (convergence, error field)
python3 tests/test_03_laplacian.py    # any substep standalone
pytest tests/                 # the substeps are pytest-discoverable too

# Manuscript artefacts:
python3 paper_benchmarks.py [b1 nb b2 b3 b5] [--full]   # FIRM-vs-GFDM benchmark battery
python3 paper_figures.py [--quick]                       # paper figures -> figures/
```

[`firm/README.md`](firm/README.md) documents every module, the locked conventions (kernel,
normalization, projection order), and the full findings.

---

## Building the manuscript

The figures it references are committed under `firm/figures/`, so it builds out of the box:

```bash
cd paper
pdflatex firm_boundary_closure
bibtex   firm_boundary_closure
pdflatex firm_boundary_closure
pdflatex firm_boundary_closure
```

A pre-built `firm_boundary_closure.pdf` is included. To regenerate the figures, run the
`paper_figures.py` / `make_figures.py` scripts in `firm/`.

---

## Status and scope

- **Validated:** the renormalized operators, the boundary closure (Neumann projection +
  Gram corner + algebraic ghost), free-surface detection, and the elliptic/Poisson pipeline
  — verified against analytic ground truths and irregular-domain benchmarks in 2D.
- **Not yet validated:** the velocity operators near boundaries and the implicit-viscosity
  Helmholtz system (spec §6–7) are *derived but not exercised* by the current test suite,
  and are flagged as such.

FIRM is a **research idea under active development** — interfaces and results may change.

---

## Citation

If you use this work, please cite the manuscript:

> J. Bašić and C. Peng, *A boundary-integrated closure for renormalised meshless operators
> on elliptic boundary value problems.*

---

## Authors

- **Josip Bašić** — Faculty of Electrical Engineering, Mechanical Engineering and Naval
  Architecture, University of Split, Croatia
- **Chong Peng** — School of Civil Engineering, Southeast University, Nanjing, China

## License

No license has been chosen yet, so all rights are reserved by default. Please contact the
authors before reusing or redistributing this work.
