# A Unified Boundary-Integrated Meshless Lagrangian Method (FIRM) — v3

## Abstract

We present **FIRM** (Flux-Integrated Renormalized Meshless), a meshless discretization
framework for incompressible flow that integrates wall (Neumann) and free-surface
(Robin/Dirichlet) boundary conditions directly into renormalized differential operators,
eliminating solid particles and free-surface instabilities. The method rests on three
principles:

1. **Trust the neighbors for what they can see. Impose physics for what they cannot.**
2. **All projections are performed in physical space on the raw offset vector $\mathbf{o}_i$ before renormalization by $\mathbf{B}_i$.**
3. **The boundary-condition *type* (Neumann vs. Dirichlet) — not the physical quantity — determines whether the known datum enters the RHS (flux) or the self-interaction term (value).**

This v3 revision incorporates the results of a bottom-up numerical validation suite (103
analytic checks across 12 substeps + a capstone; see §9). The validation confirmed the
operator algebra and the free-surface closure to round-off, quantified the convergence
behaviour, and **identified the original flux-only Neumann wall closure as the accuracy
bottleneck**. v3 adds a **mirror-ghost wall closure** (§2.5) that completes the boundary
stencil and reduces the wall-limited error by ~10–17× while preserving linear-exactness
and the single code path for everything else.

---

## 0. Changelog from v2

| Area | v2 | v3 |
|------|----|----|
| Kernel & support | poly6-style $(1-q^2)^3$, $\kappa\in[2,3]$ | **poly6 $(1-q^2)^3$ confirmed best, support $2.5\Delta$** (study at matched support beat the spiky $(1-0.4\,r/\Delta)^3$ and Wendland at small support; §1.1, §9); operators kernel-agnostic; $2.0$–$2.25\Delta$ viable for quasi-uniform clouds |
| Normalization $N_i$ | $\tfrac{2}{d}\mathrm{tr}\mathbf{B}_i$ in §1.2, but prose dropped the $\tfrac{2}{d}$ | **Locked** to $N_i=\tfrac{2}{d}\mathrm{tr}\mathbf{B}_i$ everywhere; pinned by a 3D constant-Laplacian test |
| Wall Neumann closure | projection + flux RHS only | projection retained; **mirror-ghost closure added** (§2.5) as the accurate option |
| §5.1 "moment-matrix augmentation (rejected)" | rejected as "a damped wrong answer" | superseded — the *correct* fix is **stencil completion** (ghosts), not damping |
| Convergence claims | qualitative | **quantified** (§9): Dirichlet/free-surface ~order 1.3; flux-only Neumann ~order 1.3 but ~25× larger constant; ghost closure restores accuracy |
| Surface activation | rational $c\approx0.2$ or smoothstep | **smoothstep recommended** (parameter-free hard zero); rational $c=0.2$ over-activates at 30% jitter (§3.4, §9) |
| Velocity / viscosity (§6–7) | full derivations | unchanged and **not yet validated** (out of scope of the v3 test suite) — flagged |

---

## 0.1 Corrections (v3.1, after seed-averaged verification + the JCP write-up)

These supersede the single-seed numbers in §9 where they conflict.

1. **The ghost-wall order was over-stated.** "Projection ≈0.4 → ghost ≈3.2" was a *single-seed*
   capstone fluke. Seed-averaged (≥6 seeds) on the wedge tank the ghost wall-region order is
   **≈1.25 (full range), ≈1.8 (fine range)** — still ~10× lower error than projection, but not
   3rd-order. The all-Neumann box "order" is null-space noise (bounces −0.5…7): measure Neumann
   order on a *well-posed* problem (mixed Dirichlet/Neumann, or the wedge tank), never the
   all-Neumann box.

2. **Ghost rows want the denom normalization, not trb (implementable fix).** With `norm="trb"`
   the ghost error *plateaus* (order ≈1.4). With `norm="denom"` (§1.2, exact for |x|²) it keeps
   converging and **approaches 2nd order (≈1.8)** on a *straight* Neumann boundary. Change:
   `mirror_ghost_terms(...)` takes `norm=` and computes `N=2d/Σ Wᵢⱼ|xᵢⱼ|²wᵢⱼ` on the completed
   support; pass `norm` through `poisson.assemble`/`bvp.assemble`. Use `denom` on straight/smooth
   Neumann; keep `trb` at non-orthogonal corners (more robust there).

3. **The non-orthogonal corner is the real cap.** At a wedge the reflection group is truncated to
   {R₁, R₂, R₂∘R₁}; this caps the order (~1.3) regardless of normalization. Only straight/smooth
   Neumann boundaries reach the higher order in (2).

4. **GFDM/FPM constraint-row is NOT the cheap high-order fix (corrects §11).** The exact-flux KKT
   constraint row is curvature-limited exactly like the FIRM projection closure (~order 1, large
   constant), and FIRM ghosting beats it. Only the *staggered* GMLS/KKT (Trask et al.) is
   genuinely high-order; the plain appended constraint-row is not.

---

## 1. Fundamental Discrete Operators

### 1.1 Notation

For Lagrangian particle $i$ and each neighbor $j$ within support radius $h=\kappa\Delta$ ($\kappa\in[2,3]$, $\Delta$ the nominal spacing):

* $\mathbf{x}_{ij} = \mathbf{x}_j - \mathbf{x}_i$ — relative position
* $W_{ij} = W(|\mathbf{x}_{ij}|, h)$ — kernel weight
* $f_{ij} = f_j - f_i$ — field difference

**Kernel (chosen default).** Poly6-style, with compact support $h=\kappa\Delta$:

$$\boxed{W(r,h) = \max\!\big(0,\;1 - q^2\big)^3,\quad q=r/h,\qquad \kappa = 2.5}$$

The renormalized operators **consume the weights and renormalize by $\mathbf{B}_i$, so they are kernel-agnostic** — linear-exactness is independent of the kernel and the support radius (§9). A kernel/support study (`compare_kernels.py`, §9) compared poly6, the spiky cubic $(1-q)^3$ (which equals $(1-0.4\,r/\Delta)^3$ at $h=2.5\Delta$), and a Wendland-C2 at **matched** support: the **flat-topped poly6 uses its neighbors most evenly and wins at SMALL support** (lowest Poisson-solution error) and gives the **cleanest mirror-ghost wall convergence**; the concentrated spiky cubic effectively uses fewer neighbors and is competitive only at large support, where the "small support" goal is lost. **Support radius:** $h=2.5\Delta$ is robust (~15 interior and $\ge4$ wall neighbors at 30% jitter); for the **quasi-uniform clouds of incompressible flow** (first neighbor ring $\approx\Delta$, low effective disorder) $h\in[2.0,2.25]\Delta$ is viable and cheaper — and the mirror-ghost wall closure (§2.5) augments truncated wall support, relieving the binding constraint. Below $\sim2\Delta$ under strong jitter the support hits the 2D rank floor (3 neighbors) and the operator becomes noisy (§9, §10).

### 1.2 Geometric Quantities

$$\mathbf{o}_i = \sum_j W_{ij}\,\mathbf{x}_{ij}, \qquad S_i = \sum_j W_{ij}, \qquad \mathbf{M}_i = \sum_j W_{ij}\,\mathbf{x}_{ij}\otimes\mathbf{x}_{ij}, \qquad \mathbf{B}_i = \mathbf{M}_i^{-1}$$

$\mathbf{o}_i$ is the **support-deficiency vector**: $\approx\mathbf{0}$ for full symmetric support, and points inward (away from missing neighbors) near a boundary. $\mathbf{B}_i$ (symmetric positive-definite when $\mathbf{M}_i$ has full rank, i.e. $\geq d+1$ non-degenerate neighbors) is the renormalization tensor.

**Normalization scalar (locked):**

$$\boxed{N_i = \frac{2}{d}\,\mathrm{tr}(\mathbf{B}_i)}$$

The $2/d$ factor arises from $\nabla^2 f = \lim_{r\to0}\frac{2d}{r^2}(\bar f_r - f)$ and is **mandatory** for correct second-derivative magnitude. In 2D it happens to equal $\mathrm{tr}(\mathbf{B}_i)$ (so 2D tests cannot distinguish the two); a 3D constant-Laplacian test confirms the $2/d$ form (§9, check 3c).

**Alternative "denominator" normalization (optional).** A self-normalizing variant uses the corrected second moment directly:

$$N_i^{\mathrm{denom}} = \frac{2d}{\sum_j W_{ij}\,|\mathbf{x}_{ij}|^2\,w_{ij}}, \qquad w_{ij}=1-\mathbf{x}_{ij}\cdot\mathbf{B}_i\mathbf{o}_i$$

By construction this makes the Laplacian **exact for the isotropic quadratic $|\mathbf{x}|^2$** ($\nabla^2=2d$). It equals $\tfrac{2}{d}\mathrm{tr}\mathbf{B}_i$ whenever $\mathbf{M}_i$ is isotropic (e.g. a regular lattice) and differs only under anisotropy/jitter. *(v3 finding, §9: with the default poly6 kernel it is **~8–15% more accurate** and the edge **grows with jitter** ($N^{\mathrm{denom}}/N^{\mathrm{trb}}$ error ratio $1.00\to0.73$ over jitter $0\to0.4$); for a concentrated kernel the edge is marginal. It preserves linear-exactness and is stable with a guard against a near-zero/negative denominator at pathological boundary rows. Available as `norm="denom"`; $\tfrac{2}{d}\mathrm{tr}\mathbf{B}$ remains the default.)*

### 1.3 Gradient Operator

$$\boxed{\nabla f_i = \mathbf{B}_i \sum_j W_{ij}\,f_{ij}\,\mathbf{x}_{ij}}$$

**Exact for linear functions** on *any* particle distribution: for $f=\mathbf{a}\cdot\mathbf{x}+b$, $\;\mathbf{B}_i\sum_j W_{ij}(\mathbf{a}\cdot\mathbf{x}_{ij})\mathbf{x}_{ij}=\mathbf{B}_i\mathbf{M}_i\mathbf{a}=\mathbf{a}$. *(Validated to ~$10^{-15}$ including boundary/truncated/wedge rows.)*

### 1.4 Renormalized Laplacian

Starting from the naive form $\nabla^2 f_i = N_i\sum_j W_{ij}(f_{ij}-\mathbf{x}_{ij}\cdot\nabla f_i)$ and substituting the discrete gradient, $\sum_j W_{ij}\mathbf{x}_{ij}=\mathbf{o}_i$ exits, giving the single-sum form:

$$\boxed{\nabla^2 f_i = N_i \sum_j W_{ij}\,f_{ij}\,\underbrace{(1 - \mathbf{x}_{ij}\cdot\mathbf{B}_i\mathbf{o}_i)}_{w_{ij}}}$$

**Linear consistency.** For $f=\mathbf{a}\cdot\mathbf{x}+b$: $\nabla^2 f = N_i[\mathbf{a}\cdot\mathbf{o}_i - \mathbf{a}\cdot\mathbf{M}_i\mathbf{B}_i\mathbf{o}_i]=0$ on **any** distribution, including truncated boundary clouds. *(Validated to ~$10^{-13}$.)*

> **Important accuracy note (new in v3, §9).** This operator is *linear*-consistent but **not** second-order consistent on disordered clouds. On a regular lattice it is 2nd-order; under **fixed-relative jitter** its pointwise truncation error is $O(1)$ (does not vanish as $\Delta\to0$). The Poisson *solution* nonetheless converges (supraconvergence): the $O(1)$ truncation error is a discrete divergence of a consistent flux and is damped by the solve. This is why FIRM "works" despite the bare operator's $O(1)$ pointwise error — but it caps the achievable solution order at ~1.3 (see §9).

---

## 2. Wall Boundary Integration (Neumann)

At a solid wall the pressure satisfies $\nabla p_i\cdot\mathbf{n}_w = g$ ($g=0$ for slip walls; $g=\rho\,\mathbf{g}\cdot\mathbf{n}_w$ for hydrostatics). No particles sit on the wall; the modification activates on any fluid particle whose support is truncated by the wall (within $\sim h$).

### 2.1 Wall Normals and the Tangential Projector

A particle interacts with $K$ nearby wall segments, each with outward unit normal $\mathbf{n}_k$, distance $\delta_{w,k}$, foot point $\mathbf{x}_{f,k}$, and prescribed flux $g_k$. Use **all** segments within $h_w\approx h$ (a single nearest-wall query is a bug at corners). Raw proximity scores and blend weights (used only when blending multiple wall data into one quantity):

$$\widehat\beta_k = \max\!\Big(0,\,1-\tfrac{\delta_{w,k}}{h_w}\Big)^2, \qquad \beta_k = \widehat\beta_k\Big/\!\sum_m\widehat\beta_m$$

A **tangential projector** $\mathbf{P}_{\mathrm{tan}}$ removes all wall-normal directions:

* **Option AN (Averaged Normal).** $\mathbf{n}_{\mathrm{eff}}=\frac{\sum_k\beta_k\mathbf{n}_k}{|\sum_k\beta_k\mathbf{n}_k|}$, $\;\mathbf{P}_{\mathrm{tan}}=\mathbf{I}-\mathbf{n}_{\mathrm{eff}}\otimes\mathbf{n}_{\mathrm{eff}}$. Simple; **exact for single walls and smooth curved walls**, but introduces $O(\Delta)$ error at non-orthogonal corners (the single averaged normal cannot span the full normal subspace). *(Validated: AN leaks $\sim$O($\Delta$) at a wedge; see §9.)*
* **Option GGP (General Gram Projector).** $\mathbf{N}=[\mathbf{n}_1\,|\cdots|\,\mathbf{n}_K]$, $\mathbf{G}=\mathbf{N}^{\!\top}\mathbf{N}$, $\;\mathbf{P}_{\mathrm{tan}}=\mathbf{I}-\mathbf{N}\mathbf{G}^{-1}\mathbf{N}^{\!\top}$. Symmetric, idempotent, $\mathbf{P}_{\mathrm{tan}}\mathbf{n}_k=\mathbf{0}\,\forall k$. **Exact for arbitrary corner geometry.** Regularize near-parallel walls with $\mathbf{G}_{\mathrm{reg}}=\mathbf{G}+\epsilon\mathbf{I}$, $\epsilon\sim0.01$. *(All projector properties validated to ~$10^{-16}$; 3 non-coplanar 3D walls give $\mathbf{P}_{\mathrm{tan}}=\mathbf{0}$.)*

For $K=1$ and for orthogonal normals the two options coincide.

### 2.2 The Projection-Ordering Principle

> **Project in physical space first, then renormalize.**

$$\boxed{\mathbf{o}_{w,i}=\mathbf{P}_{\mathrm{tan}}\,\mathbf{o}_i, \qquad \mathbf{V}_i=\mathbf{B}_i\,\mathbf{o}_{w,i}}$$

The alternative $\mathbf{P}_{\mathrm{tan}}(\mathbf{B}_i\mathbf{o}_i)$ subtracts a component in the $\mathbf{n}$ direction rather than the correct $\mathbf{B}_i\mathbf{n}$ direction; the two coincide only if $\mathbf{n}$ is an eigenvector of $\mathbf{B}_i$ (never, for realistic clouds). *(Validated: the two orderings differ, and only project-then-renormalize preserves wall linear consistency.)*

### 2.3 The Projection (flux-only) Wall Laplacian

Decompose $\nabla f_i=\mathbf{P}_{\mathrm{tan}}\nabla f_i + (\mathbf{I}-\mathbf{P}_{\mathrm{tan}})\nabla f_i$; reconstruct the tangential part from neighbors and substitute the known flux for the normal part. The result, in $N_i$-normalized form:

$$\sum_j W_{ij}\,f_{ij}\,(1-\mathbf{x}_{ij}\cdot\mathbf{V}_i) = \frac{\nabla^2 f_i}{N_i} + b_{\mathrm{wall},i}$$

$$b_{\mathrm{wall},i}=\begin{cases} g_{\mathrm{eff}}\,(\mathbf{o}_i\cdot\mathbf{n}_{\mathrm{eff}}), & \text{AN } (g_{\mathrm{eff}}=\sum_k\beta_k g_k)\\[2pt] (\mathbf{N}^{\!\top}\mathbf{o}_i)^{\!\top}\mathbf{G}^{-1}\mathbf{g}, & \text{GGP}\end{cases}$$

**Linear consistency** holds for both options (the flux RHS exactly cancels the residual $N_i\,\mathbf{a}\cdot(\mathbf{I}-\mathbf{P}_{\mathrm{tan}})\mathbf{o}_i$). *(Validated to ~$10^{-17}$ for single walls and GGP wedges; AN wedge residual is bounded and $O(\Delta)$.)*

> **Accuracy limitation (the key v3 finding, §9).** This closure imposes only the **first**-derivative flux at the wall. It is linear-exact, but for nonlinear solutions the **normal second derivative** $\partial^2 p/\partial n^2$ is left to one-sided support and is poorly resolved. By the boundary-layer mechanism (LeVeque, *FD Methods*, §2.12): an $O(h)$ boundary-row truncation error sits at a node whose column of the inverse operator is $O(1)$, so it is **undamped** and pollutes the solution with a large constant. Net effect: the wall is the **accuracy bottleneck** — the solution still converges (~order 1.3) but with an error constant ~25× larger than at Dirichlet/free-surface boundaries. The projection closure is correct and elegant (single code path, no ghosts) but should be regarded as the *baseline*; §2.5 gives the accurate alternative.

### 2.4 The 3-Wall Corner

For $K=d$ non-coplanar walls, $\mathbf{P}_{\mathrm{tan}}=\mathbf{0}$, hence $\mathbf{V}_i=\mathbf{0}$, all $w_{ij}=1$, and the particle is fully determined by the wall fluxes and RHS. Physically correct — the particle is enclosed.

### 2.5 Mirror-Ghost Wall Closure (new in v3 — the Neumann accuracy fix)

The cure for §2.3's limitation is **stencil completion**: instead of projecting out the unresolved normal direction, *fill it* with ghost samples so $\partial^2 p/\partial n^2$ becomes a genuine two-sided difference, and let the prescribed flux set the ghost values (the classical LeVeque ghost-node / image-particle construction). This supersedes the "moment-matrix augmentation" idea rejected in v2 §5.1: that damped the noise without fixing the missing curvature; this *supplies* the missing curvature.

**Construction.** For a wall particle $i$ with near-wall segments $\{(\mathbf{n}_k,\mathbf{x}_{f,k},g_k)\}$, reflect across the **reflection group** generated by those segments. Let $R_k$ be reflection across the plane through $\mathbf{x}_{f,k}$ with normal $\mathbf{n}_k$. The set of compositions $\mathcal{C}$ is all non-empty subsets of the $K$ segments applied in order:

* $K=1$: $\{R_1\}$
* $K=2$ (corner/wedge): $\{R_1,\,R_2,\,R_2\!\circ\!R_1\}$ — the last is the **corner double-reflection** that fills the far quadrant.

Reflect **every fluid neighbor $j$ AND particle $i$ itself**. For a source point $\mathbf{x}_s$ (a neighbor, or $\mathbf{x}_i$) and a composition, accumulate the mirrored position and flux increment:

$$\mathbf{x}_s \xrightarrow{R_k}\; \mathbf{x}_s - 2\sigma_k\mathbf{n}_k,\quad \sigma_k=(\mathbf{x}_s-\mathbf{x}_{f,k})\cdot\mathbf{n}_k, \qquad \text{value } p_{\mathrm{ghost}} = p_s + \sum_{k\in\text{comp}} (-2\sigma_k g_k)$$

Each resulting ghost within $h$ is appended to the support with weight $W(|\mathbf{x}_{\mathrm{ghost}}-\mathbf{x}_i|,h)$ and **included in $\mathbf{M}_i$, $\mathbf{o}_i$, $N_i$** (restoring near-symmetric support). The Laplacian row then couples the ghost back to its **source particle** $p_s$ (a neighbor or $i$) with the standard weight $w_{ij}=1-\mathbf{x}_{\mathrm{ghost}}\cdot\mathbf{B}_i\mathbf{o}_i$, and the constant increment $-2\sigma g$ moves to the RHS.

**Two essential details** (each verified necessary):
1. **Reflect $i$ itself.** Its mirror $\mathbf{x}_{i'}$ sits at distance $2\delta_w$ — the *closest, highest-weight* ghost. Omitting it leaves a hole that dents $M_{nn}$, re-tilts $\mathbf{o}_i$, and drops a flux contribution. In the assembly the $i$-ghost couples $p_i$ to itself ($+c$ and $-c$ on the diagonal cancel) while its flux increment survives on the RHS.
2. **Corner double-reflection** $R_2\!\circ\!R_1$. Without it a corner particle gets only $\tfrac34$-sphere support (each face reflected independently); the composition fills the final quadrant.

**Properties.** Linear-exact (a ghost of a linear field equals the true field at the mirrored point, so $\nabla^2(\text{linear})=0$ is preserved — validated to ~$10^{-14}$). It supplies the normal curvature, so the wall ceases to be the bottleneck. Applied to **pure-Neumann-wall particles**; surface/contact-line and interior particles keep the projection path, so the single-code-path elegance is retained everywhere else.

**Scope.** The reflection plane per segment $(\mathbf{x}_{f,k},\mathbf{n}_k)$ is exact for straight wall segments; genuinely curved walls need a local per-segment plane (straightforward extension). A penalty-enforced flux (soft GFD constraint) does **not** work — the flux must be supplied by an *exact* completed stencil (or an exact constraint, §11).

---

## 3. Free Surface Integration (Robin/Dirichlet)

At the free surface $p=p_{\mathrm{target}}$ ($0$ for a clean surface, $\gamma\kappa$ with surface tension). There are no surface particles; every fluid particle with air-side truncation receives the condition, smoothly activated.

### 3.1 Dirichlet → Robin

Taylor-expanding to the zero-pressure surface at distance $\delta$ along outward normal $\mathbf{n}_s$:

$$\nabla p_i\cdot\mathbf{n}_s = \frac{p_{\mathrm{target}}-p_i}{\delta}$$

a Robin condition whose "flux" depends on $p_i$ — this is what produces a diagonal (self-interaction) modification rather than a pure RHS term.

### 3.2 Natural Estimates and Singularity Cancellation

The surface direction is the support deficiency **that the walls cannot explain** — computed from the *wall-projected* offset (so a corner does not contaminate the surface normal):

$$\mathbf{r}_i = \mathbf{B}_i\,\mathbf{o}_{w,i}, \qquad \mathbf{n}_s = -\mathbf{r}_i/|\mathbf{r}_i|, \qquad \delta = \frac{|\mathbf{o}_{w,i}\cdot\mathbf{n}_s|}{S_i}$$

Substituting these into the Robin enforcement strength $-(\mathbf{o}\cdot\mathbf{n}_s)/\delta$, the $|\mathbf{r}_i|$ and $|\mathbf{o}\cdot\mathbf{n}_s|$ factors cancel:

$$\boxed{-\frac{\mathbf{o}_{w,i}\cdot\mathbf{n}_s}{\delta} = S_i}$$

So the enforcement strength is simply $S_i$ — never singular, self-scaling, **tuning-free**. *(Validated to ~$10^{-9}$, with the sign $\mathbf{o}\cdot\mathbf{n}_s<0$ confirmed.)*

### 3.3 Threshold-Free Detection

$$\lambda_i = |\mathbf{r}_i|\,\Delta \quad(\approx0 \text{ interior},\; O(1) \text{ surface}), \qquad \sigma_i=\sigma(\lambda_i)\in[0,1]$$

Activation choices:
* **Smoothstep (recommended):** $\sigma_i=S\!\big(3(\lambda_i-\tfrac23)\big)$ with $S$ the cubic smoothstep — **a hard zero for $\lambda_i\le\tfrac23$** (interior particles are *exactly* untouched) and unity for $\lambda_i\ge1$.
* **Rational:** $\sigma_i=\lambda_i^2/(\lambda_i^2+c^2)$, $c\approx$ disorder level. *(v3 finding: $c=0.2$ over-activates interior particles at 30% jitter — median interior $\sigma\approx0.24$. An adaptive $c=3\,\mathrm{median}(\lambda_{\mathrm{interior}})$ suppresses the bulk but, unlike smoothstep, never gives a hard zero. Prefer smoothstep.)*

The surface-normal removal is regularized to avoid normalizing $\mathbf{r}_i$:

$$\mathbf{o}_{*,i} = \mathbf{o}_{w,i} - \sigma_i\,\frac{(\mathbf{o}_{w,i}\cdot\mathbf{r}_i)\,\mathbf{r}_i}{|\mathbf{r}_i|^2+\eta^2}, \qquad \eta=c/\Delta$$

An optional ray-cast suppression (cast along $-\hat{\mathbf{r}}_i$; attenuate $\sigma_i$ if a wall is hit within $\sqrt{d}\,\Delta\ldots2\sqrt{d}\,\Delta$) prevents false activation at corners. *(Ray-cast helper validated; off by default when the surface is clear of walls.)*

### 3.4 Unified Correction Vector

$$\boxed{\mathbf{V}_i=\mathbf{B}_i\,\mathbf{o}_{*,i}, \qquad w_{ij}=1-\mathbf{x}_{ij}\cdot\mathbf{V}_i}$$

with $\mathbf{o}_{w,i}=\mathbf{P}_{\mathrm{tan}}\mathbf{o}_i$ and $\mathbf{r}_i=\mathbf{B}_i\mathbf{o}_{w,i}$. At a contact line the surface residual lies in the wall-tangential subspace, so the two projections do not interfere. *(Validated: interior/wall-only/surface-only/contact-line reductions all hold.)*

---

## 4. The Unified Pressure Poisson Equation

$$\boxed{\sum_j W_{ij}\,w_{ij}\,p_{ij} - \sigma_i S_i\,p_i = b_i}$$

$$b_i = \frac{\nabla^2 p_i^{\text{(src)}}}{N_i} + b_{\mathrm{wall},i} - \sigma_i S_i\,p_{\mathrm{target}}, \qquad \nabla^2 p^{\text{(src)}}=\frac{\rho}{\Delta t}\nabla\cdot\mathbf{u}^*$$

Assembled (difference form): off-diagonal $A_{ij}\mathrel{+}=W_{ij}w_{ij}$; diagonal $A_{ii}\mathrel{+}=-\sum_j W_{ij}w_{ij}-\sigma_i S_i$. The source enters as $f_i/N_i$ (the $N_i$ normalization is essential and must use the $2/d$ form); the wall flux $b_{\mathrm{wall}}$ is **not** divided by $N_i$ (already normalized); the surface term contributes $-\sigma_i S_i$ to the diagonal and $-\sigma_i S_i p_{\mathrm{target}}$ to the RHS.

With the **mirror-ghost closure (§2.5)** a pure-Neumann-wall row is assembled instead from the ghost-completed stencil: the source is $f_i/N_i$ (with the ghost-augmented $N_i$), ghosts couple to their source particles, and flux increments go to the RHS — no separate $b_{\mathrm{wall}}$ term.

**Solvability.** Any free-surface particle ($\sigma_i>0$) pins the level and removes the constant null space. **A pure-Neumann domain is singular up to a constant — handle it with a zero-mean constraint (Lagrange / mean removal), not a single-point pin.** *(v3 finding: a single-point pin + raw error makes a pure-Neumann solve look divergent; it is not — it converges once the null space is handled correctly.)*

---

## 5. Consistent Gradient Reconstruction

Used only in the velocity correction $\mathbf{u}_i^{n+1}=\mathbf{u}_i^*-(\Delta t/\rho)\nabla p_i$.

* **Interior / Free surface:** raw $\nabla p_i=\mathbf{B}_i\sum_j W_{ij}p_{ij}\mathbf{x}_{ij}$ — no correction. (The surface field already encodes the BC through the solve; the surface must be free to accelerate normally.)
* **Wall / Contact line:** replace the unreliable normal component with the known flux:
  * AN: $\nabla p_i \leftarrow \nabla p_i-(\nabla p_i\cdot\mathbf{n}_{\mathrm{eff}})\mathbf{n}_{\mathrm{eff}}+g_{\mathrm{eff}}\mathbf{n}_{\mathrm{eff}}$
  * GGP: $\nabla p_i \leftarrow \mathbf{P}_{\mathrm{tan}}\nabla p_i+\mathbf{N}\mathbf{G}^{-1}\mathbf{g}$

*(Validated: the correction imposes $\nabla p\cdot\mathbf{n}=g$ exactly and leaves the tangential component unchanged; AN$=$GGP for a single wall.)*

---

## 6. Velocity Operators Near Boundaries *(unchanged from v2; not yet validated)*

The BC *type* determines the treatment (not the quantity):

| Quantity | Wall | Free surface |
|---|---|---|
| Pressure | Neumann $\partial p/\partial n=g$ | Dirichlet/Robin $p=p_{\mathrm{target}}$ |
| Velocity (no-slip) | Dirichlet $\mathbf{u}=\mathbf{u}_{\mathrm{wall}}$ → Robin diagonal + value RHS | Neumann (stress-free) → projection only |

The no-slip wall is structurally the free-surface pressure case (Robin: diagonal $-\sum_k\frac{\mathbf{o}_i\cdot\mathbf{n}_k}{\delta_{w,k}}\beta_k\mathbf{u}_i$, RHS the wall-velocity target). The stress-free surface is the slip-wall pressure case (projection only). Near-wall divergence is corrected by replacing the poorly-resolved normal–normal velocity gradient with the impermeability Robin estimate $-\mathbf{u}_i^*\cdot\mathbf{n}_w/\delta_w$. *These should inherit the same Neumann/Dirichlet accuracy characteristics as §2–4; in particular the velocity-divergence wall correction is a flux-only closure and is a candidate for the same ghost completion.*

---

## 7. Implicit Viscosity — Helmholtz System *(unchanged from v2; not yet validated)*

$\mathbf{u}_i^*-\Delta t\,\nu\nabla^2\mathbf{u}_i^*=\mathbf{u}_i^n+\Delta t\,\mathbf{g}$, a Helmholtz system. The identity term guarantees diagonal dominance; the discrete operator is **nonsymmetric** (row-local $w_{ij}$, $N_i$), so use **BiCGStab** or restarted **GMRES** with Jacobi preconditioning. Both viscosity limits are well-conditioned.

---

## 8. Full Projection Timestep

0. Neighbor search ($h=\kappa\Delta$, $\kappa\in[2,3]$).
1. Geometry: $\mathbf{o}_i,S_i,\mathbf{M}_i,\mathbf{B}_i,N_i$.
2. Wall detection: near segments, $\beta_k$, $\mathbf{P}_{\mathrm{tan}}$, $\mathbf{o}_{w,i}$.
3. Surface detection: $\mathbf{r}_i,\lambda_i,\sigma_i$ (+ ray-cast); build $\mathbf{o}_{*,i}$, $\mathbf{V}_i$, $w_{ij}$.
4. Predict velocity (implicit viscosity Helmholtz solve).
5. Divergence of $\mathbf{u}^*$ (+ wall correction).
6. Pressure Poisson solve (§4; choose projection or ghost wall closure).
7. Pressure gradient (+ wall correction, §5) and velocity correction.
8. Move particles; optional tangential shifting near boundaries.

---

## 9. Numerical Validation Summary

A bottom-up suite (numpy/scipy; `firm/` directory; **103 analytic checks**, runnable via `run_all.py`, `pytest tests/`) validates each substep against an independent ground truth, culminating in a complex manufactured Poisson on a 30%-jittered polygonal tank with a non-orthogonal wedge and a free surface.

**Round-off-exact (any cloud, incl. truncated/wedge rows):**
* Gradient and Laplacian linear-exactness: ~$10^{-15}$, ~$10^{-13}$.
* $\mathbf{B}_i\mathbf{M}_i=\mathbf{I}$ and SPD on every row.
* AN/GGP projector algebra (idempotent, symmetric, $\mathbf{P}\mathbf{n}_k=0$, $K{=}1$/orthogonal $\Rightarrow$ AN$=$GGP, 3 walls $\Rightarrow\mathbf{P}=0$): ~$10^{-16}$.
* Wall-Neumann linear consistency $\sum W w_{ij}f_{ij}=b_{\mathrm{wall}}$: ~$10^{-17}$; project-then-renormalize required.
* Surface singularity cancellation $-(\mathbf{o}\cdot\mathbf{n}_s)/\delta=S_i$: exact.
* Hydrostatic (linear) Poisson recovered to ~$10^{-14}$ with **GGP** walls.

**Convergence (the non-obvious part):**
* The bare Laplacian is 2nd-order on a regular lattice but **$O(1)$ pointwise under fixed-relative jitter**; the Poisson **solution supraconverges**.
* **Boundary closure governs the order.** Dirichlet/free-surface: ~order 1.2–1.7 (the free-surface region is the *most* accurate part of the capstone). **Flux-only Neumann walls converge (~order 1.3) but with a ~25× larger error constant** — the accuracy bottleneck (LeVeque undamped-boundary-column mechanism). *(An earlier "Neumann diverges" claim was a single-pin + raw-error artifact on the pure-Neumann null space — corrected.)*
* **GGP vs AN:** GGP is exact for linear fields at the wedge; AN leaks $O(\Delta)$. For nonlinear fields both give comparable *total* error because the wall-closure error dominates.
* **N_i convention** pinned by a 3D constant-Laplacian test (rel. error 3.8% with $2/d$; the bare $\mathrm{tr}\mathbf{B}$ would be 1.5× off in 3D). The optional **denominator normalization** $N^{\mathrm{denom}}=2d/\sum_j W_{ij}|\mathbf{x}_{ij}|^2 w_{ij}$ (exact for $|\mathbf{x}|^2$) is ~8–15% more accurate at practical resolutions (multi-seed Dirichlet box: $N^{\mathrm{denom}}/N^{\mathrm{trb}}$ error ratio $\approx 0.9$) but slightly lower order; tank: projection-walls error $-9\%$, ghost-walls $\approx$ tied; linear-exact and stable. Modest, free — `norm="denom"`.

**Mirror-ghost wall closure (§2.5):** on the manufactured tank it cuts the wall-region error by roughly **an order of magnitude** vs the projection closure, is consistently lower at every resolution, and stays linear-exact. *(Order: the "projection $\approx0.4\to$ ghost $\approx3.2$" here was single-seed; seed-averaged it is projection $\approx1.0\to$ ghost $\approx1.3$–$1.8$, and with `norm="denom"` the ghost approaches 2nd order on a straight wall — see §0.1.)* The corner double-reflection and the self-reflection of $i$ were both necessary for stability.

**Robustness sweeps (jitter, kernel, support radius):**
* **Jitter / anisotropy** $\in[0,0.4]\Delta$: linear-exactness holds to round-off at *every* level (structural). $\mathrm{cond}(\mathbf{M})$ rises smoothly ($1.9\to6.2$) but $\mathbf{M}$ stays invertible; the Poisson error grows gracefully and monotonically (regular lattice most accurate). **Smoothstep activation never false-triggers** in the interior even at 40% jitter (median $\lambda\approx0.16\ll\tfrac23$), whereas rational $c{=}0.2$ climbs to median $\approx0.39$. The **denom-vs-trb edge widens with anisotropy** ($N^{\mathrm{denom}}/N^{\mathrm{trb}}$ error ratio $1.00\to0.97\to0.88\to0.81\to0.73$), confirming the $|\mathbf{x}|^2$-exactness rationale.
* **Kernel shape & support radius** (`compare_kernels.py`, at matched support): operators are **kernel-agnostic** (linear-exact for poly6 $(1-q^2)^3$, spiky $(1-q)^3$, Wendland-C2 at every radius). **At small support poly6 wins decisively** — e.g. $h=2.0\Delta$ (8 interior neighbors): poly6 $2.3\times10^{-3}$ vs spiky $6.6\times10^{-3}$ vs Wendland $1.3\times10^{-2}$; the concentrated spiky kernel down-weights far neighbors so it is effectively *under-supported* at small $h$ and is competitive only at $h\gtrsim2.5\Delta$ (where the small-support goal is lost). poly6 also gives the cleanest **mirror-ghost convergence** (poly6 clean, spiky noisy; absolute orders single-seed here, corrected in §0.1). Clean convergence needs adequate support: $h=2.5\Delta$ ($\ge4$ wall / $\sim15$ interior neighbors) is robust; $h\in\{1.8,2.0,2.2\}\Delta$ hit the 2D rank floor (3 neighbors) under strong jitter and become noisy. **For incompressible flow the cloud stays quasi-uniform (low effective disorder), so $h\in[2.0,2.25]\Delta$ is viable and cheaper**, with the ghost closure relieving the wall-truncation constraint.

**Not yet validated:** velocity operators (§6), implicit viscosity (§7), and a full timestep (§8).

### 9.1 Figures

Generated by `firm/make_figures.py` (reusing the test/capstone helpers) into `firm/figures/`:

(figures use the default poly6 kernel at $h=2.5\Delta$, except `fig_kernel_radius` which sweeps kernels/radii.)

* **`fig_convergence.png`** — *left:* the four capstone variants (AN/GGP × exact/natural surface) converge ~order 0.5–0.65 and are comparable for the complex field (the wall-closure error masks the projector difference). *Right:* the headline — flux-only **projection** walls plateau (~order 0.4, error ~3.5×10⁻²) while the **mirror-ghost** closure drops by ~10× (order single-seed; seed-averaged ≈1.3–1.8, §0.1).
* **`fig_jitter.png`** (4 panels, jitter 0→0.4Δ at fixed dx) — (a) Poisson error rises gracefully with disorder, `denom` below `trb`; (b) the `denom/trb` error ratio falls monotonically $1.00\to0.73$ (the `|x|²`-exactness edge widens with anisotropy); (c) $\mathrm{cond}(\mathbf{M})$ rises smoothly $1.9\to6.2$ ($\mathbf{M}$ stays invertible); (d) interior false-activation — rational $c{=}0.2$ climbs while **smoothstep stays pinned at 0** through 40% jitter (poly6's low $\lambda$ floor).
* **`fig_kernel_radius.png`** (3 panels, kernels × radii at 30% jitter) — (a) pointwise Laplacian error vs radius; (b) **Poisson solution error: poly6 is lowest/most robust at small radii**, the spiky $(1-q)^3$ is worst at small support and only catches up near $2.5$–$2.6\Delta$; (c) the rank-floor: minimum (box) neighbor count is stuck at 3 for $h\le2.2\Delta$, jumping to ~6 at $h=2.6\Delta$.

Plus the capstone's own `capstone_error_field.png` (error concentrates on the walls/wedge, vanishing toward the free surface) and `capstone_convergence.png`.

---

## 10. Known Limitations

1. **Thin fluid structures** (thickness $\lesssim 2\Delta$): two-sided truncation partially cancels, $\lambda_i\to0$, surface detection fails. The fluid domain must be thick relative to $h$.
2. **Severely isolated / under-supported particles** ($<d+1$ neighbors): $\mathbf{M}_i$ loses rank. Detect and fall back (e.g. pin $p=0$); never silently ridge-regularize. Related: a **small support radius under jitter** drives the *minimum* neighbor count to the rank floor (3 in 2D at $h\le2.2\Delta$, 30% jitter) — convergence stays bounded but becomes noisy. Use $h\gtrsim2.6\Delta$ (or lower jitter / particle shifting) for clean convergence (§9).
3. **AN at non-orthogonal corners:** $O(\Delta)$ tangential-projector leak (use GGP).
4. **Surface tension:** supported via $p_{\mathrm{target}}=\gamma\kappa$, but accurate meshless curvature estimation is unaddressed.
5. **Activation sensitivity:** rational $c$ is disorder-dependent; smoothstep is parameter-free and preferred.
6. **Curved-wall ghosts (§2.5):** the per-segment reflection plane is exact only for straight segments; curved walls need a local-plane variant.
7. **Pure-Neumann null space:** requires zero-mean handling, not a single-point pin.

---

## 11. Future Directions

1. **Guaranteed high-order Neumann walls.** The ghost closure removes the bottleneck empirically (~order 1.3–1.8 seed-averaged, approaching 2nd order on a straight wall with `norm="denom"`; §0.1) but is not a *provably* high-order scheme. For a guaranteed Neumann$=$Dirichlet rate, adopt the **GMLS/KKT staggered construction** (Trask, Perego, Bochev, *SIAM J. Sci. Comput.* 39(2):A479, 2017). *(The plain GFDM/FPM appended constraint-row of Tiwari–Kuhnert 2001 is NOT high-order — it is curvature-limited like the projection closure and FIRM ghosting beats it, §0.1. A soft penalty flux does not suffice either — confirmed.)*
2. **Curved/complex-geometry ghosts.** Generalize §2.5 to per-segment local planes and to $K\ge3$ corners (full reflection group); validate on wedge/triangular-obstacle and hull geometries.
3. **Contact-line ghosting.** Extend the ghost completion to wall+surface contact-line particles (currently on the projection path).
4. **Validate the velocity/viscosity pipeline.** Build manufactured tests for the no-slip Robin velocity Laplacian, the stress-free surface, the near-wall divergence correction, and the Helmholtz solve (nonsymmetry, BiCGStab/GMRES convergence) — and check whether the velocity-divergence wall correction needs the same ghost treatment.
5. **End-to-end benchmarks.** Lid-driven cavity (wall treatment + implicit viscosity), Taylor–Green (interior order), dam break / sloshing (free-surface dynamics + contact lines), hydrostatic-at-rest stability over many steps.
6. **Second-order interior operator.** Even the interior is only linear-consistent (supraconvergent ~1.3). A locally 2nd-order-consistent operator (full-Hessian MLS/GFD) would lift the whole-domain order and likely make Neumann walls 2nd-order without ghosts.
7. **Surface curvature & tension.** A robust meshless $\kappa$ estimator (e.g. from $\mathbf{r}_i$/the surface-normal field) to enable $p_{\mathrm{target}}=\gamma\kappa$.
8. **Solver/preconditioning at scale.** Block-Jacobi/ILU(0) for the (nonsymmetric) Poisson and Helmholtz systems; matrix-free Krylov; performance on large 3D clouds.

---

## 12. Recommended Implementation Progression

* **Phase 1 — AN + projection walls.** Hydrostatic box (linear consistency), dam break (surface detection + Robin), lid-driven cavity (viscosity + walls). Sufficient to validate the core.
* **Phase 2 — GGP + mirror-ghost walls.** Rerun Phase 1; add non-orthogonal geometry (wedge, triangular obstacle). Switch on `wall_closure="ghost"` where wall accuracy matters. The AN→GGP and projection→ghost swaps are isolated, local code changes — everything downstream consumes $\mathbf{o}_{w,i}$/the assembled row regardless.
* **Phase 3 — Complex geometry & dynamics.** Full target application: arbitrary wall angles, dynamic free surface, optionally the GMLS/KKT or GFDM constraint-row wall for guaranteed high order.
