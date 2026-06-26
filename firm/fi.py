"""
Full-Inverse (FI) 2nd-order meshless Laplacian baseline -- Asai et al. (2023),
"A class of second-derivatives in SPH with 2nd-order accuracy", CMAME 415:116203.

This is a *comparison baseline*, not part of the FIRM trust anchor
(``firm_core.py``). Asai's FI operator is the natural same-family 2nd-order
point of comparison for FIRM: it descends directly from our LDD method
(Basic et al. 2018/2022, Asai's refs [38,39]) and extends it to genuine
second-order accuracy by *including the cross-derivative terms LDD/FIRM omit*.
Asai's Table 1 records that the cross-free reduction (their "Block Diagonal")
is equivalent to LDD and to Schwaiger -- i.e. to FIRM's interior operator -- so
FI is the honest "what does adding the cross term buy you?" baseline.

Implementation (FIRM idiom, per the project plan): rather than the literal
``(r.grad_tilde_W)/|r|^4`` SPH weighting of the paper, the operator is written
in the suite's existing scalar-kernel-weight + renormalise-by-B style, so it
reuses ``firm_core`` machinery and drops into the same harnesses with no new
kernel-derivative code. It is the algebraically-equivalent construction: a
weighted least-squares fit of the second-derivative coefficients given the
linear-exact corrected gradient.

Per particle, in offset coordinates X = x/s (s = rms neighbour distance, for
conditioning), the Taylor expansion gives, for each neighbour k,

    f_ij,k - X_k . <grad f>(1)  ~=  1/2 * shat_k . u ,
        u = [f_XX, f_YY, 2 f_XY] ,

where <grad f>(1) = G @ fij is the renormalised (linear-exact) corrected
gradient. Crucially the basis is the *corrected* second-derivative basis

    shat_k = s_vec_k - P @ X_k ,   s_vec_k = [X1^2, X2^2, X1 X2] ,
    P = sum_m s_vec_m (x) c_m   (3 x d) ,   c_m = G[:, m] ,

not the raw s_vec: the corrected gradient applied to a quadratic returns
grad + 1/2 sum_m c_m (X_m^T H X_m), so the Hessian leaks into <grad f>(1). The
P @ X_k term (Asai's "R_(2) contribution", Eq. 22) subtracts that leakage, which
is exactly what makes the operator second-order-exact on a *disordered* cloud.
Using the raw s_vec instead gives only the linear-consistent class (see below).

The coefficients u are recovered by the weighted normal equations

    (sum_k w_k shat_k shat_k^T) u  =  2 sum_k w_k shat_k {f_ij,k - X_k.<grad f>(1)}

and the Laplacian is  f_XX + f_YY = (u[0] + u[1]) / s^2  (the cross term enters
the fit -- which is what makes the operator exact for any quadratic, including
cross terms -- but not the trace).

Dropping the cross column gives the cross-free "Block Diagonal" reduction
(``fi_row_bd``). Asai's Table 1 places BD (and the simpler NI/SUM variants) in
the same *linear-consistent* class as Schwaiger's operator and our LDD family:
exact for quadratics on a regular/symmetric cloud, but -- like FIRM's interior
operator (``firm_core.laplacian_interior``) -- losing pointwise second order on
a disordered cloud. FI's cross coupling is precisely what restores it.

Conventions match ``gfdm.py`` / ``poisson.py``: the assembled difference form is
``<lap p>_i = sum_j d_j (p_j - p_i) + bc_value`` set equal to the source. FI is an
interior 2nd-order operator only -- it carries no boundary closure, so
``bc_value`` is always 0 (mirroring ``gfdm_row`` with no constraints).
"""
import numpy as np

import firm_core as fc
import geometry2d as g2

try:
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve
    _HAVE_SPARSE = True
except Exception:  # pragma: no cover
    _HAVE_SPARSE = False


# --------------------------------------------------------------- helpers
def _scale(xij):
    """rms neighbour distance used to condition the moment systems."""
    return np.sqrt(max(float(np.mean((xij * xij).sum(1))), 1e-30))


def _grad_rows(X, w, ridge):
    """Corrected (linear-exact) gradient as stencil ROWS in scaled coords X.

    Returns G of shape (d, K) with  <grad f>(1)_X = G @ fij  (gradient w.r.t. X).
    This is exactly ``firm_core.grad_op``'s renormalised gradient B @ sum w f x,
    written row-wise so it can be reused inside the second-derivative fit.
    """
    d = X.shape[1]
    M1 = np.einsum("k,ki,kj->ij", w, X, X)
    if ridge:
        M1 = M1 + ridge * (float(np.trace(M1)) / d + 1e-30) * np.eye(d)
    B1 = np.linalg.inv(M1)
    return B1 @ (w[:, None] * X).T           # (d, K)


def _svec(X):
    """Second-derivative basis [X1^2, X2^2, X1 X2] (2D) on scaled offsets X (K,2)."""
    x, y = X[:, 0], X[:, 1]
    return np.stack([x * x, y * y, x * y], axis=1)   # (K, 3)


def _corrected_basis(sv, X, G):
    """Corrected second-derivative basis shat = s_vec - X @ P^T, where the
    (n_basis x d) matrix P = sv^T @ G^T is Asai's R_(2) contribution (the part of
    the Hessian that leaks into the corrected gradient G). Subtracting it is what
    makes the FI fit second-order-exact on a disordered cloud (see module docstring)."""
    P = sv.T @ G.T                     # (n_basis, d)
    return sv - X @ P.T                # (K, n_basis)


# --------------------------------------------------------------- the local row
def fi_grad_row(xij, w, ridge=1e-8):
    """Local FI gradient stencil: returns (dx_row, dy_row) of length K with
    <grad f>_i = (sum_k dx_row[k] (f_nb[k]-f_i), sum_k dy_row[k] (...)).

    Mirrors ``gfdm.gfdm_grad_row``. Algebraically identical to
    ``firm_core.grad_op`` (renormalised, linear-exact), returned row-wise.
    """
    xij = np.asarray(xij, float)
    w = np.asarray(w, float)
    s = _scale(xij)
    G = _grad_rows(xij / s, w, ridge)        # gradient w.r.t X = x/s
    return G[0] / s, G[1] / s                 # back to physical coords


def fi_row(xij, w, ridge=1e-8):
    """Local Asai Full-Inverse 2nd-order Laplacian stencil on neighbour offsets
    ``xij`` (K,2), weights ``w`` (K,).

    Returns ``(d, bc_value)`` with ``d`` of length K such that
        <lap f>_i = sum_k d[k] * (f_nb[k] - f_i) + bc_value ,
    matching ``gfdm.gfdm_row``'s contract. ``bc_value`` is always 0.0 (FI is an
    interior operator with no boundary closure).

    Exact for any quadratic (incl. cross terms) -> 2nd-order; linear-exact
    because the corrected gradient annihilates the linear part.
    """
    xij = np.asarray(xij, float)
    w = np.asarray(w, float)
    s = _scale(xij)
    X = xij / s
    G = _grad_rows(X, w, ridge)               # (2, K): <grad f>_X = G @ fij
    sv = _svec(X)                             # (K, 3)
    shat = _corrected_basis(sv, X, G)         # (K, 3) Hessian-leakage-corrected
    # {f_ij - X . <grad f>(1)} as a row in fij:  g = E @ fij,  E[k] = e_k - X_k @ G
    E = np.eye(len(w)) - X @ G                # (K, K)
    Mfi = np.einsum("k,ki,kj->ij", w, shat, shat)   # (3, 3) weighted second-moment
    if ridge:
        Mfi = Mfi + ridge * (float(np.trace(Mfi)) / 3.0 + 1e-30) * np.eye(3)
    rhs_rows = 2.0 * (w[:, None] * shat).T @ E   # (3, K)
    u_rows = np.linalg.solve(Mfi, rhs_rows)    # (3, K): rows for [f_XX, f_YY, 2 f_XY]
    d = (u_rows[0] + u_rows[1]) / (s * s)      # Laplacian = f_XX + f_YY, back to phys
    return d, 0.0


def fi_row_bd(xij, w, ridge=1e-8):
    """Block-Diagonal (BD) reduction of the FI operator: drop the cross-derivative
    column so [f_XX, f_YY] are fitted without the X1 X2 basis. Asai's Table 1 records
    BD == LDD == Schwaiger, i.e. this should reproduce FIRM's interior Laplacian
    (``firm_core.laplacian_interior``). Returns ``(d, bc_value)`` like ``fi_row``.

    Provided to *test* the BD==LDD equivalence claim, not for production use.
    """
    xij = np.asarray(xij, float)
    w = np.asarray(w, float)
    s = _scale(xij)
    X = xij / s
    G = _grad_rows(X, w, ridge)
    sv = _svec(X)[:, :2]                       # [X1^2, X2^2] only -- no cross column
    shat = _corrected_basis(sv, X, G)          # (K, 2)
    E = np.eye(len(w)) - X @ G
    Mss = np.einsum("k,ki,kj->ij", w, shat, shat)  # (2, 2)
    if ridge:
        Mss = Mss + ridge * (float(np.trace(Mss)) / 2.0 + 1e-30) * np.eye(2)
    rhs_rows = 2.0 * (w[:, None] * shat).T @ E    # (2, K)
    u_rows = np.linalg.solve(Mss, rhs_rows)     # rows for [f_XX, f_YY]
    d = (u_rows[0] + u_rows[1]) / (s * s)
    return d, 0.0


# --------------------------------------------------------------- global assembly
def assemble_fi(pos, dx, field, *, dir_mask=None, poly=None, h_factor=2.5, ridge=1e-8, pin=None):
    """Assemble (A, b, info) for the FI Poisson baseline.

    Mirrors ``gfdm.assemble_gfdm``. With ``poly`` given, particles within ``h`` of a
    wall receive the FI + algebraic-ghost Neumann closure: the support is completed
    by ``firm_core.ghost_complete`` (operator-agnostic mirror images carrying the
    Neumann flux) and the FI operator is evaluated on the completed support. Because
    the ghost completion is linear-reproducing, FI inherits linear-exactness at the
    boundary without any single-sum cancellation identity. The flux enters the RHS
    through the ghost increments; FI returns the Laplacian directly, so no ``1/N``
    scaling is applied (unlike the renormalised single-sum closure). Walls and
    Dirichlet are mutually exclusive per particle (Dirichlet wins). Particles with
    fewer than 3 neighbours are pinned to the exact value.
    """
    pos = np.asarray(pos, float)
    N = len(pos)
    h = h_factor * dx
    nl = fc.neighbor_lists(pos, h)
    if dir_mask is None:
        dir_mask = np.zeros(N, bool)

    R, C, D = [], [], []
    b = np.zeros(N)
    is_wall = np.zeros(N, bool)

    for i in range(N):
        if dir_mask[i] or i == pin:
            R.append(i); C.append(i); D.append(1.0)
            b[i] = field.value(pos[i])
            continue
        nb = nl[i]
        if len(nb) < 3:                        # FI 3x3 system under-determined
            R.append(i); C.append(i); D.append(1.0)
            b[i] = field.value(pos[i])
            continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)

        near = g2.nearby_walls(pos[i], poly, h) if poly is not None else []
        if near:                               # FI + algebraic-ghost Neumann closure
            is_wall[i] = True
            normals = np.array([n for _, n, _ in near])
            feet_rel = np.array([foot - pos[i] for _, _, foot in near])
            gflux = np.array([field.wall_flux(foot, n) for _, n, foot in near])
            offs, wts, src, inc = fc.ghost_complete(xij, w, normals, feet_rel, gflux, h)
            d, _ = fi_row(offs, wts, ridge=ridge)
            diag = 0.0
            for m in range(len(d)):
                col = i if src[m] < 0 else int(nb[src[m]])
                R.append(i); C.append(col); D.append(d[m]); diag -= d[m]
                b[i] -= d[m] * inc[m]          # ghost flux increment -> RHS
            R.append(i); C.append(i); D.append(diag)
            b[i] += field.laplacian(pos[i])
            continue

        d, bc_val = fi_row(xij, w, ridge=ridge)
        diag = 0.0
        for k, j in enumerate(nb):
            R.append(i); C.append(int(j)); D.append(d[k])
            diag -= d[k]
        R.append(i); C.append(i); D.append(diag)
        b[i] = field.laplacian(pos[i]) - bc_val

    if _HAVE_SPARSE:
        A = sps.csr_matrix((D, (R, C)), shape=(N, N))
    else:  # pragma: no cover
        A = np.zeros((N, N))
        for r, c, v in zip(R, C, D):
            A[r, c] += v
    return A, b, dict(h=h, nl=nl, is_wall=is_wall)


def solve(A, b):
    if _HAVE_SPARSE and hasattr(A, "tocsr"):
        return spsolve(A.tocsr(), b)
    return np.linalg.solve(A, b)
