"""
GFDM / 2nd-order WLSQ Laplacian baseline (for the JCP boundary-closure paper).

This module is a *comparison baseline*, not part of the FIRM trust anchor
(``firm_core.py``). It provides a genuinely second-order-consistent generalised
finite-difference (GFDM) Laplacian -- the "Asai-type" full-quadratic-basis
weighted-least-squares operator -- and two Neumann wall closures:

  * ``"penalty"``    : the soft flux constraint (a single large weighted row);
                       known NOT to converge (see ``prototype_neumann.py``).
  * ``"constraint"`` : the EXACT flux equality constraint enforced by a local
                       KKT/Lagrange saddle system (Tiwari--Kuhnert constraint row).

The interior operator is 2nd-order on a regular lattice and remains 2nd-order
*pointwise* under fixed-relative jitter (unlike the FIRM operator, which is only
linear-consistent and supraconverges). The price is sensitivity to the local
neighbour count: the 2D quadratic basis needs >= 5 well-spread neighbours, so the
moment system is more ill-conditioned than FIRM's at small support.

Conventions match ``poisson.py``: the assembled difference form is

    sum_j d_j (p_j - p_i)  =  source_i  -  bc_value_i ,

i.e. <lap p>_i = sum_j d_j (p_j - p_i) + bc_value_i is set equal to the source.
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


# --------------------------------------------------------------- the local row
def _basis(xs):
    """Full quadratic Taylor basis on scaled offsets xs=(K,2):
    columns [dx, dy, 1/2 dx^2, dx dy, 1/2 dy^2] -> coefficients
    [f_x, f_y, f_xx, f_xy, f_yy]."""
    x, y = xs[:, 0], xs[:, 1]
    return np.stack([x, y, 0.5 * x * x, x * y, 0.5 * y * y], axis=1)


_SEL = np.array([0.0, 0.0, 1.0, 0.0, 1.0])  # picks f_xx + f_yy


def gfdm_row(xij, w, bclist=(), ridge=1e-8, constraint="constraint"):
    """Local GFDM Laplacian stencil on neighbour offsets ``xij`` (K,2), weights ``w``.

    Returns ``(d, bc_value)`` with ``d`` of length K such that
        <lap f>_i = sum_k d[k] * (f_nb[k] - f_i) + bc_value .

    ``bclist`` is a list of ``(normal, g)`` Neumann data (flux g = grad f . n at the
    wall). With no constraints it is the pure interior operator. ``constraint`` selects
    the exact KKT equality constraint (default) or the soft ``"penalty"`` form.

    Offsets are internally scaled by the rms neighbour distance so the moment matrix
    is well conditioned; the 1/s^2 second-derivative factor is folded back into the
    returned coefficients.
    """
    xij = np.asarray(xij, float)
    w = np.asarray(w, float)
    s = np.sqrt(max(float(np.mean((xij * xij).sum(1))), 1e-30))
    A = _basis(xij / s)                       # (K,5)
    ATW = A.T * w                             # (5,K)
    G = ATW @ A                               # (5,5) weighted moment matrix
    if ridge:
        G = G + ridge * (float(np.trace(G)) / 5.0 + 1e-30) * np.eye(5)
    inv_s2 = 1.0 / (s * s)

    if not bclist:
        d = _SEL @ np.linalg.solve(G, ATW)    # (K,)
        return d * inv_s2, 0.0

    if constraint == "penalty":
        w_bc = 1e3 * float(w.mean())
        rhs = np.zeros(5)
        Gp = G.copy()
        for n, g in bclist:
            a = np.array([n[0], n[1], 0.0, 0.0, 0.0])
            Gp = Gp + w_bc * np.outer(a, a)
            rhs += w_bc * a * (g * s)          # scaled flux value
        Ginv = np.linalg.inv(Gp)
        d = _SEL @ Ginv @ ATW
        bc_value = float(_SEL @ Ginv @ rhs)
        return d * inv_s2, bc_value * inv_s2

    # exact equality constraint via the KKT saddle system
    m = len(bclist)
    C = np.zeros((m, 5))
    gvec = np.zeros(m)
    for r, (n, g) in enumerate(bclist):
        C[r] = [n[0], n[1], 0.0, 0.0, 0.0]
        gvec[r] = g * s                        # n . grad f = g  ->  scaled by s
    K = np.zeros((5 + m, 5 + m))
    K[:5, :5] = G
    K[:5, 5:] = C.T
    K[5:, :5] = C
    Kinv = np.linalg.inv(K)
    d = _SEL @ (Kinv[:5, :5] @ ATW)            # data part -> stencil
    bc_value = float(_SEL @ (Kinv[:5, 5:] @ gvec))  # constraint part -> RHS
    return d * inv_s2, bc_value * inv_s2


def gfdm_grad_row(xij, w, ridge=1e-8):
    """Local GFDM gradient stencil: returns (dx_row, dy_row) of length K with
    <grad f>_i = (sum_k dx_row[k] (f_nb[k]-f_i), sum_k dy_row[k] (...))."""
    xij = np.asarray(xij, float)
    w = np.asarray(w, float)
    s = np.sqrt(max(float(np.mean((xij * xij).sum(1))), 1e-30))
    A = _basis(xij / s)
    ATW = A.T * w
    G = ATW @ A
    if ridge:
        G = G + ridge * (float(np.trace(G)) / 5.0 + 1e-30) * np.eye(5)
    sol = np.linalg.solve(G, ATW)              # (5,K)
    return sol[0] / s, sol[1] / s


# --------------------------------------------------------------- global assembly
def assemble_gfdm(pos, dx, field, *, dir_mask=None, poly=None, h_factor=2.5,
                  ridge=1e-8, neumann="constraint", pin=None):
    """Assemble (A, b, info) for -- the GFDM Poisson baseline.

    dir_mask : bool (N,) particles pinned to the exact Dirichlet value (boundary).
    poly     : if given, particles within h of a wall get Neumann constraint rows
               (flux from ``field.wall_flux``). Walls and Dirichlet are mutually
               exclusive per particle (Dirichlet wins).
    pin      : optional index pinned to the exact value (pure-Neumann null space).
    neumann  : 'constraint' (exact KKT) or 'penalty'.
    """
    pos = np.asarray(pos, float)
    N = len(pos)
    h = h_factor * dx
    nl = fc.neighbor_lists(pos, h)
    if dir_mask is None:
        dir_mask = np.zeros(N, bool)
    seg = g2.polygon_segments(poly) if poly is not None else None

    R, C, D = [], [], []
    b = np.zeros(N)
    is_wall = np.zeros(N, bool)

    for i in range(N):
        if dir_mask[i] or i == pin:
            R.append(i); C.append(i); D.append(1.0)
            b[i] = field.value(pos[i])
            continue
        nb = nl[i]
        if len(nb) < 5:                        # quadratic basis under-determined
            R.append(i); C.append(i); D.append(1.0)
            b[i] = field.value(pos[i])
            continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)

        bclist = ()
        if seg is not None:
            hit = g2.nearest_segment(pos[i], seg[0], seg[1], seg[2], h)
            if hit is not None:
                _, n, foot = hit
                is_wall[i] = True
                bclist = [(n, field.wall_flux(foot, n))]

        d, bc_val = gfdm_row(xij, w, bclist, ridge=ridge, constraint=neumann)
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
    return A, b, dict(is_wall=is_wall, h=h, nl=nl)


def solve(A, b):
    if _HAVE_SPARSE and hasattr(A, "tocsr"):
        return spsolve(A.tocsr(), b)
    return np.linalg.solve(A, b)
