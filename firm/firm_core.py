"""
FIRM core operators (docs/spec.md Sections 1-5), numpy-only.

This is the single trust anchor: every substep test AND the capstone import
these exact functions, so green substeps genuinely imply a trustworthy capstone.
Conventions are LOCKED here (see plan):
  * kernel  W(r,h) = (1-q^2)^3 for q=r/h<1, else 0
  * N_i     = (2/d) tr(B_i)                       (Sec 1.2 spherical-mean form)
  * project in physical space, THEN renormalize   (Sec 2.4)
"""
from itertools import combinations

import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAVE_KDTREE = True
except Exception:  # pragma: no cover
    _HAVE_KDTREE = False


# --------------------------------------------------------------- kernel / search
def kernel(r, h):
    """Poly6-style weight  W = max(0, 1 - q^2)^3,  q = r/h  (compact support h).

    Chosen after a kernel/support study (compare_kernels.py): at MATCHED support the
    flat-topped poly6 uses its neighbours more evenly than the concentrated spiky
    cubic (1-q)^3 == (1-0.4 r/dx)^3, giving lower Poisson-solution error at SMALL
    support and cleaner mirror-ghost wall convergence (the spiky kernel is competitive
    only at large support). Default support h = 2.5*dx (~15 interior / >=4 wall
    neighbours at 30% jitter); 2.0-2.25*dx is viable for the quasi-uniform clouds of
    incompressible flow (first neighbour ring ~ dx).
    """
    q = np.asarray(r, float) / h
    return np.where(q < 1.0, (1.0 - q * q) ** 3, 0.0)


def neighbor_lists(pos, h):
    """For each particle, indices of other particles within h (self excluded)."""
    pos = np.asarray(pos, float)
    n = len(pos)
    if _HAVE_KDTREE:
        tree = cKDTree(pos)
        raw = tree.query_ball_point(pos, h)
        return [np.array([j for j in raw[i] if j != i], dtype=int) for i in range(n)]
    out = []
    for i in range(n):
        d = np.linalg.norm(pos - pos[i], axis=1)
        out.append(np.where((d < h) & (d > 1e-12))[0])
    return out


# --------------------------------------------------------------- Sec 1.2 geometry
class Geom:
    __slots__ = ("o", "S", "M", "B", "N", "d", "K", "ok")

    def __init__(self, o, S, M, B, N, d, K, ok):
        self.o, self.S, self.M, self.B, self.N = o, S, M, B, N
        self.d, self.K, self.ok = d, K, ok


def geom_quantities(xij, w, ridge=0.0):
    """o, S, M, B=M^{-1}, N=(2/d)tr(B) from neighbor offsets xij (K,d) and weights w (K,)."""
    xij = np.asarray(xij, float)
    w = np.asarray(w, float)
    d = xij.shape[1]
    K = len(w)
    o = (w[:, None] * xij).sum(0)
    S = float(w.sum())
    M = np.einsum("k,ki,kj->ij", w, xij, xij)
    ok = K >= d + 1
    if ridge:
        M = M + ridge * np.eye(d)
    B = np.linalg.inv(M)
    N = (2.0 / d) * float(np.trace(B))
    return Geom(o, S, M, B, N, d, K, ok)


# --------------------------------------------------------------- Sec 1.3/1.4 ops
def grad_op(B, xij, w, fij):
    """Renormalized gradient  B . sum_j W_ij f_ij x_ij  (linear-exact)."""
    xij = np.asarray(xij, float)
    rhs = (w[:, None] * (fij[:, None] * xij)).sum(0)
    return B @ rhs


def correction_weights(xij, V):
    """w_ij = 1 - x_ij . V  (Sec 1.4 / 3.6)."""
    return 1.0 - np.asarray(xij, float) @ V


def laplacian_interior(geom, xij, w, fij):
    """Boxed Sec 1.4 operator:  N sum_j W_ij f_ij (1 - x_ij . B o)."""
    V = geom.B @ geom.o
    wij = correction_weights(xij, V)
    return geom.N * float((w * fij * wij).sum())


def laplacian_normalization(geom, xij, w, wij, mode="trb"):
    """Laplacian normalization N_i. 'trb' = (2/d) tr(B) (default). 'denom' =
    2d / sum_j W_ij |x_ij|^2 w_ij -- exact for the isotropic quadratic |x|^2;
    falls back to 'trb' if the (signed) denominator is near zero/negative."""
    if mode == "trb":
        return geom.N
    xij = np.asarray(xij, float)
    den = float((w * (xij * xij).sum(1) * wij).sum())
    return (2.0 * geom.d) / den if den > 1e-30 else geom.N


def laplacian_naive(geom, xij, w, fij):
    """Two-term form  N sum_j W_ij (f_ij - x_ij . grad f)  (Sec 1.4 start)."""
    g = grad_op(geom.B, xij, w, fij)
    return geom.N * float((w * (fij - xij @ g)).sum())


# --------------------------------------------------------------- Sec 2.2 projectors
def proj_AN(normals, betas):
    """Averaged-normal tangential projector. Returns (P_tan, n_eff)."""
    normals = np.asarray(normals, float)
    betas = np.asarray(betas, float)
    d = normals.shape[1]
    n_eff = (betas[:, None] * normals).sum(0)
    nn = np.linalg.norm(n_eff)
    n_eff = n_eff / nn
    return np.eye(d) - np.outer(n_eff, n_eff), n_eff


def proj_GGP(normals, eps=0.0):
    """General Gram projector. Returns (P_tan, Nmat (d,K), Ginv)."""
    normals = np.asarray(normals, float)
    d, K = normals.shape[1], normals.shape[0]
    Nmat = normals.T  # (d, K)
    G = Nmat.T @ Nmat
    if eps:
        G = G + eps * np.eye(K)
    Ginv = np.linalg.inv(G)
    P = np.eye(d) - Nmat @ Ginv @ Nmat.T
    return P, Nmat, Ginv


def proximity_betas(deltas, h_w):
    """Normalized wall-blending weights from raw scores max(0, 1 - d/h_w)^2 (Sec 2.2)."""
    deltas = np.asarray(deltas, float)
    bhat = np.maximum(0.0, 1.0 - deltas / h_w) ** 2
    s = bhat.sum()
    return bhat / s if s > 0 else np.zeros_like(bhat)


# --------------------------------------------------------------- Sec 3.3/3.4 surface
def lambda_detect(r, dx):
    return float(np.linalg.norm(r)) * dx


def sigma_rational(lam, c=0.2):
    return lam * lam / (lam * lam + c * c)


def sigma_smoothstep(lam):
    t = 3.0 * (lam - 2.0 / 3.0)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return 3.0 * t * t - 2.0 * t * t * t


_ACTIVATION = {"rational": sigma_rational, "smoothstep": lambda lam, c=0.2: sigma_smoothstep(lam)}


def surface_proj_regularized(o_w, r, sigma, eta):
    """sigma * (o_w . r) r / (|r|^2 + eta^2)  -- regularized surface-normal removal."""
    r2 = float(r @ r)
    return sigma * (float(o_w @ r) / (r2 + eta * eta)) * r


# --------------------------------------------------------------- Sec 2.3 / 4.2 flux
def wall_flux_rhs_AN(g_eff, o, n_eff):
    return g_eff * float(o @ n_eff)


def wall_flux_rhs_GGP(Nmat, Ginv, o, g):
    return float((Nmat.T @ o) @ (Ginv @ np.asarray(g, float)))


# --------------------------------------------------------------- Sec 5.2 grad recon
def grad_recon_wall_AN(grad_p, n_eff, g_eff):
    return grad_p - float(grad_p @ n_eff) * n_eff + g_eff * n_eff


def grad_recon_wall_GGP(grad_p, P, Nmat, Ginv, g):
    return P @ grad_p + Nmat @ (Ginv @ np.asarray(g, float))


# --------------------------------------------------------------- mirror-ghost wall closure
def _reflect_comps(K):
    """Reflection group compositions for K near walls: all non-empty subsets
    (singles + corner double-reflections + ...). K=1 -> [(0,)]; K=2 ->
    [(0,),(1,),(0,1)] (the (0,1) entry is the corner double-reflection R_B(R_A))."""
    out = []
    for r in range(1, K + 1):
        out.extend(combinations(range(K), r))
    return out


def reflect_complete(xi, xij, w, planes, h):
    """General even/odd mirror-ghost stencil completion across a reflection group
    (Sec 5.1 cure; LeVeque ghost-node). Reflects every fluid neighbour AND particle i
    across the group generated by the nearby boundary planes, assigning each ghost a
    boundary-consistent value, and returns the completed support so any interior
    operator can be run on it. This is the single device behind every ghost closure:
    even reflection for a Neumann wall, odd reflection for a Dirichlet/free surface,
    and a composed even+odd reflection at a contact line.

    Parameters
    ----------
    xi     : (d,) absolute node position (used only to evaluate a callable Dirichlet
             value at a ghost's perpendicular foot).
    xij, w : fluid-neighbour offsets (relative to xi) and kernel weights.
    planes : list of dicts, one per boundary plane, each
             {'n': outward unit normal (d,), 'foot': point on the plane RELATIVE to xi,
              'kind': 'neumann' | 'dirichlet',
              'data': for 'neumann', the flux g = grad p . n (scalar);
                      for 'dirichlet', the surface value p_target -- a scalar (constant
                      surface value, e.g. 0 or a surface-tension datum) or a callable
                      f(x_abs) evaluated at the ghost's perpendicular foot (a spatially
                      varying value, e.g. a manufactured solution)}.

    Returns aligned arrays (offsets, weights, src_local, mult, const). A completed term
    has field value  mult*p_src + const, where src_local[m] indexes the fluid neighbour
    (-1 == particle i). For real neighbours (mult, const) = (1, 0). For a ghost the value
    accumulates over the reflections of its composition:
        neumann (even):   value -> value - 2 sigma g          (mult unchanged)
        dirichlet (odd):  value -> 2 p_target - value         (mult flips sign).
    Linear-reproducing: a ghost of a field satisfying the boundary condition equals the
    true field at the mirrored point, so any linear-exact operator inherits consistency
    -- which is why the renormalised single-sum, GFDM and Full-Inverse operators all
    accept this completion with no cancellation identity re-derived.
    """
    xij = np.asarray(xij, float)
    xi = np.asarray(xi, float)
    n_nb, d = xij.shape
    comps = _reflect_comps(len(planes))
    offs = list(xij)
    wts = list(np.asarray(w, float))
    src_local = list(range(n_nb))
    mult = [1.0] * n_nb
    const = [0.0] * n_nb
    sources = [(k, xij[k]) for k in range(n_nb)] + [(-1, np.zeros(d))]  # neighbours + i
    for sl, p0 in sources:
        for comp in comps:
            p = p0.copy()
            mu = 1.0
            cs = 0.0
            for pidx in comp:
                pl = planes[pidx]
                nrm = np.asarray(pl["n"], float)
                foot = np.asarray(pl["foot"], float)
                sig = float((p - foot) @ nrm)
                if pl["kind"] == "dirichlet":
                    data = pl["data"]
                    pt = float(data(xi + (p - sig * nrm))) if callable(data) else float(data)
                    cs = 2.0 * pt - cs
                    mu = -mu
                else:  # neumann
                    cs += -2.0 * sig * float(pl["data"])
                p = p - 2.0 * sig * nrm
            rr = float(np.linalg.norm(p))
            if 1e-12 < rr < h:
                offs.append(p); wts.append(kernel(rr, h))
                src_local.append(sl); mult.append(mu); const.append(cs)
    return np.array(offs), np.array(wts), np.array(src_local), np.array(mult), np.array(const)


def ghost_complete(xij, w, normals, feet_rel, g, h):
    """Even (Neumann) mirror-ghost completion: a thin wrapper over ``reflect_complete``
    for the all-Neumann case, where every ghost value is purely additive (mult == 1,
    value = p_src + inc). Kept for the renormalised single-sum closure
    (``mirror_ghost_terms``) and the Full-Inverse Neumann path (``fi.assemble_fi``).
    Returns (offsets, weights, src_local, inc).
    """
    planes = [dict(n=np.asarray(normals[k], float), foot=np.asarray(feet_rel[k], float),
                   kind="neumann", data=float(g[k])) for k in range(len(normals))]
    d = np.asarray(xij, float).shape[1]
    offs, wts, src_local, _mult, const = reflect_complete(np.zeros(d), xij, w, planes, h)
    return offs, wts, src_local, const   # mult == 1 throughout; const == inc


def mirror_ghost_terms(xij, w, normals, feet_rel, g, h, norm="trb"):
    """Mirror-ghost completion of a Neumann-wall particle's stencil for the
    renormalised single-sum operator. Completes the support via ``ghost_complete``
    and forms the renormalised coefficients ``coeff = W_ij w_ij`` and the
    normalization ``N``. Returns aligned arrays (src_local, coeff, inc, N); see
    ``ghost_complete`` for the support semantics.
    """
    offs, wts, src_local, inc = ghost_complete(xij, w, normals, feet_rel, g, h)
    gm = geom_quantities(offs, wts)
    V = gm.B @ gm.o
    wij = correction_weights(offs, V)
    coeff = wts * wij
    # the normalization of the ghost-completed row. The sum-version denominator
    # N=2d/sum(W|x|^2 w) (exact for |x|^2) removes the trace form's plateau and
    # restores ~2nd-order convergence at the Neumann boundary; see the paper.
    if norm == "denom":
        den = float((wts * (offs * offs).sum(1) * wij).sum())
        N = (2.0 * gm.d) / den if den > 1e-30 else gm.N
    else:
        N = gm.N
    return np.array(src_local), coeff, np.array(inc), N


# --------------------------------------------------------------- unified local op
class Local:
    """Local unified-Poisson operator pieces for one particle (Sec 4)."""
    __slots__ = ("geom", "w", "wij", "P_tan", "n_eff", "Nmat", "Ginv", "o_w",
                 "r", "sigma", "n_s", "delta_est", "surf_point", "o_star", "V",
                 "robin_diag", "b_wall", "has_surface")


def particle_operator(xi, xij, w, *, walls=None, surface=None, dx=None,
                      wall_proj="GGP", eps_gram=0.01, c=0.2, activation="rational",
                      raycast=None):
    """Assemble the unified local operator for one particle (no field knowledge).

    Parameters
    ----------
    xi      : (d,) particle position (needed only to return the natural surface point)
    xij, w  : fluid-neighbor offsets (K,d) and kernel weights (K,)
    walls   : None | dict(normals=(Kw,d), deltas=(Kw,), g=(Kw,) Neumann flux, h_w=float)
    surface : None | dict(mode='natural') | dict(mode='exact', n_s, delta, ...)
    dx      : nominal spacing (for lambda_i and eta=c/dx)

    Returns a ``Local`` holding wij, robin_diag, b_wall, surf_point, etc.
    The CALLER supplies the manufactured source and the surface target value
    (evaluated at loc.surf_point for natural, or provided in `surface` for exact).
    """
    geom = geom_quantities(xij, w)
    loc = Local()
    loc.geom = geom
    loc.w = np.asarray(w, float)
    loc.P_tan = loc.n_eff = loc.Nmat = loc.Ginv = None
    loc.r = loc.sigma = loc.n_s = loc.delta_est = loc.surf_point = None
    loc.robin_diag = 0.0
    loc.b_wall = 0.0
    loc.has_surface = False

    d = geom.d
    # ---- wall (Neumann) projection, Sec 2.2-2.4 ----------------------------
    if walls is not None and len(walls["normals"]) > 0:
        normals = np.asarray(walls["normals"], float)
        deltas = np.asarray(walls["deltas"], float)
        g = np.asarray(walls["g"], float)
        betas = proximity_betas(deltas, walls["h_w"])
        if wall_proj == "AN":
            P, n_eff = proj_AN(normals, betas)
            loc.P_tan, loc.n_eff = P, n_eff
            g_eff = float((betas * g).sum())
            loc.b_wall = wall_flux_rhs_AN(g_eff, geom.o, n_eff)
        else:  # GGP
            P, Nmat, Ginv = proj_GGP(normals, eps=eps_gram if len(normals) > 1 else 0.0)
            loc.P_tan, loc.Nmat, loc.Ginv = P, Nmat, Ginv
            loc.b_wall = wall_flux_rhs_GGP(Nmat, Ginv, geom.o, g)
        o_w = P @ geom.o
    else:
        loc.P_tan = np.eye(d)
        o_w = geom.o.copy()
    loc.o_w = o_w

    # ---- free surface (Robin/Dirichlet), Sec 3.3-3.6 ----------------------
    o_star = o_w.copy()
    if surface is not None:
        loc.has_surface = True
        r = geom.B @ o_w
        loc.r = r
        mode = surface.get("mode", "natural")
        if mode == "natural":
            lam = lambda_detect(r, dx)
            act = _ACTIVATION[activation]
            sigma = act(lam, c) if activation == "rational" else act(lam)
            rn = float(np.linalg.norm(r))
            n_s = -r / rn if rn > 0 else np.zeros(d)
            # natural distance estimate (Sec 3.3): delta = |o_w . n_s| / S
            odn = float(o_w @ n_s)
            delta_est = abs(odn) / geom.S if geom.S > 0 else 0.0
            # optional ray-cast wall suppression (Sec 3.4.1)
            if raycast is not None and sigma > 0 and rn > 0:
                d_wall = raycast(xi, -r / rn)
                d_start, d_end = np.sqrt(d) * dx, 2.0 * np.sqrt(d) * dx
                atten = np.clip((d_wall - d_start) / (d_end - d_start), 0.0, 1.0)
                sigma *= atten
            eta = c / dx
            o_star = o_w - surface_proj_regularized(o_w, r, sigma, eta)
            loc.sigma, loc.n_s, loc.delta_est = sigma, n_s, delta_est
            loc.surf_point = np.asarray(xi, float) + delta_est * n_s
            loc.robin_diag = sigma * geom.S            # cancellation -> sigma*S
        elif mode == "exact":
            n_s = np.asarray(surface["n_s"], float)
            delta = float(surface["delta"])
            sigma = float(surface.get("sigma", 1.0))
            odn = float(o_w @ n_s)
            o_star = o_w - sigma * odn * n_s            # remove exact normal component
            loc.sigma, loc.n_s, loc.delta_est = sigma, n_s, delta
            loc.robin_diag = sigma * (-odn) / delta     # beta_surf = -(o_w.n_s)/delta
            loc.surf_point = np.asarray(xi, float) + delta * n_s
    loc.o_star = o_star
    loc.V = geom.B @ o_star
    loc.wij = correction_weights(xij, loc.V)
    return loc
