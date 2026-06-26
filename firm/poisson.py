"""
Unified FIRM pressure-Poisson assembly + solve (docs/spec.md Sec 4).

Shared by the hydrostatic substep test and the capstone so both exercise the
SAME assembly. Normalized difference form (Sec 4.1):

    sum_j W_ij w_ij p_ij  -  sigma_i S_i p_i  =  b_i

    A[i,j] += W_ij w_ij                 (off-diagonal, j != i)
    A[i,i] += -sum_j W_ij w_ij - robin_diag_i
    b[i]    = lap_exact_i / N_i  +  b_wall_i  -  robin_diag_i * p_target_i

with w_ij = 1 - x_ij . V_i and V_i = B_i o_*,i from firm_core.particle_operator.
Wall flux is analytic (g_k = grad p_exact . n_k at the wall foot); the surface
target is the manufactured value at the (estimated or exact) surface point.
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


def assemble(pos, dx, field, *, poly=None, fill_h=None, wall_proj="GGP",
             surface_mode="exact", activation="rational", c=0.2, h_factor=2.5,
             eps_gram=0.01, raycast=False, wall_closure="projection", norm="trb",
             source=None):
    """Build (A, b, info) for the unified Poisson with a manufactured solution.

    poly      : wall polygon (None -> no walls)
    fill_h    : free-surface height (None -> no surface)
    surface_mode : 'exact' | 'natural' | None
    wall_closure : 'projection' (FIRM tangential projection + flux RHS) or
                   'ghost' (mirror-ghost stencil completion; lower wall error). Ghost
                   is applied to pure-Neumann-wall particles; surface/contact-line and
                   interior particles always use the projection path.
    source    : None -> use the analytic field.laplacian(pos[i]) as the RHS source
                (default). An (N,) array overrides it with a numeric per-particle
                source f_i (e.g. the discrete divergence D u* of a predicted velocity
                in a pressure-projection solve); it enters the RHS as f_i / N_i exactly
                like the analytic source. Backward-compatible: behaviour is unchanged
                when source is None.  info["N"] returns the per-particle normalization
                N_i used on each row (needed to rebuild the RHS for an iterated solve).
    """
    pos = np.asarray(pos, float)
    N = len(pos)
    h = h_factor * dx
    nl = fc.neighbor_lists(pos, h)

    ray = None
    if raycast and poly is not None:
        ray = lambda xi, dirn: g2.raycast_segment_hit(xi, dirn, poly, max_dist=4 * dx)

    R, C, D = [], [], []
    b = np.zeros(N)
    is_wall = np.zeros(N, bool)
    is_surf = np.zeros(N, bool)
    sigma = np.zeros(N)
    Nvec = np.ones(N)                       # per-particle Laplacian normalization N_i

    def _src(i):
        return field.laplacian(pos[i]) if source is None else float(source[i])

    for i in range(N):
        nb = nl[i]
        if len(nb) < 3:
            R.append(i); C.append(i); D.append(1.0)   # isolated -> pin to itself
            b[i] = field.value(pos[i])
            continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)

        walls = None
        near = None
        if poly is not None:
            near = g2.nearby_walls(pos[i], poly, h)
            if near:
                normals = np.array([n for _, n, _ in near])
                deltas = np.array([d for d, _, _ in near])
                g = np.array([field.wall_flux(foot, n) for d, n, foot in near])
                walls = dict(normals=normals, deltas=deltas, g=g, h_w=h)
                is_wall[i] = True

        surface = None
        d_surf = None
        if fill_h is not None and surface_mode is not None:
            d_surf = fill_h - pos[i, 1]
            if 0.0 < d_surf < h:
                is_surf[i] = True
                if surface_mode == "exact":
                    surface = dict(mode="exact", n_s=np.array([0.0, 1.0]), delta=d_surf, sigma=1.0)
                else:
                    surface = dict(mode="natural")

        # mirror-ghost closure for pure-Neumann-wall particles (Sec 5.1 cure)
        if wall_closure == "ghost" and is_wall[i] and not is_surf[i]:
            normals = np.array([n for _, n, _ in near])
            feet_rel = np.array([foot - pos[i] for _, _, foot in near])
            gflux = np.array([field.wall_flux(foot, n) for _, n, foot in near])
            src_local, coeff, inc, Nrm = fc.mirror_ghost_terms(xij, w, normals, feet_rel, gflux, h, norm=norm)
            diag = 0.0
            for m in range(len(coeff)):
                col = i if src_local[m] < 0 else int(nb[src_local[m]])
                R.append(i); C.append(col); D.append(coeff[m])
                diag -= coeff[m]
                b[i] -= coeff[m] * inc[m]
            R.append(i); C.append(i); D.append(diag)
            b[i] += _src(i) / Nrm
            Nvec[i] = Nrm
            continue

        loc = fc.particle_operator(pos[i], xij, w, walls=walls, surface=surface, dx=dx,
                                   wall_proj=wall_proj, eps_gram=eps_gram, c=c,
                                   activation=activation, raycast=ray)
        sigma[i] = loc.sigma if loc.sigma is not None else 0.0

        # surface target value
        p_target = 0.0
        if surface is not None:
            if surface_mode == "exact":
                p_target = float(field.value(np.array([pos[i, 0], fill_h])))
            else:
                p_target = float(field.value(loc.surf_point))

        wsum = float((w * loc.wij).sum())
        for k, j in enumerate(nb):
            R.append(i); C.append(j); D.append(w[k] * loc.wij[k])
        R.append(i); C.append(i); D.append(-wsum - loc.robin_diag)
        N_src = fc.laplacian_normalization(loc.geom, xij, w, loc.wij, mode=norm)
        b[i] = _src(i) / N_src + loc.b_wall - loc.robin_diag * p_target
        Nvec[i] = N_src

    if _HAVE_SPARSE:
        A = sps.csr_matrix((D, (R, C)), shape=(N, N))
    else:  # pragma: no cover
        A = np.zeros((N, N))
        for r, col, v in zip(R, C, D):
            A[r, col] += v
    info = dict(is_wall=is_wall, is_surf=is_surf, sigma=sigma, h=h, nl=nl, N=Nvec)
    return A, b, info


def solve(A, b):
    if _HAVE_SPARSE and hasattr(A, "tocsr"):
        return spsolve(A.tocsr(), b)
    return np.linalg.solve(A, b)


def rel_errors(p, p_exact, mask=None):
    if mask is None:
        mask = np.ones(len(p), bool)
    err = p[mask] - p_exact[mask]
    l2 = np.linalg.norm(err) / max(np.linalg.norm(p_exact[mask]), 1e-30)
    linf = np.max(np.abs(err)) / max(np.max(np.abs(p_exact[mask])), 1e-30)
    return l2, linf
