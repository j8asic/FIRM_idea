"""
FIRM boundary-value-problem assembly on a general (curved/polygonal) domain.

A thin driver layer over the ``firm_core`` trust anchor, used by the Gibou-style
benchmarks (curved Dirichlet, Robin, Neumann). It is kept separate from
``poisson.py`` (which serves the substep tests + the free-surface capstone) so the
validated assembly there is untouched; both call the same ``firm_core`` operators.

Three boundary-condition closures on one polygon boundary:

  * ``"dirichlet"`` : value integration -- the same exact ``surface`` closure used for
    the free surface, with n_s/delta taken from the nearest wall and the target value
    p = u(foot). Adds the Robin diagonal -sigma S_i and the value to the RHS.
  * ``"neumann"``   : flux integration -- tangential projection (AN/GGP) + flux RHS, or
    mirror-ghost stencil completion (``wall_closure="ghost"``).
  * ``"robin"``     : du/dn + alpha u = f. The known part f enters the flux RHS exactly
    as in the Neumann closure; the alpha u_i part is linear in the unknown and moves to
    the diagonal: A_ii += alpha * q, q = (N^T o)^T G^{-1} 1.
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


def assemble(pos, dx, field, poly, bc, *, h_factor=2.5, wall_proj="GGP",
             wall_closure="projection", robin_alpha=1.0, norm="trb", pin=None):
    """Build (A, b, info) for a Poisson BVP on ``poly`` with one BC type ``bc`` in
    {'dirichlet','neumann','robin'}. ``field`` supplies the manufactured solution
    (value/grad/laplacian) and analytic BC data. ``pin`` pins one index to the exact
    value (needed for the pure-Neumann null space)."""
    pos = np.asarray(pos, float)
    N = len(pos)
    h = h_factor * dx
    nl = fc.neighbor_lists(pos, h)
    seg_a, seg_b, seg_n = g2.polygon_segments(poly)

    R, C, D = [], [], []
    b = np.zeros(N)
    is_bnd = np.zeros(N, bool)

    for i in range(N):
        nb = nl[i]
        if len(nb) < 3 or i == pin:
            R.append(i); C.append(i); D.append(1.0)
            b[i] = field.value(pos[i])
            continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        # Smooth boundary: model the local wall by the single nearest segment (its
        # perpendicular foot lies along the local normal, which the Robin/value
        # closures rely on for linear-exactness). Genuine non-orthogonal corners are
        # exercised by the wedge capstone via poisson.assemble (GGP), not here.
        hit = g2.nearest_segment(pos[i], seg_a, seg_b, seg_n, h)
        near = [hit] if hit is not None else []

        # ---- interior -----------------------------------------------------
        if not near:
            gm = fc.geom_quantities(xij, w)
            V = gm.B @ gm.o
            wij = fc.correction_weights(xij, V)
            wsum = float((w * wij).sum())
            for k, j in enumerate(nb):
                R.append(i); C.append(int(j)); D.append(w[k] * wij[k])
            R.append(i); C.append(i); D.append(-wsum)
            Nn = fc.laplacian_normalization(gm, xij, w, wij, mode=norm)
            b[i] = field.laplacian(pos[i]) / Nn
            continue

        is_bnd[i] = True
        normals = np.array([n for _, n, _ in near])
        feet = np.array([foot for _, _, foot in near])
        dists = np.array([d for d, _, _ in near])

        # ---- Neumann: mirror-ghost stencil completion --------------------
        if bc == "neumann" and wall_closure == "ghost":
            feet_rel = feet - pos[i]
            gflux = np.array([field.wall_flux(feet[k], normals[k]) for k in range(len(near))])
            src_local, coeff, inc, Nrm = fc.mirror_ghost_terms(xij, w, normals, feet_rel, gflux, h, norm=norm)
            diag = 0.0
            for m in range(len(coeff)):
                col = i if src_local[m] < 0 else int(nb[src_local[m]])
                R.append(i); C.append(col); D.append(coeff[m])
                diag -= coeff[m]
                b[i] -= coeff[m] * inc[m]
            R.append(i); C.append(i); D.append(diag)
            b[i] += field.laplacian(pos[i]) / Nrm
            continue

        gm = fc.geom_quantities(xij, w)

        # ---- Dirichlet: exact value closure (nearest wall) ---------------
        if bc == "dirichlet":
            d0, n0, foot0 = near[0]
            surface = dict(mode="exact", n_s=n0, delta=max(d0, 1e-12), sigma=1.0)
            loc = fc.particle_operator(pos[i], xij, w, surface=surface, dx=dx)
            wsum = float((w * loc.wij).sum())
            p_target = float(field.value(foot0))
            for k, j in enumerate(nb):
                R.append(i); C.append(int(j)); D.append(w[k] * loc.wij[k])
            R.append(i); C.append(i); D.append(-wsum - loc.robin_diag)
            Nn = fc.laplacian_normalization(gm, xij, w, loc.wij, mode=norm)
            b[i] = field.laplacian(pos[i]) / Nn - loc.robin_diag * p_target
            continue

        # ---- Neumann / Robin: tangential projection + flux RHS -----------
        # Per segment the flux is g_k = grad u . n_k. For Robin (du/dn + alpha u = f)
        # the wall value satisfies u(foot_k) = u_i + delta_k g_k (perpendicular foot),
        # so g_k = (f_k - alpha u_i)/(1 + alpha delta_k) = a_k - c_k u_i with
        # a_k = f_k/(1+alpha delta_k), c_k = alpha/(1+alpha delta_k). The a-part is the
        # flux RHS; the c-part (linear in the unknown u_i) moves to the diagonal. This
        # smoothly recovers the Neumann (alpha=0) and Dirichlet (alpha->inf) closures and
        # stays linear-exact.
        if bc == "robin":
            fseg = np.array([field.grad(feet[k]) @ normals[k] + robin_alpha * field.value(feet[k])
                             for k in range(len(near))])
            denom = 1.0 + robin_alpha * dists
            avec = fseg / denom
            cvec = robin_alpha / denom
        else:
            avec = np.array([field.wall_flux(feet[k], normals[k]) for k in range(len(near))])
            cvec = np.zeros(len(near))

        if wall_proj == "AN":
            betas = fc.proximity_betas(dists, h)
            P, n_eff = fc.proj_AN(normals, betas)
            o_w = P @ gm.o
            b_wall = fc.wall_flux_rhs_AN(float((betas * avec).sum()), gm.o, n_eff)
            q = fc.wall_flux_rhs_AN(float((betas * cvec).sum()), gm.o, n_eff)
        else:  # GGP
            P, Nmat, Ginv = fc.proj_GGP(normals, eps=0.01 if len(normals) > 1 else 0.0)
            o_w = P @ gm.o
            b_wall = fc.wall_flux_rhs_GGP(Nmat, Ginv, gm.o, avec)
            q = fc.wall_flux_rhs_GGP(Nmat, Ginv, gm.o, cvec)

        V = gm.B @ o_w
        wij = fc.correction_weights(xij, V)
        wsum = float((w * wij).sum())
        for k, j in enumerate(nb):
            R.append(i); C.append(int(j)); D.append(w[k] * wij[k])
        R.append(i); C.append(i); D.append(-wsum + q)
        Nn = fc.laplacian_normalization(gm, xij, w, wij, mode=norm)
        b[i] = field.laplacian(pos[i]) / Nn + b_wall

    if _HAVE_SPARSE:
        A = sps.csr_matrix((D, (R, C)), shape=(N, N))
    else:  # pragma: no cover
        A = np.zeros((N, N))
        for r, c, v in zip(R, C, D):
            A[r, c] += v
    return A, b, dict(is_bnd=is_bnd, h=h, nl=nl)


def solve(A, b):
    if _HAVE_SPARSE and hasattr(A, "tocsr"):
        return spsolve(A.tocsr(), b)
    return np.linalg.solve(A, b)
