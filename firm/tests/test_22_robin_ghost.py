"""Substep 22 -- Robin-parity reflection ghost (du/dn + alpha u = f).

The even (Neumann) and odd (Dirichlet) reflections of ``firm_core.reflect_complete``
are the two parity limits of a single Robin transform. Reflecting a source with
signed plane distance sigma = (p - foot).n (< 0 for interior sources) and running
value v, the ghost value

    v_ghost = ((1 + alpha sigma) v - 2 sigma f) / (1 - alpha sigma)

enforces du/dn + alpha u = f on the plane: alpha = 0 recovers the even (Neumann,
g = f) transform, alpha -> inf with f = alpha p_target the odd (Dirichlet) one, and
sigma < 0 with alpha >= 0 keeps the denominator >= 1 (no singularity). For any
linear field satisfying the condition the ghost value equals the true field at the
mirrored point, so the completed stencil stays linear-exact like the even/odd cases.

This test asserts the limits and the exactness to round-off, then measures the
closure's convergence -- straight Robin edge on the unit square and the curved B3
flower -- against the existing diagonal-Robin (projection + alpha-diagonal) closure
of ``bvp.py`` on identical clouds. The diagonal closure sits near first order; the
completed two-sided stencil restores ~2nd order, mirroring the Neumann (test 10/16)
and Dirichlet (test 17/18/20) ghost stories.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
import geometry2d as g2
import manufactured as mf
import bvp
from testkit import observed_order

try:
    import scipy.sparse as sps
    from scipy.sparse.linalg import spsolve
    _HAVE_SPARSE = True
except Exception:  # pragma: no cover
    _HAVE_SPARSE = False

TITLE = "Substep 22 -- Robin-parity reflection ghost (straight edge + curved flower)"
ALPHA = 1.0
JSON_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "figures", "paper_extra_numbers.json")
SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
SQ_SEG = g2.polygon_segments(SQUARE)
FLOWER = g2.flower_polygon(n=720, base=0.5, amp=0.08, k=8, center=(0.5, 0.5))
FL_SEG = g2.polygon_segments(FLOWER)


def _robin_ghost_row(pos_i, xij, w, nrm, foot, frob, h):
    """Robin-ghost completion of one boundary row for the renormalised single-sum
    (FIRM) operator with the sum normalisation N = 2d / sum(W |x|^2 wij).
    Returns (offs, src, mult, cst, coeff, N)."""
    planes = [dict(n=nrm, foot=foot - pos_i, kind="robin",
                   data=dict(alpha=ALPHA, f=frob))]
    offs, wts, src, mult, cst = fc.reflect_complete(pos_i, xij, w, planes, h)
    gm = fc.geom_quantities(offs, wts)
    wij = fc.correction_weights(offs, gm.B @ gm.o)
    coeff = wts * wij
    den = float((wts * (offs * offs).sum(1) * wij).sum())
    N = (2.0 * gm.d) / den if den > 1e-30 else gm.N
    return offs, src, mult, cst, coeff, N


# ------------------------------------------------------------------ (a) parity limits
def _limit_checks(dx=0.05, jitter=0.30, seed=3):
    """Compare the robin transform against its parity limits on a jittered
    half-plane cloud (wall y=cut, outward +e_y): alpha=0 vs the even (neumann, g=f)
    transform and alpha=1e12 with f=alpha*p_target vs the odd (dirichlet) one.
    Returns (max_diff_neumann, max_diff_dirichlet) over all boundary rows."""
    h = 2.5 * dx
    pos, nrm, cut = g2.half_plane_cloud(dx, jitter, seed)
    nl = fc.neighbor_lists(pos, h)
    g_const, p_t, big = 0.37, -0.53, 1e12
    max_neu = max_dir = 0.0
    for i in np.where(pos[:, 1] > cut - h)[0]:
        nb = nl[i]
        if len(nb) < 3:
            continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        foot = np.array([0.0, cut - pos[i, 1]])
        _, _, _, mn, cn = fc.reflect_complete(
            pos[i], xij, w, [dict(n=nrm, foot=foot, kind="neumann", data=g_const)], h)
        _, _, _, mr, cr = fc.reflect_complete(
            pos[i], xij, w,
            [dict(n=nrm, foot=foot, kind="robin", data=dict(alpha=0.0, f=g_const))], h)
        max_neu = max(max_neu, float(np.abs(mn - mr).max()), float(np.abs(cn - cr).max()))
        _, _, _, md, cd = fc.reflect_complete(
            pos[i], xij, w, [dict(n=nrm, foot=foot, kind="dirichlet", data=p_t)], h)
        _, _, _, mr, cr = fc.reflect_complete(
            pos[i], xij, w,
            [dict(n=nrm, foot=foot, kind="robin", data=dict(alpha=big, f=big * p_t))], h)
        max_dir = max(max_dir, float(np.abs(md - mr).max()), float(np.abs(cd - cr).max()))
    return max_neu, max_dir


# --------------------------------------------------------------- (b) linear exactness
def _linear_exactness(dx=0.05, jitter=0.30, seed=5):
    """Linear field u = a.x + b satisfying du/dn + ALPHA u = f on y=0 (f evaluated at
    each perpendicular foot). Over all Robin-edge rows of a jittered unit square,
    returns the max |ghost value - true field at the mirrored point|, the max
    |completed renormalised Laplacian| (must vanish: lap u = 0) and the max raw row
    residual (row applied to the exact field minus its zero RHS)."""
    h = 2.5 * dx
    a = np.array([0.4, -0.7])
    lin = mf.linear_field(a, 0.23)
    nrm = np.array([0.0, -1.0])

    def frob(X):
        return float(lin.grad(X) @ nrm + ALPHA * lin.value(X))

    pos = g2.jittered_box(dx, jitter, seed)
    nl = fc.neighbor_lists(pos, h)
    m_ghost = m_lap = m_row = 0.0
    for i in np.where(pos[:, 1] < h)[0]:
        nb = nl[i]
        if len(nb) < 3:
            continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        foot = np.array([pos[i, 0], 0.0])
        offs, src, mult, cst, coeff, N = _robin_ghost_row(pos[i], xij, w, nrm, foot, frob, h)
        u_src = np.array([lin.value(pos[i]) if s < 0 else lin.value(pos[nb[s]]) for s in src])
        vals = mult * u_src + cst
        for m in range(len(nb), len(offs)):              # ghost terms only
            m_ghost = max(m_ghost, abs(vals[m] - float(lin.value(pos[i] + offs[m]))))
        resid = float((coeff * vals).sum()) - float(coeff.sum()) * float(lin.value(pos[i]))
        m_lap = max(m_lap, abs(N * resid))               # laplacian of a linear field = 0
        m_row = max(m_row, abs(resid))                   # row consistency vs the zero RHS
    return m_ghost, m_lap, m_row


# ------------------------------------------------- (c) straight-edge convergence
def _square_robin_edge(dx, jitter, seed, field, closure):
    """Unit square: Robin (du/dn + ALPHA u = f) on y=0, FIRM Dirichlet value closure
    on the other three edges, FIRM interior. closure in {'ghost', 'diagonal'};
    identical clouds and identical non-Robin rows, so only the Robin closure differs.
    Returns the Robin-strip rel-L2."""
    pos = g2.jittered_box(dx, jitter, seed)
    n = len(pos)
    h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    isR = (pos[:, 1] < h) & (pos[:, 0] > h) & (pos[:, 0] < 1 - h)
    isB = ((pos[:, 0] < h) | (pos[:, 0] > 1 - h) | (pos[:, 1] < h) | (pos[:, 1] > 1 - h))
    nrm = np.array([0.0, -1.0])

    def frob(X):
        return float(field.grad(X) @ nrm + ALPHA * field.value(X))

    R, C, D = [], [], []
    b = np.zeros(n)
    for i in range(n):
        nb = nl[i]
        if len(nb) < 3:
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        if isR[i]:
            foot = np.array([pos[i, 0], 0.0])
            if closure == "ghost":
                _, src, mult, cst, coeff, N = _robin_ghost_row(pos[i], xij, w, nrm, foot, frob, h)
                diag = 0.0
                for m in range(len(coeff)):
                    col = i if src[m] < 0 else int(nb[src[m]])
                    R.append(i); C.append(col); D.append(coeff[m] * mult[m])
                    diag -= coeff[m]; b[i] -= coeff[m] * cst[m]
                R.append(i); C.append(i); D.append(diag)
                b[i] += field.laplacian(pos[i]) / N
            else:  # the existing bvp.py diagonal-Robin closure (projection + alpha diag)
                delta = float(pos[i, 1])
                den = 1.0 + ALPHA * delta
                avec = np.array([frob(foot) / den])
                cvec = np.array([ALPHA / den])
                gm = fc.geom_quantities(xij, w)
                P, Nm, Gi = fc.proj_GGP(np.array([nrm]))
                b_wall = fc.wall_flux_rhs_GGP(Nm, Gi, gm.o, avec)
                q = fc.wall_flux_rhs_GGP(Nm, Gi, gm.o, cvec)
                wij = fc.correction_weights(xij, gm.B @ (P @ gm.o))
                for k, j in enumerate(nb):
                    R.append(i); C.append(int(j)); D.append(w[k] * wij[k])
                R.append(i); C.append(i); D.append(-float((w * wij).sum()) + q)
                b[i] = field.laplacian(pos[i]) / gm.N + b_wall
            continue
        if isB[i]:  # FIRM Dirichlet value closure w.r.t. the nearest square edge
            hit = g2.nearest_segment(pos[i], SQ_SEG[0], SQ_SEG[1], SQ_SEG[2], h)
            d0, n0, foot0 = hit
            loc = fc.particle_operator(
                pos[i], xij, w,
                surface=dict(mode="exact", n_s=n0, delta=max(d0, 1e-12), sigma=1.0), dx=dx)
            for k, j in enumerate(nb):
                R.append(i); C.append(int(j)); D.append(w[k] * loc.wij[k])
            R.append(i); C.append(i); D.append(-float((w * loc.wij).sum()) - loc.robin_diag)
            b[i] = field.laplacian(pos[i]) / loc.geom.N - loc.robin_diag * float(field.value(foot0))
            continue
        gm = fc.geom_quantities(xij, w)
        wij = fc.correction_weights(xij, gm.B @ gm.o)
        for k, j in enumerate(nb):
            R.append(i); C.append(int(j)); D.append(w[k] * wij[k])
        R.append(i); C.append(i); D.append(-float((w * wij).sum()))
        b[i] = field.laplacian(pos[i]) / gm.N
    A = sps.csr_matrix((D, (R, C)), shape=(n, n))
    p = spsolve(A, b)
    pe = field.value(pos)
    return float(np.linalg.norm((p - pe)[isR]) / max(np.linalg.norm(pe[isR]), 1e-30))


# ------------------------------------------------------ (d) curved (B3 flower)
def _flower_robin(dx, jitter, seed, field, closure):
    """B3 flower with Robin (du/dn + ALPHA u = f) everywhere on the boundary.
    'ghost' = Robin-parity reflection across the nearest boundary facet (exactly the
    test_18 local-plane treatment); 'diagonal' = the existing bvp.assemble Robin
    closure. Returns interior rel-L2 (the B3 metric) on the identical cloud."""
    pos = g2.polygon_cloud(FLOWER, dx, jitter, seed)
    pe = field.value(pos)
    if closure == "diagonal":
        A, b, info = bvp.assemble(pos, dx, field, FLOWER, "robin", robin_alpha=ALPHA)
        p = bvp.solve(A, b)
        m = ~info["is_bnd"]
        return float(np.linalg.norm((p - pe)[m]) / max(np.linalg.norm(pe[m]), 1e-30))
    n = len(pos)
    h = 2.5 * dx
    nl = fc.neighbor_lists(pos, h)
    R, C, D = [], [], []
    b = np.zeros(n)
    bnd = np.zeros(n, bool)
    for i in range(n):
        nb = nl[i]
        if len(nb) < 3:
            R.append(i); C.append(i); D.append(1.0); b[i] = field.value(pos[i]); continue
        xij = pos[nb] - pos[i]
        w = fc.kernel(np.linalg.norm(xij, axis=1), h)
        hit = g2.nearest_segment(pos[i], FL_SEG[0], FL_SEG[1], FL_SEG[2], h)
        if hit is not None:
            bnd[i] = True
            _, nrm, foot = hit

            def frob(X, nv=nrm):
                return float(field.grad(X) @ nv + ALPHA * field.value(X))

            _, src, mult, cst, coeff, N = _robin_ghost_row(pos[i], xij, w, nrm, foot, frob, h)
            diag = 0.0
            for m in range(len(coeff)):
                col = i if src[m] < 0 else int(nb[src[m]])
                R.append(i); C.append(col); D.append(coeff[m] * mult[m])
                diag -= coeff[m]; b[i] -= coeff[m] * cst[m]
            R.append(i); C.append(i); D.append(diag)
            b[i] += field.laplacian(pos[i]) / N
            continue
        gm = fc.geom_quantities(xij, w)
        wij = fc.correction_weights(xij, gm.B @ gm.o)
        for k, j in enumerate(nb):
            R.append(i); C.append(int(j)); D.append(w[k] * wij[k])
        R.append(i); C.append(i); D.append(-float((w * wij).sum()))
        b[i] = field.laplacian(pos[i]) / gm.N
    A = sps.csr_matrix((D, (R, C)), shape=(n, n))
    p = spsolve(A, b)
    m = ~bnd
    return float(np.linalg.norm((p - pe)[m]) / max(np.linalg.norm(pe[m]), 1e-30))


def _save_json(payload):
    d = {}
    if os.path.exists(JSON_OUT):
        with open(JSON_OUT) as f:
            d = json.load(f)
    d["robin_ghost"] = payload
    with open(JSON_OUT, "w") as f:
        json.dump(d, f, indent=2)


def run(rep):
    # ---- (a) parity limits -------------------------------------------------
    m_neu, m_dir = _limit_checks()
    rep.check_below("robin(alpha=0) reproduces the even/neumann transform (g=f)",
                    m_neu, 1e-14, f"max |mult/const diff| = {m_neu:.2e}")
    rep.check_below("robin(alpha=1e12, f=alpha p_t) reproduces the odd/dirichlet transform",
                    m_dir, 1e-9,
                    f"max |mult/const diff| = {m_dir:.2e} (finite-alpha residue O(1/(alpha|sigma|)))")

    # ---- (b) linear exactness ----------------------------------------------
    m_ghost, m_lap, m_row = _linear_exactness()
    rep.check_below("robin ghost of a BC-satisfying linear field = field at the mirrored point",
                    m_ghost, 1e-13, f"max |diff| = {m_ghost:.2e}")
    rep.check_below("completed renormalised Laplacian vanishes on that linear field",
                    m_lap, 1e-12, f"max |L u| = {m_lap:.2e}")
    rep.check_below("robin-ghost row is consistent on that linear field",
                    m_row, 1e-12, f"max |row residual| = {m_row:.2e}")

    # ---- (c) straight Robin edge, ghost vs diagonal on identical clouds ----
    field_s = mf.complex_field(np.pi, 0.3)
    dxs_s = [0.08, 0.04, 0.02, 0.01]
    seeds = list(range(6))
    e_sg = [float(np.median([_square_robin_edge(dx, 0.30, s, field_s, "ghost")
                             for s in seeds])) for dx in dxs_s]
    o_sg = observed_order(dxs_s, e_sg)
    e_sd = [float(np.median([_square_robin_edge(dx, 0.30, s, field_s, "diagonal")
                             for s in seeds])) for dx in dxs_s]
    o_sd = observed_order(dxs_s, e_sd)
    print(f"  [info] straight edge: ghost errs={['%.2e' % e for e in e_sg]} order={o_sg:.2f} | "
          f"diagonal errs={['%.2e' % e for e in e_sd]} order={o_sd:.2f} (recorded, no assert)")
    rep.check_order("robin ghost converges ~2nd order on the straight edge", o_sg,
                    expected=1.8, slack=0.5,
                    detail=f"ghost order {o_sg:.2f} (errs={['%.2e' % e for e in e_sg]}); "
                           f"diagonal closure order {o_sd:.2f}")

    # ---- (d) curved B3 flower, ghost vs diagonal on identical clouds -------
    field_f = mf.trig_field(np.pi)
    dxs_f = [0.05, 0.035, 0.025, 0.018]
    e_fg = [float(np.median([_flower_robin(dx, 0.30, s, field_f, "ghost")
                             for s in seeds])) for dx in dxs_f]
    o_fg = observed_order(dxs_f, e_fg)
    e_fd = [float(np.median([_flower_robin(dx, 0.30, s, field_f, "diagonal")
                             for s in seeds])) for dx in dxs_f]
    o_fd = observed_order(dxs_f, e_fd)
    rep.check("robin ghost beats the diagonal-Robin closure on the B3 flower",
              o_fg > o_fd + 0.4,
              f"ghost order {o_fg:.2f} (finest {e_fg[-1]:.2e}) vs "
              f"diagonal {o_fd:.2f} (finest {e_fd[-1]:.2e})")

    _save_json(dict(
        meta=dict(alpha=ALPHA, jitter=0.30, seeds=len(seeds),
                  note="Robin-parity reflection ghost vs the bvp.py diagonal-Robin "
                       "closure on identical clouds; median over seeds; sum "
                       "normalisation N=2d/sum(W|x|^2 w) on the ghost rows"),
        limits=dict(alpha0_vs_neumann_maxdiff=m_neu,
                    alpha1e12_vs_dirichlet_maxdiff=m_dir,
                    linear_ghost_vs_mirrored_point=m_ghost,
                    linear_laplacian_max=m_lap,
                    linear_row_residual_max=m_row),
        straight_edge=dict(field="complex(pi,0.3)", dxs=dxs_s,
                           ghost_errs=e_sg, ghost_order=o_sg,
                           diagonal_errs=e_sd, diagonal_order=o_sd),
        flower=dict(field="trig(pi)", dxs=dxs_f,
                    ghost_errs=e_fg, ghost_order=o_fg, ghost_finest=e_fg[-1],
                    diagonal_errs=e_fd, diagonal_order=o_fd, diagonal_finest=e_fd[-1]),
    ))


if __name__ == "__main__":
    import testkit
    rep = testkit.Reporter(TITLE)
    testkit.section(TITLE)
    run(rep)
    sys.exit(0 if rep.summary() else 1)


def test_substep():
    import testkit
    rep = testkit.Reporter(TITLE)
    run(rep)
    assert rep.failed == 0, f"{rep.failed} failed: {rep.fails}"
