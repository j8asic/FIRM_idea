"""Substep 15 -- independent analytic ground-truth for the FI second-derivative moments.

Asai et al. (2023), Eqs. (27)-(32), give the *closed-form* continuum moment
integrals that the Full-Inverse second-derivative system inverts on a uniform
cloud. They are pure analysis -- no particle code -- so they are an independent
check (against analytic truth, not against other numerical code) that our
understanding of the FI moment assembly is correct:

    F = (r . grad W) / |r|^4 ,   M'_ss[a,b] = integral_Omega F (r_a)^2 (r_b)^2 dx ,

with the kernel normalised so integral W dx = 1. Asai reports, for the poly6
kernel on a uniform distribution,

    2D:  M'_ss = [[3/4, 1/4], [1/4, 3/4]] ,   (M'_ss)^-1 = [[3/2, -1/2], [-1/2, 3/2]] ,
         M'_cc = [1/4]
    3D:  M'_ss = [[3/5,1/5,1/5],[1/5,3/5,1/5],[1/5,1/5,3/5]] ,
         (M'_ss)^-1 = [[2,-1/2,-1/2],[-1/2,2,-1/2],[-1/2,-1/2,2]] ,   M'_cc = (1/5) I

This test (a) confirms the published matrices are genuine inverses (transcription
check), and (b) reconstructs the moments by direct quadrature of the analytic
poly6 gradient on a fine regular lattice and checks they match -- the diagonal/
off-diagonal ratio is exactly 3:1 in both dimensions (normalisation-independent),
and the normalised matrix and its inverse reproduce Asai's Eqs. (29)-(32). The
poly6 derivative used here lives only in this verification; the production FI
operator (``fi.py``) is kernel-derivative-free.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TITLE = "Substep 15 -- analytic FI moment ground-truth (Asai 2023, Eqs. 27-32)"


def _moments(d, h=1.0, n=None):
    """Direct quadrature of M'_ss[a,b] = int F (r_a)^2 (r_b)^2 dx and M'_cc on a
    fine regular lattice in d dimensions. F = (r.gradW)/|r|^4 with the analytic
    poly6 gradient gradW = -6/h^2 (1-q^2)^2 r, normalised so int W dx = 1.
    Returns (M_ss (d,d), m_cc scalar) up to the (sign) convention of F."""
    n = n or (201 if d == 2 else 81)
    ax = np.linspace(-h, h, n)
    cell = (2.0 * h / (n - 1)) ** d
    grids = np.meshgrid(*([ax] * d), indexing="ij")
    R = np.stack([g.ravel() for g in grids], axis=1)
    r2 = (R * R).sum(1)
    inside = (r2 > 1e-12) & (r2 < h * h)
    R, r2 = R[inside], r2[inside]
    q2 = r2 / (h * h)
    F = -6.0 / (h * h) * (1.0 - q2) ** 2 / r2        # (r.gradW)/|r|^4
    intW = (((1.0 - q2) ** 3) * cell).sum()          # normalise kernel: int W dx = 1
    Mss = np.empty((d, d))
    for a in range(d):
        for b in range(d):
            Mss[a, b] = (F * R[:, a] ** 2 * R[:, b] ** 2 * cell).sum() / intW
    # cross block: s_c = r1 r2 -> M'_cc = int F (r1 r2)^2 dx  (== off-diagonal of M'_ss)
    m_cc = (F * (R[:, 0] * R[:, 1]) ** 2 * cell).sum() / intW
    return Mss, m_cc


def run(rep):
    # --- (a) transcription check: Asai's published matrices are genuine inverses
    Mss2 = np.array([[3 / 4, 1 / 4], [1 / 4, 3 / 4]])
    inv2 = np.array([[3 / 2, -1 / 2], [-1 / 2, 3 / 2]])
    rep.check_below("2D: published M'_ss and (M'_ss)^-1 are inverses (Eq 31)",
                    np.abs(Mss2 @ inv2 - np.eye(2)).max(), 1e-12,
                    f"max|M Minv - I| = {np.abs(Mss2 @ inv2 - np.eye(2)).max():.1e}")
    Mss3 = np.full((3, 3), 1 / 5) + (3 / 5 - 1 / 5) * np.eye(3)
    inv3 = np.full((3, 3), -1 / 2) + (2 - (-1 / 2)) * np.eye(3)
    rep.check_below("3D: published M'_ss and (M'_ss)^-1 are inverses (Eq 29)",
                    np.abs(Mss3 @ inv3 - np.eye(3)).max(), 1e-12,
                    f"max|M Minv - I| = {np.abs(Mss3 @ inv3 - np.eye(3)).max():.1e}")

    # --- (b) reconstruct moments by quadrature of the analytic poly6 gradient
    for d, Mref, cc_ref in [(2, Mss2, 1 / 4), (3, Mss3, 1 / 5)]:
        M, m_cc = _moments(d)
        M, m_cc = np.abs(M), abs(m_cc)            # F sign is a convention; compare magnitudes
        diag = float(np.mean(np.diag(M)))
        off = float(np.mean(M[~np.eye(d, dtype=bool)]))
        rep.check_close(f"{d}D: diagonal/off-diagonal moment ratio = 3:1 (norm-independent)",
                        diag / off, 3.0, tol=2e-2,
                        detail=f"diag {diag:.4f} / off {off:.4f} = {diag/off:.4f} (Asai 3.000)")
        Mn = M * (Mref[0, 0] / diag)              # normalise diagonal to Asai's value
        rep.check_below(f"{d}D: quadrature M'_ss matches Asai Eq {29 if d==3 else 31}",
                        np.abs(Mn - Mref).max(), 2e-2,
                        detail=f"max|M_quad - M_Asai| = {np.abs(Mn - Mref).max():.2e}")
        rep.check_below(f"{d}D: (quadrature M'_ss)^-1 matches Asai analytic inverse",
                        np.abs(np.linalg.inv(Mn) - np.linalg.inv(Mref)).max(), 1e-1,
                        detail=f"max|inv diff| = {np.abs(np.linalg.inv(Mn) - np.linalg.inv(Mref)).max():.2e}")
        rep.check_close(f"{d}D: cross block M'_cc equals the off-diagonal value (Eq {32 if d==2 else 30})",
                        m_cc * (Mref[0, 0] / diag), cc_ref, tol=2e-2,
                        detail=f"M'_cc {m_cc*(Mref[0,0]/diag):.4f} (Asai {cc_ref:.4f})")


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
