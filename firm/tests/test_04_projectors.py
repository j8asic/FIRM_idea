"""Substep 4 -- Wall tangential projectors AN & GGP (docs/spec.md Sec 2.2, 2.7)."""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import firm_core as fc
from testkit import RND

TITLE = "Substep 4 -- Wall projectors (AN / GGP)"


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def run(rep):
    # --- properties on a non-orthogonal 130-deg pair (2D)
    th = np.deg2rad(130)
    n1, n2 = np.array([1.0, 0.0]), np.array([np.cos(th), np.sin(th)])
    P, Nmat, Ginv = fc.proj_GGP(np.array([n1, n2]), eps=0.0)
    rep.check_below("GGP idempotent (P^2 = P)", np.linalg.norm(P @ P - P), RND)
    rep.check_below("GGP symmetric", np.linalg.norm(P - P.T), RND)
    rep.check_below("GGP P n1 = 0", np.linalg.norm(P @ n1), RND)
    rep.check_below("GGP P n2 = 0", np.linalg.norm(P @ n2), RND)

    # --- K=1: AN == GGP
    PA, neff = fc.proj_AN(np.array([n1]), np.array([1.0]))
    PG, _, _ = fc.proj_GGP(np.array([n1]))
    rep.check_below("K=1: AN == GGP", np.linalg.norm(PA - PG), RND)

    # --- orthogonal pair: GGP -> 0 (2D, two orthogonal walls span everything);
    #     AN with a single averaged normal does NOT (documents AN's limitation)
    e1, e2 = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    PGo, _, _ = fc.proj_GGP(np.array([e1, e2]))
    PAo, _ = fc.proj_AN(np.array([e1, e2]), np.array([0.5, 0.5]))
    rep.check_below("orthogonal K=2: GGP P = 0", np.linalg.norm(PGo), RND)
    rep.check("orthogonal K=2: AN P != 0 (AN limitation)", np.linalg.norm(PAo) > 0.1,
              f"||P_AN|| = {np.linalg.norm(PAo):.3f}")

    # --- K=2 closed-form Ginv
    c = float(n1 @ n2)
    Gref = (1.0 / (1.0 - c * c)) * np.array([[1.0, -c], [-c, 1.0]])
    rep.check_below("K=2 closed-form G^{-1}", np.linalg.norm(Ginv - Gref), RND,
                    f"max diff = {np.linalg.norm(Ginv - Gref):.2e}")

    # --- near-parallel regularization (Sec 2.2): n1.n2 -> 0.999
    nA = np.array([1.0, 0.0]); nB = _unit(np.array([0.999, np.sqrt(1 - 0.999 ** 2)]))
    Preg, _, Gi = fc.proj_GGP(np.array([nA, nB]), eps=0.01)
    rep.check("near-parallel: G_reg invertible & finite", np.all(np.isfinite(Gi)),
              f"||P_reg|| = {np.linalg.norm(Preg):.3f}")

    # --- 3 non-coplanar walls in 3D -> P = 0 (Sec 2.7)
    m1, m2, m3 = np.array([1.0, 0, 0]), _unit([0.2, 1.0, 0]), _unit([0.1, 0.1, 1.0])
    P3, _, _ = fc.proj_GGP(np.array([m1, m2, m3]))
    rep.check_below("3D 3 non-coplanar walls: P = 0", np.linalg.norm(P3), RND)

    # --- proximity betas sum to 1
    betas = fc.proximity_betas(np.array([0.2, 0.5]), h_w=1.0)
    rep.check_close("proximity betas sum to 1", betas.sum(), 1.0, RND,
                    f"betas = {betas.round(4)}")


if __name__ == "__main__":
    import testkit
    rep = testkit.Reporter(TITLE)
    testkit.section(TITLE)
    run(rep)
    sys.exit(0 if rep.summary() else 1)


def test_substep():
    """pytest entry point: every check in this substep must pass."""
    import testkit
    rep = testkit.Reporter(TITLE)
    run(rep)
    assert rep.failed == 0, f"{rep.failed} failed: {rep.fails}"
