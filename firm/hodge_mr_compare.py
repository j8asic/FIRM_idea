"""Fixed-step vs minimal-residual (MR) iterated Helmholtz-Hodge decomposition.

Reuses the manufactured fixture and operators of hodge_projection.py. For each case it
reports, for BOTH the fixed unit-step defect correction (iterate) and the optimal-step MR
variant (iterate_mr): the single-projection reduction r1/r0, the floor min_k r_k/r0 (at k*),
and the asymptotic per-step contraction rho_inf (->1 = stall, >1 = divergence). The decisive
rows are the `raw` ones (full truncated wall divergence as the source), where the fixed step
diverges (rho>1) and the MR step must stay non-expansive (rho<=1).

Run:  python firm/hodge_mr_compare.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hodge_projection as H


def _stats(rs):
    rr = np.array(rs) / max(rs[0], 1e-30)
    kf = int(np.argmin(rr))
    rho, _ = H.contraction(rs)
    return rs[1] / max(rs[0], 1e-30), float(rr[kf]), kf, rho


def main():
    print("=" * 104)
    print("ITERATED HODGE: fixed unit step (Richardson) vs minimal-residual optimal step")
    print("=" * 104)
    cases = [
        ("box", 0.045, "projection", 0.0),
        ("box", 0.045, "projection", 0.3),
        ("box", 0.045, "ghost", 0.3),
        ("tank", 0.045, "projection", 0.3),
        ("tank", 0.045, "ghost", 0.3),
    ]
    hdr = ("  domain dx     closure    jit  raw |   FIXED: r1/r0  floor(@k*)  rho_inf |"
           "   MR: r1/r0  floor(@k*)  rho_inf")
    print(hdr)
    print("  " + "-" * 100)
    for domain, dx, wc, jit in cases:
        case = H.run_case(domain, dx, wc, "trb", 7, jit)
        for raw in (False, True):
            f1, ff, fk, frho = _stats(H.iterate(case, raw=raw))
            m1, mf, mk, mrho = _stats(H.iterate_mr(case, raw=raw))
            print("  %-5s  %.3f  %-10s %.1f  %3s | %8.3f  %.2e(%2d) %7.3f |"
                  "  %8.3f  %.2e(%2d) %7.3f"
                  % (domain, dx, wc, jit, "Y" if raw else "n",
                     f1, ff, fk, frho, m1, mf, mk, mrho))
    print("  " + "-" * 100)
    print("  rho_inf -> 1 : stall (residual floor, not zero).  rho_inf > 1 : divergence.")
    print("  MR guarantees ||r_{k+1}|| <= ||r_k|| (rho_inf <= 1) on every row, incl. raw.")
    print("=" * 104)


if __name__ == "__main__":
    main()
