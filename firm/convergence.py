"""
Convergence harness for the FIRM boundary-closure paper, matching the 2017 protocol:

  * chaos levels c in {30, 60, 90}% (Basic et al. 2018): a node is displaced by a
    random vector in [-c*dx/2, +c*dx/2] per axis, so the geometry2d ``jitter``
    half-amplitude is jitter = c/2 (chaos 30% -> jitter 0.15, 60% -> 0.30, 90% -> 0.45);
  * refinement by halving dx;
  * >= 20-run (seed) averaging on scattered clouds (median, robust to outliers);
  * the 2017 normalised-RMSE error metric  ||<f> - f||_2 / ||f||_2  (poisson.rel_errors).

A driver supplies ``solve(dx, jitter, seed) -> error`` and this module sweeps it.
"""
import numpy as np

from testkit import observed_order

CHAOS = [0.30, 0.60, 0.90]
SEEDS = list(range(20))            # >= 20-run averaging (2017 protocol)


def jitter_for_chaos(c):
    """Map the 2017 chaos fraction c to the geometry2d jitter half-amplitude."""
    return 0.5 * c


def refine(solve, dxs, jitter, seeds=SEEDS):
    """Median error over seeds at each dx, plus the log-log convergence order.
    ``solve(dx, jitter, seed)`` returns a scalar normalised error."""
    errs = []
    for dx in dxs:
        vals = [solve(dx, jitter, s) for s in seeds]
        errs.append(float(np.median(vals)))
    return errs, observed_order(dxs, errs)


def sweep_chaos(solve, dxs, chaos=CHAOS, seeds=SEEDS):
    """Run ``refine`` at every chaos level. Returns {c: (errs, order)}."""
    return {c: refine(solve, dxs, jitter_for_chaos(c), seeds) for c in chaos}


def print_table(name, dxs, results, methods=None):
    """Pretty-print an error-vs-dx table. ``results`` is {label: (errs, order)}."""
    labels = methods if methods is not None else list(results.keys())
    print(f"\n{name}   (rel-L2 vs dx; chaos-averaged, median over seeds)")
    print("  dx     " + "".join(f"{lab:>16}" for lab in labels))
    for i, dx in enumerate(dxs):
        print(f"  {dx:6.4f} " + "".join(f"{results[lab][0][i]:>16.3e}" for lab in labels))
    print("  order  " + "".join(f"{results[lab][1]:>16.2f}" for lab in labels))
