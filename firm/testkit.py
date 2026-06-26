"""
Tiny zero-dependency test harness for the FIRM substep suite.

Each test module exposes ``run(reporter)``; ``run_all.py`` aggregates. A check
records a PASS/FAIL with a human-readable detail line so a researcher can SEE
the error magnitudes and convergence orders, not just a green dot. Also works
under pytest (the test_* functions assert), but the primary interface is the
printed report.
"""
import sys
import numpy as np

RND = 1e-10  # round-off tolerance for linear-exact / consistency / identities


class Reporter:
    def __init__(self, title=""):
        self.title = title
        self.passed = 0
        self.failed = 0
        self.fails = []

    def check(self, name, ok, detail=""):
        ok = bool(ok)
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {name}" + (f"  --  {detail}" if detail else ""))
        if ok:
            self.passed += 1
        else:
            self.failed += 1
            self.fails.append(name)
        return ok

    def check_close(self, name, value, target, tol=RND, detail=""):
        """value within tol of target (absolute)."""
        d = abs(float(value) - float(target))
        return self.check(name, d <= tol, detail or f"|{value:.3e} - {target:.3e}| = {d:.3e} (tol {tol:.1e})")

    def check_below(self, name, value, tol=RND, detail=""):
        v = float(value)
        return self.check(name, v <= tol, detail or f"{v:.3e} <= {tol:.1e}")

    def check_order(self, name, observed, expected, slack=0.3, detail=""):
        ok = observed >= expected - slack
        return self.check(name, ok, detail or f"observed order {observed:.2f} (expected >= {expected - slack:.2f})")

    def summary(self):
        total = self.passed + self.failed
        head = f"{self.title}: " if self.title else ""
        print(f"{head}{self.passed}/{total} passed" + (f"  FAILED: {self.fails}" if self.fails else ""))
        return self.failed == 0


def section(label):
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)


def observed_order(dxs, errs):
    """Least-squares slope of log(err) vs log(dx)."""
    dxs = np.asarray(dxs, float)
    errs = np.asarray(errs, float)
    good = errs > 0
    return float(np.polyfit(np.log(dxs[good]), np.log(errs[good]), 1)[0])


def run_module(mod):
    """Run a module's ``run(reporter)`` and return True if all checks passed."""
    rep = Reporter(getattr(mod, "TITLE", mod.__name__))
    section(rep.title)
    mod.run(rep)
    ok = rep.summary()
    if not ok and __name__ == "__main__":
        sys.exit(1)
    return ok
