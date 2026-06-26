"""
Run the entire FIRM validation suite: substeps 1-9 + the capstone.

    python3 run_all.py            # full suite, prints per-check PASS/FAIL + summary
    python3 run_all.py --plot     # also save capstone figures
    python3 tests/test_03_laplacian.py   # any substep standalone
    pytest tests/                 # the test_* functions are pytest-discoverable too

Exits nonzero if any check fails.
"""
import os
import sys
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

import testkit


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    tests_dir = os.path.join(HERE, "tests")
    modules = []
    for fname in sorted(os.listdir(tests_dir)):
        if fname.startswith("test_") and fname.endswith(".py"):
            modules.append(_load(os.path.join(tests_dir, fname), fname[:-3]))
    modules.append(_load(os.path.join(HERE, "capstone_poisson.py"), "capstone_poisson"))

    total_pass = total_fail = 0
    failed_modules = []
    for mod in modules:
        rep = testkit.Reporter(getattr(mod, "TITLE", mod.__name__))
        testkit.section(rep.title)
        mod.run(rep)
        rep.summary()
        total_pass += rep.passed
        total_fail += rep.failed
        if rep.failed:
            failed_modules.append(rep.title)

    testkit.section("SUITE SUMMARY")
    print(f"  total checks: {total_pass + total_fail}   passed: {total_pass}   failed: {total_fail}")
    if failed_modules:
        print(f"  modules with failures: {failed_modules}")
    else:
        print("  ALL GREEN -- every substep and the capstone passed.")

    if "--plot" in sys.argv:
        cap = modules[-1]
        res, ords = cap.sweep()
        cap.make_plots(res, ords, os.path.join(HERE, "capstone"))

    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
