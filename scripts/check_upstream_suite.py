# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Check the port and wrong masks against the pinned upstream CPU suite.

    python scripts/check_upstream_suite.py PATH_TO_KARPO_DECISION_CHECKOUT

At the pinned commit, port has 35 passed; top-left has 15 failed; window-1,
window+1 and no-window have 12 failed each. Each run makes 138 attend calls.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

PINNED_COMMIT = "6600011eb5cf46232d056ec0717dc726146d29de"
VARIANTS = ("port", "top-left", "window-1", "window+1", "no-window")
TESTS = ("tests/test_gemma_blocks.py", "tests/test_gemma_padded.py", "tests/test_gemma_images.py")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkout", help="path to the pinned karpo-decision source checkout")
    checkout = Path(parser.parse_args().checkout).resolve()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True, text=True,
                            check=True).stdout.strip()
    if commit != PINNED_COMMIT:
        print(f"warning: upstream checkout is {commit}, expected {PINNED_COMMIT}", file=sys.stderr)

    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo / "tests"), str(repo), env.get("PYTHONPATH"))))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    success = True
    for variant in VARIANTS:
        env["KARPO_PORT_VARIANT"] = variant
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p",
                                 "upstream_plugin", *TESTS], cwd=checkout, env=env, capture_output=True, text=True)
        output = result.stdout + result.stderr
        calls = re.search(rf"variant={re.escape(variant)} attend calls=(\d+)", output)
        failed = re.search(r"(\d+) failed", output)
        passed = re.search(r"(\d+) passed", output)
        count = int(calls.group(1)) if calls else 0
        failures = int(failed.group(1)) if failed else 0
        passes = int(passed.group(1)) if passed else 0
        print(f"{variant}: {passes} passed, {failures} failed, attend calls={count}", flush=True)
        if variant == "port":  # noqa: SIM108
            valid = result.returncode == 0 and count > 0
        else:
            valid = failures > 0 and count > 0
        if not valid:
            success = False
            print(output, file=sys.stderr)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
