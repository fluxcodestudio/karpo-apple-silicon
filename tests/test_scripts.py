# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""What ``scripts/compare_reference.py`` and ``scripts/check_upstream_suite.py`` do before they need a checkpoint or
an upstream checkout: the command lines they refuse, and the summary the comparison writes. Needs torch only."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import karpo_apple_silicon

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def script(name: str):
    """``scripts/<name>.py`` as a module (the scripts folder is not a package)."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CompareReference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compare = script("compare_reference")

    def main(self, *options: str) -> None:
        """Run the command line with files that do not exist: anything past the argument checks fails on them."""
        arguments = ["compare_reference.py", *(part for name in ("checkpoint", "reference", "rows", "out")
                                               for part in (f"--{name}", f"no-such-{name}.jsonl")), *options]
        with mock.patch.object(sys, "argv", arguments):
            self.compare.main()

    def test_wrong_arguments_are_refused_before_any_file_is_read(self):
        for options, message in (
                (("--variants", "port,nothing"), "variants must be from"),
                (("--variants", "float16,port"), "float16 must be the last variant"),
                (("--variants", "float16,float16"), "float16 must be the last variant"),
                (("--per-cell", "1,2,3"), "--per-cell takes 4 non-negative integers"),
                (("--per-cell", "1,2,3,4,5"), "--per-cell takes 4 non-negative integers"),
                (("--per-cell", "1,2,many,4"), "--per-cell takes 4 non-negative integers"),
                (("--per-cell", "1,-2,3,4"), "--per-cell takes 4 non-negative integers"),
                (("--control-cells", "single-long,every"), "--control-cells must be from"),
                (("--control-cells", ""), "--control-cells must be from")):
            stderr = io.StringIO()
            with self.subTest(options=options):
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    self.main(*options)
                self.assertEqual(raised.exception.code, 2)  # argparse's usage error
                self.assertIn(message, stderr.getvalue())

    def test_right_arguments_get_as_far_as_the_files(self):
        for options in ((), ("--variants", "port,no-window,top-left,window-1,float16"), ("--per-cell", "0,0,3,0"),
                        ("--control-cells", "multi-long")):
            with self.subTest(options=options), self.assertRaises(FileNotFoundError):
                self.main(*options)

    def test_the_summary_of_finished_requests(self):
        """``run`` over requests a progress file already holds asks the model nothing, so its summary is exact:
        per cell, per track and overall, a request whose token count differed counted but not compared."""
        sample = [("single-short", "a", "r1", {}, {}), ("multi-long", "a", "r2", {}, {}),
                  ("single-short", "b", "r3", {}, {})]
        records = [dict(zip(("variant", "run_id", "seconds", "tokens", "diffs", "top_changed"), values, strict=True))
                   for values in (("port", "r1", 1.0, 100, [0.01], 0), ("port", "r2", 3.0, 2000, [0.02, 0.10], 1),
                                  ("port", "r3", 2.0, 300, None, 0),         # its token count differed
                                  ("float16", "r1", 9.0, 100, [0.5], 1))]    # another variant's: not counted

        def entry(requests, questions, mismatches, changed, diffs, over, tokens, seconds):
            return {"requests": requests, "questions": questions, "token_mismatches": mismatches,
                    "top_answers_changed": changed, "max_abs_probability_difference": max(diffs) if diffs else None,
                    "mean_abs_probability_difference": statistics.fmean(diffs) if diffs else None,
                    "p99_abs_probability_difference": max(diffs) if diffs else None,
                    "questions_differing_by_more_than_0.05": over, "median_input_tokens": statistics.median(tokens),
                    "max_input_tokens": max(tokens), "median_seconds_per_request": seconds}

        expected = {"all": entry(3, 3, 1, 1, [0.01, 0.02, 0.10], 1, [100, 2000, 300], 2.0),
                    "multi-long": entry(1, 2, 0, 1, [0.02, 0.10], 1, [2000], 3.0),
                    "single-short": entry(2, 1, 1, 0, [0.01], 0, [100, 300], 1.5),
                    "track:a": entry(2, 3, 0, 1, [0.01, 0.02, 0.10], 1, [100, 2000], 2.0),
                    "track:b": entry(1, 0, 1, 0, [], 0, [300], 2.0)}
        with tempfile.TemporaryDirectory() as folder:
            progress = Path(folder) / "comparison.json.progress.jsonl"
            progress.write_text("".join(json.dumps(record) + "\n" for record in records))
            with contextlib.redirect_stdout(io.StringIO()):
                summary = self.compare.run(None, sample, self.compare.nothing, self.compare.nothing, progress, "port")
            self.assertEqual(len(progress.read_text().splitlines()), len(records))  # nothing was run again
        self.assertEqual(json.dumps(summary), json.dumps(expected))  # the same entries in the same order


class CheckUpstreamSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.check = script("check_upstream_suite")

    def test_the_pinned_commit_is_the_packages(self):
        self.assertIs(self.check.TESTED_UPSTREAM_COMMIT, karpo_apple_silicon.TESTED_UPSTREAM_COMMIT)
        self.assertNotIn(karpo_apple_silicon.TESTED_UPSTREAM_COMMIT, (SCRIPTS / "check_upstream_suite.py").read_text())
        self.assertLessEqual({"TESTED_UPSTREAM_COMMIT", "TESTED_UPSTREAM_COMMITS"}, set(karpo_apple_silicon.__all__))

    def test_a_path_that_is_not_a_git_checkout_is_one_line_and_exit_status_2(self):
        with tempfile.TemporaryDirectory() as folder:
            plain = Path(folder) / "plain folder"
            plain.mkdir()
            # git looks no higher than the temporary folder, wherever that is
            with mock.patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": str(Path(folder).resolve())}):
                for path in (plain, Path(folder) / "no such folder"):
                    stderr = io.StringIO()
                    with self.subTest(path=path.name):
                        with mock.patch.object(sys, "argv", ["check_upstream_suite.py", str(path)]), \
                             contextlib.redirect_stderr(stderr):
                            self.assertEqual(self.check.main(), 2)
                        self.assertEqual(stderr.getvalue().count("\n"), 1)
                        self.assertIn(f"{path.resolve()} is not a git checkout", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
