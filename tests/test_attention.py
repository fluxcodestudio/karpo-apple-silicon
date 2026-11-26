# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests that need torch only (no weights, no GPU). The last class also needs karpo-decision.
``KARPO_TEST_NO_MPS=1`` leaves out the Apple GPU class.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import builtins
import importlib.util
import unittest
import warnings
from itertools import pairwise
from unittest.mock import patch

import torch
from _support import attention_calls, needs_apple_gpu

import karpo_apple_silicon
from karpo_apple_silicon import attend, install, window_mask
from karpo_apple_silicon.reference import (
    dense_attend,
    wrong_no_window,
    wrong_top_left,
    wrong_window_minus_one,
    wrong_window_plus_one,
)


def allowed(i: int, j: int, lq: int, lk: int, window: int) -> bool:
    """The rule, stated the slow way."""
    d = lk - lq
    return i + d - window <= j <= i + d


def reference(q, k, v, q_bounds, k_bounds, scale, window):
    """One softmax per query and head, in float64, with explicit loops."""
    heads, kv_heads = q.shape[1], k.shape[1]
    group = heads // kv_heads
    out = torch.zeros(q.shape, dtype=torch.float64)
    for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:], strict=True):
        lq, lk = qb - qa, kb - ka
        for h in range(heads):
            kh = h // group
            for i in range(lq):
                js = [j for j in range(lk) if allowed(i, j, lq, lk, window)]
                logits = torch.stack([(q[qa + i, h].double() * k[ka + j, kh].double()).sum() * scale for j in js])
                weights = torch.softmax(logits, 0)
                out[qa + i, h] = sum(w * v[ka + j, kh].double() for w, j in zip(weights, js, strict=True))
    return out


class WindowMask(unittest.TestCase):
    def test_matches_brute_force(self):
        for lq, lk, window in ((5, 5, 2), (3, 7, 2), (4, 9, 100), (1, 6, 0), (6, 6, 0), (1, 1, 5), (7, 12, 3),
                               (30, 30, 1023), (10, 40, 9)):
            mask = window_mask(lq, lk, window)
            self.assertEqual(mask.shape, (lq, lk))
            for i in range(lq):
                for j in range(lk):
                    self.assertEqual(bool(mask[i, j]), allowed(i, j, lq, lk, window), (lq, lk, window, i, j))

    def test_every_query_has_a_key_and_at_most_window_plus_one(self):
        for lq, lk, window in ((5, 5, 2), (3, 7, 2), (6, 6, 0), (9, 20, 4)):
            counts = window_mask(lq, lk, window).sum(1)
            self.assertTrue((counts >= 1).all())
            self.assertTrue((counts <= window + 1).all())

    def test_last_query_reads_last_key_and_nothing_later(self):
        mask = window_mask(4, 9, 3)
        self.assertTrue(mask[-1, -1])      # aligned at the end
        self.assertTrue(mask[0, 5])        # first query sits on key d = 5
        self.assertFalse(mask[0, 6])       # and never reads ahead
        self.assertFalse(mask[0, 1])       # nor further back than the window (5 - 3 = 2)
        self.assertTrue(mask[0, 2])

    def test_square_and_wide_window_is_plain_causal(self):
        self.assertTrue(torch.equal(window_mask(6, 6, 1000), torch.ones(6, 6).tril().bool()))


class Attend(unittest.TestCase):
    def check(self, heads, kv_heads, dim, q_bounds, k_bounds, window, scale, seed=0):
        generator = torch.Generator().manual_seed(seed)
        q = torch.randn(q_bounds[-1], heads, dim, generator=generator)
        k = torch.randn(k_bounds[-1], kv_heads, dim, generator=generator)
        v = torch.randn(k_bounds[-1], kv_heads, dim, generator=generator)
        got = attend(q, k, v, torch.tensor(q_bounds, dtype=torch.int32), torch.tensor(k_bounds, dtype=torch.int32),
                     max(b - a for a, b in pairwise(q_bounds)),
                     max(b - a for a, b in pairwise(k_bounds)), scale, window)
        want = reference(q, k, v, q_bounds, k_bounds, scale, window)
        self.assertEqual(got.shape, q.shape)
        self.assertTrue(torch.allclose(got.double(), want, atol=1e-5), float((got.double() - want).abs().max()))

    def test_one_sequence_same_lengths(self):
        self.check(heads=2, kv_heads=2, dim=8, q_bounds=[0, 6], k_bounds=[0, 6], window=2, scale=1.0)

    def test_grouped_heads(self):
        self.check(heads=4, kv_heads=2, dim=8, q_bounds=[0, 5], k_bounds=[0, 5], window=3, scale=0.5)
        self.check(heads=8, kv_heads=2, dim=4, q_bounds=[0, 5], k_bounds=[0, 9], window=3, scale=0.25, seed=1)
        self.check(heads=6, kv_heads=1, dim=4, q_bounds=[0, 4], k_bounds=[0, 4], window=1, scale=1.0, seed=2)

    def test_packed_sequences_of_unequal_length(self):
        # four sequences: equal lengths, more keys than queries (a branch), a single row, a long one
        self.check(heads=4, kv_heads=2, dim=8, q_bounds=[0, 3, 7, 8, 19], k_bounds=[0, 3, 12, 15, 30], window=2,
                   scale=0.5, seed=3)

    def test_window_wider_than_every_sequence(self):
        self.check(heads=2, kv_heads=1, dim=8, q_bounds=[0, 4, 9], k_bounds=[0, 6, 11], window=1023, scale=1.0, seed=4)

    def test_window_zero_reads_only_the_aligned_key(self):
        self.check(heads=2, kv_heads=2, dim=4, q_bounds=[0, 3], k_bounds=[0, 5], window=0, scale=1.0, seed=5)

    def test_chunk_boundaries(self):
        for chunk in (1, 2, 3, 5, 7):
            with self.subTest(chunk=chunk), patch.object(karpo_apple_silicon, "CHUNK", chunk):
                self.check(2, 2, 8, [0, 6], [0, 6], 2, 1.0)
                self.check(4, 2, 8, [0, 3, 7, 8, 19], [0, 3, 12, 15, 30], 2, 0.5, seed=3)
                self.check(2, 1, 8, [0, 4, 9], [0, 6, 11], 1023, 1.0, seed=4)
                self.check(2, 2, 4, [0, 3], [0, 5], 0, 1.0, seed=5)

    def test_every_call_answers_a_chunk_over_the_keys_its_window_reaches(self):
        """What the port is for: however long a sequence is, no call gets more than ``CHUNK`` queries or more
        than ``CHUNK + window`` keys, so no score table grows with the square of the length. The results cannot
        show this: they are the same whatever ``CHUNK`` is."""
        chunk, window = 5, 3
        q_lengths, k_lengths = [31, 12, 2], [31, 40, 2]  # the first two are several chunks and windows long
        generator = torch.Generator().manual_seed(9)
        q = torch.randn(sum(q_lengths), 4, 8, generator=generator)
        k = torch.randn(sum(k_lengths), 2, 8, generator=generator)
        q_cu, k_cu = (torch.tensor([0, *lengths]).cumsum(0) for lengths in (q_lengths, k_lengths))
        with patch.object(karpo_apple_silicon, "CHUNK", chunk), attention_calls() as calls:
            attend(q, k, k, q_cu, k_cu, max(q_lengths), max(k_lengths), 1.0, window)
        self.assertEqual(sum(rows for rows, _ in calls), sum(q_lengths))  # every query answered, and once
        self.assertLessEqual(max(rows for rows, _ in calls), chunk)
        self.assertLessEqual(max(keys for _, keys in calls), chunk + window)

    def test_empty_sequence_is_skipped(self):
        self.check(heads=2, kv_heads=2, dim=4, q_bounds=[0, 3, 3, 6], k_bounds=[0, 3, 5, 8],
                   window=2, scale=1.0, seed=6)

    def test_nothing_to_answer_is_an_empty_tensor_like_the_queries(self):
        """No query rows at all, in sequences without queries or in no sequence: karpo-decision goes on to
        reshape what comes back, so it must be a tensor of the queries' shape and type."""
        for dtype in (torch.float32, torch.bfloat16):
            q, k = torch.zeros(0, 4, 8, dtype=dtype), torch.ones(5, 2, 8, dtype=dtype)
            for q_bounds, k_bounds in (([0, 0, 0], [0, 3, 5]), ([0], [0])):
                with self.subTest(dtype=dtype, q_bounds=q_bounds):
                    got = attend(q, k, k, torch.tensor(q_bounds), torch.tensor(k_bounds), 0, 3, 1.0, 2)
                    self.assertIsInstance(got, torch.Tensor)
                    self.assertEqual((got.shape, got.dtype), (torch.Size([0, 4, 8]), dtype))

    def test_sequences_do_not_leak_into_each_other(self):
        generator = torch.Generator().manual_seed(7)
        q, k, v = (torch.randn(8, 2, 4, generator=generator) for _ in range(3))
        cu = torch.tensor([0, 3, 8], dtype=torch.int32)
        before = attend(q, k, v, cu, cu, 5, 5, 1.0, 10)
        k2, v2 = k.clone(), v.clone()
        k2[3:], v2[3:] = 0.0, 100.0  # change the second sequence only
        after = attend(q, k2, v2, cu, cu, 5, 5, 1.0, 10)
        self.assertTrue(torch.equal(before[:3], after[:3]))
        self.assertFalse(torch.allclose(before[3:], after[3:]))

    def test_half_precision_stays_close(self):
        generator = torch.Generator().manual_seed(8)
        q, k, v = (torch.randn(9, 4, 16, generator=generator) for _ in range(3))
        cu = torch.tensor([0, 4, 9], dtype=torch.int32)
        full = attend(q, k, v, cu, cu, 5, 5, 0.25, 2)
        for dtype in (torch.bfloat16, torch.float16):
            half = attend(q.to(dtype), k.to(dtype), v.to(dtype), cu, cu, 5, 5, 0.25, 2)
            self.assertEqual(half.dtype, dtype)
            self.assertLess(float((half.float() - full).abs().max()), 0.05)

    def test_query_without_any_key_is_an_error(self):
        q, k = torch.randn(5, 2, 4), torch.randn(3, 2, 4)
        with self.assertRaisesRegex(ValueError, "without any key"):
            attend(q, k, k, torch.tensor([0, 5]), torch.tensor([0, 3]), 5, 3, 1.0, 2)

    def test_mismatched_boundaries_and_heads_are_errors(self):
        for function in (attend, dense_attend):  # the oracle refuses them in the same words
            with self.subTest(function=function.__name__):
                q, k = torch.randn(4, 3, 4), torch.randn(4, 2, 4)
                with self.assertRaisesRegex(ValueError, "3 query heads are not a multiple of 2 key/value heads"):
                    function(q, k, k, torch.tensor([0, 4]), torch.tensor([0, 4]), 4, 4, 1.0, 2)
                q = torch.randn(4, 2, 4)
                with self.assertRaisesRegex(ValueError, "2 query sequences but 1 key sequences"):
                    function(q, k, k, torch.tensor([0, 2, 4]), torch.tensor([0, 4]), 4, 4, 1.0, 2)


class DenseReference(unittest.TestCase):
    def test_small_cases_match_brute_force(self):
        cases = ((2, 2, 8, [0, 6], [0, 6], 2, 1.0),
                 (4, 2, 8, [0, 5], [0, 5], 3, 0.5),
                 (8, 2, 4, [0, 5], [0, 9], 3, 0.25),
                 (6, 1, 4, [0, 4], [0, 4], 1, 1.0),
                 (4, 2, 8, [0, 3, 7, 8, 19], [0, 3, 12, 15, 30], 2, 0.5),
                 (2, 1, 8, [0, 4, 9], [0, 6, 11], 1023, 1.0),
                 (2, 2, 4, [0, 3], [0, 5], 0, 1.0),
                 (2, 2, 4, [0, 3, 3, 6], [0, 3, 5, 8], 2, 1.0))
        for heads, kv_heads, dim, q_bounds, k_bounds, window, scale in cases:
            with self.subTest(q_bounds=q_bounds, k_bounds=k_bounds, window=window):
                generator = torch.Generator().manual_seed(11)
                q = torch.randn(q_bounds[-1], heads, dim, generator=generator)
                k = torch.randn(k_bounds[-1], kv_heads, dim, generator=generator)
                v = torch.randn(k_bounds[-1], kv_heads, dim, generator=generator)
                q_cu, k_cu = torch.tensor(q_bounds), torch.tensor(k_bounds)
                got = dense_attend(q, k, v, q_cu, k_cu, max(q_bounds), max(k_bounds), scale, window)
                want = reference(q, k, v, q_bounds, k_bounds, scale, window)
                self.assertTrue(torch.allclose(got.double(), want, atol=1e-5))

    def test_wrong_mask_is_detected(self):
        """Each wrong-on-purpose mask gives another result than the port. The sequence has more keys than
        queries: with as many of each, the diagonal aligned at the start is the right one."""
        generator = torch.Generator().manual_seed(12)
        q = torch.randn(6, 2, 4, generator=generator)
        k, v = (torch.randn(10, 2, 4, generator=generator) for _ in range(2))
        q_cu, k_cu = torch.tensor([0, 6]), torch.tensor([0, 10])
        correct = attend(q, k, v, q_cu, k_cu, 6, 10, 0.5, 2)
        torch.testing.assert_close(dense_attend(q, k, v, q_cu, k_cu, 6, 10, 0.5, 2), correct)
        for mask in (wrong_no_window, wrong_top_left, wrong_window_minus_one, wrong_window_plus_one):
            with self.subTest(mask=mask.__name__):
                wrong = dense_attend(q, k, v, q_cu, k_cu, 6, 10, 0.5, 2, mask=mask)
                self.assertGreater(float((correct - wrong).abs().max()), 0.01)

    def test_real_chunk_and_large_windows(self):
        for q_lengths, k_lengths, window, heads, kv_heads in (
            ([700, 40, 300], [700, 139, 300], 99, 4, 2),
            ([1300], [1300], 1023, 2, 2),
        ):
            with self.subTest(q_lengths=q_lengths, window=window):
                generator = torch.Generator().manual_seed(window)
                q = torch.randn(sum(q_lengths), heads, 8, generator=generator, dtype=torch.float64)
                k = torch.randn(sum(k_lengths), kv_heads, 8, generator=generator, dtype=torch.float64)
                v = torch.randn(sum(k_lengths), kv_heads, 8, generator=generator, dtype=torch.float64)
                q_cu = torch.tensor([0, *q_lengths]).cumsum(0)
                k_cu = torch.tensor([0, *k_lengths]).cumsum(0)
                args = (q, k, v, q_cu, k_cu, max(q_lengths), max(k_lengths), 0.5, window)
                self.assertLess(float((attend(*args) - dense_attend(*args)).abs().max()), 1e-12)


@needs_apple_gpu
class AgainstMPS(unittest.TestCase):
    def test_device_and_precision(self):
        generator = torch.Generator().manual_seed(13)
        q = torch.randn(270, 4, 8, generator=generator)
        k = torch.randn(285, 2, 8, generator=generator)
        v = torch.randn(285, 2, 8, generator=generator)
        q_cu, k_cu = torch.tensor([0, 260, 270]), torch.tensor([0, 275, 285])
        cpu = attend(q, k, v, q_cu, k_cu, 260, 275, 0.5, 31)
        for dtype, tolerance in ((torch.float32, 1e-4), (torch.float16, 0.05), (torch.bfloat16, 0.05)):
            with self.subTest(dtype=dtype):
                got = attend(q.to(device="mps", dtype=dtype), k.to(device="mps", dtype=dtype),
                             v.to(device="mps", dtype=dtype), q_cu, k_cu, 260, 275, 0.5, 31)
                self.assertEqual(got.dtype, dtype)
                self.assertLess(float((got.cpu().float() - cpu).abs().max()), tolerance)


class InstallWithoutUpstream(unittest.TestCase):
    """``install()`` where importing karpo-decision fails. The import is made to fail here, so this runs with
    and without karpo-decision installed."""

    def install_with_import_failing(self, error: ModuleNotFoundError) -> ModuleNotFoundError:
        """What ``install()`` raises when importing anything of karpo-decision raises ``error``."""
        real_import = builtins.__import__

        def failing(name, *arguments, **options):
            if name.split(".")[0] == "karpo_decision":
                raise error
            return real_import(name, *arguments, **options)

        self.addCleanup(karpo_apple_silicon.uninstall)
        with patch.object(builtins, "__import__", failing), self.assertRaises(ModuleNotFoundError) as raised:
            install()
        self.assertFalse(karpo_apple_silicon._originals)
        return raised.exception

    def test_a_missing_karpo_decision_is_answered_with_its_install_line(self):
        for name in ("karpo_decision", "karpo_decision.families.gemma"):
            with self.subTest(missing=name):
                error = ModuleNotFoundError(f"No module named {name!r}", name=name)
                raised = self.install_with_import_failing(error)
                self.assertIn("karpo-decision is installed separately: pip install \"karpo-decision @ git+https://"
                              "github.com/machinepulse-ai/karpo-decision@"
                              f"{karpo_apple_silicon.TESTED_UPSTREAM_COMMIT}\"", str(raised))
                self.assertIs(raised.__cause__, error)

    def test_another_missing_module_is_raised_as_it_is(self):
        """karpo-decision is there and something it imports is not: the install line would be the wrong advice."""
        error = ModuleNotFoundError("No module named 'transformers'", name="transformers")
        self.assertIs(self.install_with_import_failing(error), error)


@unittest.skipUnless(importlib.util.find_spec("karpo_decision"), "karpo-decision is not installed")
class AgainstUpstreamPlan(unittest.TestCase):
    """The patch fed by karpo-decision's own ``block_plan``, checked against the rule as upstream words it, by
    token position: a token reads tokens of its own sequence that are in the shared block or in its own branch,
    at positions from its own back to ``sliding_window - 1`` before it."""

    WINDOW = 5  # the model's ``sliding_window``; karpo-decision passes ``WINDOW - 1`` to attend

    @classmethod
    def setUpClass(cls):
        from karpo_decision.families.gemma import packed, sliding_attention

        # In a tuple: a staticmethod kept as an attribute of this class would come back as its bare function.
        cls.found = (sliding_attention.attend, packed.Packed.__dict__["_global"])

    @classmethod
    def tearDownClass(cls):
        """Every test here that installs also uninstalls: karpo-decision is left as it was found."""
        from karpo_decision.families.gemma import packed, sliding_attention

        attend_before, global_before = cls.found
        if sliding_attention.attend is not attend_before:
            raise AssertionError("these tests left karpo-decision's sliding_attention.attend replaced")
        if packed.Packed.__dict__["_global"] is not global_before:
            raise AssertionError("these tests left karpo-decision's Packed._global replaced")

    def setUp(self):
        self.addCleanup(karpo_apple_silicon.uninstall)

    def pack(self, layouts):
        """``layouts``: per sequence, [shared length, branch length, ...] (no branches: one block)."""
        from karpo_decision.model.packed import Pack

        segments, positions, markers, counts, lengths = [], [], [], [], []
        row = 0
        for shared, *branches in layouts:
            segments += [0] * shared
            positions += list(range(shared))
            row += shared
            if not branches:
                markers.append(row - 1)
            for number, length in enumerate(branches, start=1):
                segments += [number] * length
                positions += list(range(shared, shared + length))  # a branch continues after the shared block
                row += length
                markers.append(row - 1)
            counts.append(max(1, len(branches)))
            lengths.append(shared + sum(branches))
        cu = torch.tensor([0, *lengths]).cumsum(0)
        segment_ids = torch.tensor(segments, dtype=torch.int32)
        return Pack(input_ids=torch.zeros(row, dtype=torch.long), seqpos=torch.tensor(positions),
                    cu=cu.to(torch.int32), cu_cpu=cu, max_length=max(lengths), markers=torch.tensor(markers),
                    marker_cu=torch.tensor([0, *counts]).cumsum(0).to(torch.int32), max_markers=max(counts),
                    segment_ids=segment_ids, segment_ids_cpu=segment_ids)

    def test_plan_and_patch_follow_the_position_rule(self):
        from karpo_decision.families.gemma.packed import block_plan

        heads, kv_heads, dim, scale = 4, 2, 8, 1.0
        for layouts in ([[7]], [[3], [12]], [[9, 3, 4]], [[2, 1, 6], [11], [8, 2, 2, 5]]):
            pack = self.pack(layouts)
            plan = block_plan(pack, self.WINDOW)
            generator = torch.Generator().manual_seed(len(layouts))
            q = torch.randn(pack.tokens, heads, dim, generator=generator)
            k = torch.randn(pack.tokens, kv_heads, dim, generator=generator)
            v = torch.randn(pack.tokens, kv_heads, dim, generator=generator)
            keys, values = (k, v) if plan.s_rows is None else (k.index_select(0, plan.s_rows),
                                                               v.index_select(0, plan.s_rows))
            got = attend(q, keys, values, plan.q_cu_t, plan.s_cu_t, plan.max_q, plan.max_s, scale, self.WINDOW - 1)

            sequence = torch.bucketize(torch.arange(pack.tokens), pack.cu_cpu[1:], right=True)
            for row in range(pack.tokens):
                readable = [c for c in range(pack.tokens)
                            if sequence[c] == sequence[row]
                            and int(pack.segment_ids_cpu[c]) in (0, int(pack.segment_ids_cpu[row]))
                            and 0 <= int(pack.seqpos[row]) - int(pack.seqpos[c]) <= self.WINDOW - 1]
                for h in range(heads):
                    kh = h // (heads // kv_heads)
                    logits = torch.stack([(q[row, h] * k[c, kh]).sum() * scale for c in readable]).double()
                    want = sum(w * v[c, kh].double() for w, c in zip(torch.softmax(logits, 0), readable, strict=True))
                    self.assertTrue(torch.allclose(got[row, h].double(), want, atol=1e-5), (layouts, row, h))

    def test_install_swaps_both_without_cuda(self):
        from karpo_decision.families.gemma import packed, sliding_attention

        original_kernel = packed.sdpa_kernel
        with patch.object(torch.cuda, "is_available", return_value=False):
            install()
        self.assertIs(sliding_attention.attend, attend)
        self.assertIs(packed.Packed._global, karpo_apple_silicon.global_attend)
        self.assertIs(packed.sdpa_kernel, original_kernel)   # the replaced global layers never reach it

    def test_forced_install_with_cuda_leaves_the_global_layers(self):
        from karpo_decision.families.gemma import packed, sliding_attention

        original_global = packed.Packed.__dict__["_global"]
        with patch.object(torch.cuda, "is_available", return_value=True):
            install()
            self.assertIsNot(sliding_attention.attend, attend)
            install(force=True)
        self.assertIs(sliding_attention.attend, attend)
        self.assertIs(packed.Packed.__dict__["_global"], original_global)

    def test_wrong_global_signature_is_rejected_before_anything_is_swapped(self):
        from karpo_decision.families.gemma import packed, sliding_attention

        original_attend = sliding_attention.attend
        with patch.object(packed.Packed, "_global", staticmethod(lambda query: query)), \
             patch.object(torch.cuda, "is_available", return_value=False), \
             self.assertRaisesRegex(RuntimeError, "not supported"):
            install()
        self.assertIs(sliding_attention.attend, original_attend)
        self.assertFalse(karpo_apple_silicon._originals)

    def test_commit_warning_and_matching_commit(self):
        checked = karpo_apple_silicon.TESTED_UPSTREAM_COMMITS
        self.assertEqual(checked[0], karpo_apple_silicon.TESTED_UPSTREAM_COMMIT)  # the one the instructions name
        for commit, expected in (("abcdef0123456789", True), *((commit, False) for commit in checked)):
            with self.subTest(commit=commit), \
                 patch.object(karpo_apple_silicon, "_installed_commit", return_value=commit):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    install(force=True)
                self.assertEqual(bool(caught), expected)
                if expected:
                    self.assertEqual(caught[0].filename, __file__)  # the line that called install(), not its own
                    self.assertIn(commit[:12], str(caught[0].message))
                    self.assertIn(karpo_apple_silicon.TESTED_UPSTREAM_COMMIT[:12], str(caught[0].message))

    def test_version_warning_without_commit(self):
        with patch.object(karpo_apple_silicon, "_installed_commit", return_value=None), \
             patch.object(karpo_apple_silicon, "version", return_value="other"), \
             self.assertWarnsRegex(UserWarning, "other is installed") as warned:
            install(force=True)
        self.assertEqual(warned.filename, __file__)  # the line that called install(), not its own

    def test_matching_commit_overrides_different_version(self):
        with patch.object(karpo_apple_silicon, "_installed_commit",
                          return_value=karpo_apple_silicon.TESTED_UPSTREAM_COMMIT), \
             patch.object(karpo_apple_silicon, "version", return_value="other"), \
             warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            install(force=True)
        self.assertFalse(caught)

    def test_wrong_upstream_signature_is_rejected(self):
        from karpo_decision.families.gemma import sliding_attention

        with patch.object(sliding_attention, "attend", lambda query: query), \
             self.assertRaisesRegex(RuntimeError, "not supported"):
            install(force=True)


if __name__ == "__main__":
    unittest.main()
