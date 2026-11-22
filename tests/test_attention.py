# Copyright 2026 the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Tests that need torch only (no weights, no GPU). The last class also needs karpo-decision.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib.util
import unittest
import warnings

import torch

from karpo_apple_silicon import attend, install, window_mask


def allowed(i: int, j: int, lq: int, lk: int, window: int) -> bool:
    """The rule, stated the slow way."""
    d = lk - lq
    return i + d - window <= j <= i + d


def reference(q, k, v, q_bounds, k_bounds, scale, window):
    """One softmax per query and head, in float64, with explicit loops."""
    heads, kv_heads = q.shape[1], k.shape[1]
    group = heads // kv_heads
    out = torch.zeros(q.shape, dtype=torch.float64)
    for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]):
        lq, lk = qb - qa, kb - ka
        for h in range(heads):
            kh = h // group
            for i in range(lq):
                js = [j for j in range(lk) if allowed(i, j, lq, lk, window)]
                logits = torch.stack([(q[qa + i, h].double() * k[ka + j, kh].double()).sum() * scale for j in js])
                weights = torch.softmax(logits, 0)
                out[qa + i, h] = sum(w * v[ka + j, kh].double() for w, j in zip(weights, js))
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
                     max(b - a for a, b in zip(q_bounds, q_bounds[1:])),
                     max(b - a for a, b in zip(k_bounds, k_bounds[1:])), scale, window)
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

    def test_empty_sequence_is_skipped(self):
        self.check(heads=2, kv_heads=2, dim=4, q_bounds=[0, 3, 3, 6], k_bounds=[0, 3, 5, 8], window=2, scale=1.0, seed=6)

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
        q, k = torch.randn(4, 3, 4), torch.randn(4, 2, 4)
        with self.assertRaisesRegex(ValueError, "not a multiple"):
            attend(q, k, k, torch.tensor([0, 4]), torch.tensor([0, 4]), 4, 4, 1.0, 2)
        q = torch.randn(4, 2, 4)
        with self.assertRaisesRegex(ValueError, "key sequences"):
            attend(q, k, k, torch.tensor([0, 2, 4]), torch.tensor([0, 4]), 4, 4, 1.0, 2)


@unittest.skipUnless(importlib.util.find_spec("karpo_decision"), "karpo-decision is not installed")
class AgainstUpstreamPlan(unittest.TestCase):
    """The patch fed by karpo-decision's own ``block_plan``, checked against the rule as upstream words it, by
    token position: a token reads tokens of its own sequence that are in the shared block or in its own branch,
    at positions from its own back to ``sliding_window - 1`` before it."""

    WINDOW = 5  # the model's ``sliding_window``; karpo-decision passes ``WINDOW - 1`` to attend

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
                    want = sum(w * v[c, kh].double() for w, c in zip(torch.softmax(logits, 0), readable))
                    self.assertTrue(torch.allclose(got[row, h].double(), want, atol=1e-5), (layouts, row, h))

    def test_install_swaps_the_function_and_only_that(self):
        from karpo_decision.families.gemma import sliding_attention

        original = sliding_attention.attend
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # the tested upstream version must not warn
                install(force=True)
            self.assertIs(sliding_attention.attend, attend)
        finally:
            sliding_attention.attend = original


if __name__ == "__main__":
    unittest.main()
