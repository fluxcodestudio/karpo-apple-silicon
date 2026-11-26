# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""The global layers' attention answered a chunk of query rows at a time.

The first class needs torch only. The second compares with karpo-decision's own whole-table call.
``KARPO_TEST_NO_MPS=1`` leaves out the Apple GPU test.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import unittest
from itertools import pairwise
from unittest.mock import patch

import torch
from _support import attention_calls, needs_apple_gpu

import karpo_apple_silicon
from karpo_apple_silicon import global_attend
from karpo_apple_silicon.reference import dense_global, wrong_global_last_key_hidden

try:
    from karpo_decision.families.gemma import packed
except ImportError:
    packed = None


def reference(query, key, value, q_bounds, k_bounds, causal, scale):
    """One softmax per query and head, in float64, with explicit loops. A query reads every key of its
    sequence, or, when causal, the keys up to its own, the last query sitting on the last key."""
    heads, kv_heads = query.shape[2], key.shape[2]
    group = heads // kv_heads
    rows = []
    for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:], strict=True):
        lq, lk = qb - qa, kb - ka
        for i in range(lq):
            per_head = []
            for h in range(heads):
                kh = h // group
                js = [j for j in range(lk) if not causal or j <= i + lk - lq]
                logits = torch.stack([(query[0, qa + i, h].double() * key[0, ka + j, kh].double()).sum() * scale
                                      for j in js])
                weights = torch.softmax(logits, 0)
                per_head.append(sum(w * value[0, ka + j, kh].double() for w, j in zip(weights, js, strict=True)))
            rows.append(torch.cat(per_head))
    return torch.stack(rows)


def tensors(queries: int, keys: int, heads: int = 4, kv_heads: int = 4, size: int = 6, dtype=torch.float64):
    torch.manual_seed(queries * 1000 + keys * 10 + heads + kv_heads)
    return (torch.randn(1, queries, heads, size, dtype=dtype), torch.randn(1, keys, kv_heads, size, dtype=dtype),
            torch.randn(1, keys, kv_heads, size, dtype=dtype))


class GlobalAttend(unittest.TestCase):
    def check(self, q, k, v, q_bounds, k_bounds, causal, scale=0.4, chunks=(1, 3, 4, 256)):
        want = reference(q, k, v, q_bounds, k_bounds, causal, scale)
        for chunk in chunks:
            with self.subTest(chunk=chunk, causal=causal), patch.object(karpo_apple_silicon, "CHUNK", chunk):
                got = global_attend(q, k, v, q_bounds, k_bounds, None, causal, scale)
                self.assertEqual(got.shape, want.shape)
                torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)
        return want

    def test_one_sequence_reads_every_key(self):
        q, k, v = tensors(9, 9)
        self.check(q, k, v, [0, 9], [0, 9], causal=False)

    def test_causal_reads_up_to_its_own_key(self):
        q, k, v = tensors(9, 9)
        self.check(q, k, v, [0, 9], [0, 9], causal=True)

    def test_causal_and_bidirectional_differ(self):
        q, k, v = tensors(9, 9)
        both = [global_attend(q, k, v, [0, 9], [0, 9], None, causal, 0.4) for causal in (False, True)]
        self.assertGreater(float((both[0] - both[1]).abs().max()), 1e-3)

    def test_grouped_heads(self):
        q, k, v = tensors(7, 7, heads=4, kv_heads=2)
        self.check(q, k, v, [0, 7], [0, 7], causal=False)
        self.check(q, k, v, [0, 7], [0, 7], causal=True)

    def test_fewer_queries_than_keys(self):
        # The readout rows of a sequence over all of its keys, and a causal branch after its shared keys.
        q, k, v = tensors(5, 11)
        self.check(q, k, v, [0, 2, 5], [0, 4, 11], causal=False)
        self.check(q, k, v, [0, 2, 5], [0, 4, 11], causal=True)

    def test_more_queries_than_keys_is_answered_when_not_causal(self):
        """Every query reads every key of its sequence, however few there are: only a causal sequence can leave
        a query without a key."""
        q, k, v = tensors(12, 9, heads=4, kv_heads=2)
        q_bounds, k_bounds = [0, 9, 12], [0, 4, 9]  # nine queries over four keys, then three over five
        self.check(q, k, v, q_bounds, k_bounds, causal=False)
        torch.testing.assert_close(global_attend(q, k, v, q_bounds, k_bounds, None, False, 0.4),
                                   dense_global(q, k, v, q_bounds, k_bounds, None, False, 0.4), atol=1e-12, rtol=1e-12)

    def test_every_call_answers_a_chunk_of_query_rows(self):
        """What the replacement is for: no call gets more than ``CHUNK`` query rows, so no score table is heads
        x queries x keys of a whole sequence. Not causal, a call reads its sequence's keys, all of them; causal,
        none past the one its last row sits on. The results cannot show this: they are the same whatever
        ``CHUNK`` is."""
        chunk = 4
        q, k, v = tensors(30, 41)
        q_bounds, k_bounds = [0, 19, 30], [0, 23, 41]  # both several chunks long, with more keys than queries
        for causal in (False, True):
            with patch.object(karpo_apple_silicon, "CHUNK", chunk), attention_calls() as calls:
                global_attend(q, k, v, q_bounds, k_bounds, None, causal, 0.4)
            remaining = iter(calls)
            for (qa, qb), (ka, kb) in zip(pairwise(q_bounds), pairwise(k_bounds), strict=True):
                lq, lk, answered = qb - qa, kb - ka, 0
                while answered < lq:
                    rows, keys = next(remaining)
                    answered += rows
                    with self.subTest(causal=causal, sequence=(qa, qb), rows_answered=answered):
                        self.assertLessEqual(rows, chunk)
                        if causal:
                            self.assertLessEqual(keys, answered + lk - lq)
                        else:
                            self.assertEqual(keys, lk)
                self.assertEqual(answered, lq, "a call took rows of two sequences")
            self.assertEqual(list(remaining), [])

    def test_several_sequences_and_an_empty_one(self):
        q, k, v = tensors(12, 15, heads=4, kv_heads=2)
        for causal in (False, True):
            want = self.check(q, k, v, [0, 5, 5, 12], [0, 6, 8, 15], causal=causal)
            self.assertEqual(want.shape, (12, 4 * 6))

    def test_last_chunk_of_one_row_and_an_exact_multiple(self):
        q, k, v = tensors(8, 8)
        self.check(q, k, v, [0, 8], [0, 8], causal=False, chunks=(7, 4, 8))      # 7 + 1, 4 + 4, 8
        self.check(q, k, v, [0, 8], [0, 8], causal=True, chunks=(7, 4, 8))

    def test_sequences_do_not_leak_into_each_other(self):
        q, k, v = tensors(10, 10)
        whole = global_attend(q, k, v, [0, 4, 10], [0, 4, 10], None, False, 0.4)
        changed = v.clone()
        changed[0, 4:] += 5.0                                                   # only the second sequence's values
        again = global_attend(q, k, changed, [0, 4, 10], [0, 4, 10], None, False, 0.4)
        torch.testing.assert_close(again[:4], whole[:4])
        self.assertGreater(float((again[4:] - whole[4:]).abs().max()), 1e-3)

    def test_matches_the_whole_table_reference(self):
        q, k, v = tensors(300, 310, heads=4, kv_heads=2, size=8)
        for causal in (False, True):
            with self.subTest(causal=causal):
                got = global_attend(q, k, v, [0, 120, 300], [0, 125, 310], None, causal, 0.3)
                want = dense_global(q, k, v, [0, 120, 300], [0, 125, 310], None, causal, 0.3)
                torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)

    def test_wrong_reference_is_detected(self):
        q, k, v = tensors(6, 6)
        right = dense_global(q, k, v, [0, 6], [0, 6], None, False, 0.4)
        wrong = wrong_global_last_key_hidden(q, k, v, [0, 6], [0, 6], None, False, 0.4)
        self.assertGreater(float((right - wrong).abs().max()), 1e-3)

    def test_half_precision_stays_close(self):
        q, k, v = tensors(40, 40, dtype=torch.float32)
        want = reference(q, k, v, [0, 40], [0, 40], False, 0.3)
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), patch.object(karpo_apple_silicon, "CHUNK", 16):
                got = global_attend(q.to(dtype), k.to(dtype), v.to(dtype), [0, 40], [0, 40], None, False, 0.3)
                self.assertEqual(got.dtype, dtype)
                self.assertLess(float((got.double() - want).abs().max()), 0.06)

    def test_nothing_to_answer_is_an_empty_result(self):
        q, k, v = tensors(4, 4)
        got = global_attend(q[:, :0], k, v, [0, 0], [0, 4], None, False, 0.4)
        self.assertEqual(got.shape, (0, 4 * 6))

    def test_errors(self):
        q, k, v = tensors(6, 6)
        for function in (global_attend, dense_global):  # the oracle refuses them in the same words
            with self.subTest(function=function.__name__):
                with self.assertRaisesRegex(ValueError, "1 query sequences but 2 key sequences"):
                    function(q, k, v, [0, 6], [0, 3, 6], None, False, 0.4)
                with self.assertRaisesRegex(ValueError, "4 query heads are not a multiple of 3 key/value heads"):
                    function(q, k[:, :, :3], v[:, :, :3], [0, 6], [0, 6], None, False, 0.4)
        with self.assertRaisesRegex(ValueError, "without any key"):
            global_attend(q, k, v, [0, 6], [0, 4], None, True, 0.4)

    @needs_apple_gpu
    def test_apple_gpu_matches_cpu(self):
        q, k, v = tensors(300, 310, heads=4, kv_heads=2, size=8, dtype=torch.float32)
        for causal in (False, True):
            cpu = global_attend(q, k, v, [0, 120, 300], [0, 125, 310], None, causal, 0.3)
            for dtype, tolerance in ((torch.float32, 1e-4), (torch.float16, 0.05), (torch.bfloat16, 0.05)):
                with self.subTest(causal=causal, dtype=dtype):
                    got = global_attend(*(t.to("mps", dtype) for t in (q, k, v)), [0, 120, 300], [0, 125, 310],
                                        None, causal, 0.3)
                    self.assertLess(float((got.cpu().float() - cpu).abs().max()), tolerance)


@unittest.skipUnless(packed is not None, "karpo-decision is required")
class AgainstUpstream(unittest.TestCase):
    """karpo-decision's own function, with its CUDA-only kernel selection lifted for the length of a call."""

    def upstream(self, *arguments):
        with patch.object(packed, "sdpa_kernel", lambda *_: contextlib.nullcontext()):
            return packed.Packed.__dict__["_global"].__func__(*arguments)

    def test_same_result_as_one_call_per_sequence(self):
        cases = (((9, 9), [0, 9], [0, 9]), ((5, 11), [0, 2, 5], [0, 4, 11]), ((12, 15), [0, 5, 5, 12], [0, 6, 8, 15]))
        for (queries, keys), q_bounds, k_bounds in cases:
            for heads, kv_heads in ((4, 4), (4, 2)):
                q, k, v = tensors(queries, keys, heads=heads, kv_heads=kv_heads)
                lens = torch.zeros(len(q_bounds) - 1, 2, 1, dtype=torch.int32)
                for causal in (False, True):
                    want = self.upstream(q, k, v, q_bounds, k_bounds, lens, causal, 0.4)
                    for chunk in (1, 3, 256):
                        with self.subTest(queries=queries, heads=heads, kv_heads=kv_heads, causal=causal,
                                          chunk=chunk), patch.object(karpo_apple_silicon, "CHUNK", chunk):
                            got = global_attend(q, k, v, q_bounds, k_bounds, lens, causal, 0.4)
                            torch.testing.assert_close(got, want, atol=1e-12, rtol=1e-12)

    def test_signature_is_the_one_replaced(self):
        import inspect

        self.assertEqual(tuple(inspect.signature(packed.Packed._global).parameters),
                         tuple(inspect.signature(global_attend).parameters))


if __name__ == "__main__":
    unittest.main()
