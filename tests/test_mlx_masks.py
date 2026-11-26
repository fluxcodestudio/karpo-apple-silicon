# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""The MLX path's masks, attention chunks and rotary frequencies. Needs mlx, mlx-lm and karpo-decision; no
weights. Skipped where MLX cannot start (no Metal device: sandboxes, some CI runners)."""

from __future__ import annotations

import unittest

import numpy as np
from _support import mlx_unavailable

SKIP = mlx_unavailable()


def per_row_masks(plan: dict, window: int) -> dict[str, np.ndarray]:
    """``attention_masks`` as first written, one query row at a time: the reference the vectorised one must equal."""
    query_bounds, global_bounds, sliding_bounds = plan["query_bounds"], plan["global_bounds"], plan["sliding_bounds"]
    marker_rows = plan["marker_rows"]
    length = query_bounds[-1]
    global_mask = np.zeros((length, length), dtype=bool)
    sliding_mask = np.zeros((length, length), dtype=bool)
    last_mask = np.zeros((len(marker_rows), length), dtype=bool)
    marker_lookup = {row: index for index, row in enumerate(marker_rows)}
    for block in range(len(query_bounds) - 1):
        q_start, q_end = query_bounds[block], query_bounds[block + 1]
        gathered_global = plan["global_rows"][global_bounds[block]:global_bounds[block + 1]]
        gathered_sliding = plan["sliding_rows"][sliding_bounds[block]:sliding_bounds[block + 1]]
        shift = len(gathered_sliding) - (q_end - q_start)
        for query in range(q_start, q_end):
            global_mask[query, gathered_global] = True
            aligned = query - q_start + shift
            low = max(0, aligned - window + 1)
            sliding_mask[query, gathered_sliding[low:aligned + 1]] = True
            if query in marker_lookup:
                last_mask[marker_lookup[query], gathered_global] = True
    return {"global": global_mask, "sliding": sliding_mask, "last": last_mask}


def upstream_plan(layout: list[tuple[int, list[int]]], window: int) -> dict:
    """karpo-decision's own plan for sequences of (shared rows, [rows of each branch]), as ``_pack_and_plan``
    reads it."""
    import torch
    from karpo_decision.encoding import Encoded

    from karpo_apple_silicon.mlx import _pack_and_plan

    sequences = []
    for shared, branches in layout:
        positions, segments, markers, at = list(range(shared)), [0] * shared, [], shared
        for number, rows in enumerate(branches, start=1):
            positions += range(shared, shared + rows)
            segments += [number] * rows
            at += rows
            markers.append(at - 1)
        sequences.append(Encoded(keys=tuple(str(i) for i in range(max(1, len(branches)))),
                                 counts=(2,) * max(1, len(branches)), input_ids=torch.zeros(at, dtype=torch.long),
                                 markers=torch.tensor(markers or [shared - 1]), positions=torch.tensor(positions),
                                 segment_ids=torch.tensor(segments, dtype=torch.int16)))
    return _pack_and_plan(sequences, window)[1]


@unittest.skipIf(SKIP, SKIP)
class Masks(unittest.TestCase):
    def test_one_sequence(self):
        from karpo_apple_silicon.mlx import attention_masks

        plan = {"query_bounds": [0, 4], "global_rows": [0, 1, 2, 3], "global_bounds": [0, 4],
                "sliding_rows": [0, 1, 2, 3], "sliding_bounds": [0, 4], "marker_rows": [3]}
        masks = attention_masks(plan, 3)
        self.assertEqual(masks["global"].tolist(), [[True] * 4] * 4)
        self.assertEqual(masks["sliding"].tolist(), [[True, False, False, False], [True, True, False, False],
                                                     [True, True, True, False], [False, True, True, True]])
        self.assertEqual(masks["last"].tolist(), [[True] * 4])

    def test_shared_block_and_two_branches(self):
        from karpo_apple_silicon.mlx import attention_masks

        # rows 0-2 shared, 3-4 branch one, 5-6 branch two; window 3: a branch's keys are the shared block's
        # last two rows, then its own
        plan = {"query_bounds": [0, 3, 5, 7], "global_rows": [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 5, 6],
                "global_bounds": [0, 3, 8, 13], "sliding_rows": [0, 1, 2, 1, 2, 3, 4, 1, 2, 5, 6],
                "sliding_bounds": [0, 3, 7, 11], "marker_rows": [4, 6]}
        masks = attention_masks(plan, 3)
        self.assertEqual(masks["global"][4].tolist(), [True, True, True, True, True, False, False])
        self.assertEqual(masks["global"][6].tolist(), [True, True, True, False, False, True, True])
        self.assertEqual(masks["global"][1].tolist(), [True, True, True, False, False, False, False])
        self.assertEqual(masks["sliding"][3].tolist(), [False, True, True, True, False, False, False])
        self.assertEqual(masks["sliding"][4].tolist(), [False, False, True, True, True, False, False])
        self.assertEqual(masks["sliding"][6].tolist(), [False, False, True, False, False, True, True])
        self.assertEqual(masks["last"].tolist(), [masks["global"][4].tolist(), masks["global"][6].tolist()])

    def test_sliding_mask_equals_the_torch_patch_on_upstream_plans(self):
        """Same rows readable as ``karpo_apple_silicon.window_mask`` over karpo-decision's own plan."""
        import torch
        from karpo_decision.families.gemma.packed import block_plan
        from karpo_decision.model.packed import Pack

        from karpo_apple_silicon import window_mask
        from karpo_apple_silicon.mlx import attention_masks

        window = 4
        segments = torch.tensor([0] * 9 + [1] * 3 + [2] * 5 + [0] * 6, dtype=torch.int32)
        positions = torch.tensor([*range(9), *range(9, 12), *range(9, 14), *range(6)])
        cu = torch.tensor([0, 17, 23])
        pack = Pack(input_ids=torch.zeros(23, dtype=torch.long), seqpos=positions, cu=cu.to(torch.int32), cu_cpu=cu,
                    max_length=17, markers=torch.tensor([11, 16, 22]),
                    marker_cu=torch.tensor([0, 2, 3], dtype=torch.int32), max_markers=2, segment_ids=segments,
                    segment_ids_cpu=segments)
        plan = block_plan(pack, window)
        as_dict = {"query_bounds": plan.q_cu, "global_rows": plan.g_rows.tolist(), "global_bounds": plan.g_cu,
                   "sliding_rows": plan.s_rows.tolist(), "sliding_bounds": plan.s_cu_t.tolist(),
                   "marker_rows": pack.markers.tolist()}
        got = attention_masks(as_dict, window)["sliding"]
        want = torch.zeros(23, 23, dtype=torch.bool)
        s_cu, rows = plan.s_cu_t.tolist(), plan.s_rows
        for qa, qb, ka, kb in zip(plan.q_cu[:-1], plan.q_cu[1:], s_cu[:-1], s_cu[1:], strict=True):
            local = window_mask(qb - qa, kb - ka, window - 1)
            for i in range(qb - qa):
                want[qa + i, rows[ka:kb][local[i]]] = True
        self.assertEqual(got.tolist(), want.tolist())

    def test_vectorised_masks_equal_the_per_row_loop(self):
        """Per block instead of per row, the three masks are the same arrays, on karpo-decision's own plans:
        one-block sequences beside branched ones, shared blocks shorter than, at and past the window."""
        from karpo_apple_silicon.mlx import attention_masks

        window = 8
        layouts = ([(13, [5, 7, 3]), (11, [])], [(6, [4, 9]), (7, [3]), (8, [6, 2]), (9, [10, 1, 4])], [(3, [12, 15])],
                   [(1, []), (40, [20, 3, 17]), (2, [])])
        for layout in layouts:
            plan = upstream_plan(layout, window)
            got, want = attention_masks(plan, window), per_row_masks(plan, window)
            for name in ("global", "sliding", "last"):
                with self.subTest(layout=layout, mask=name):
                    self.assertEqual(got[name].dtype, np.bool_)
                    self.assertEqual(got[name].shape, want[name].shape)
                    self.assertTrue(np.array_equal(got[name], want[name]))

    def test_more_queries_than_keys_is_refused(self):
        from karpo_apple_silicon.mlx import attention_masks

        plan = {"query_bounds": [0, 3], "global_rows": [0, 1, 2], "global_bounds": [0, 3],
                "sliding_rows": [1, 2], "sliding_bounds": [0, 2], "marker_rows": [2]}
        with self.assertRaisesRegex(ValueError, "3 queries over 2 keys"):
            attention_masks(plan, 3)

    def test_chunks_cover_every_readable_key_with_the_shortest_run(self):
        """Each chunk of query rows gets the shortest run of key rows holding every key its rows may read; laid
        back side by side, the chunk masks are the whole mask and everything outside them is unreadable."""
        from karpo_apple_silicon.mlx import _chunks, attention_masks

        window = 8
        plan = upstream_plan([(13, [5, 7, 3]), (11, []), (30, [])], window)
        for name, mask in attention_masks(plan, window).items():
            for size in (1, 4, 7, 1000):
                rebuilt = np.zeros_like(mask)
                chunks = _chunks(mask, size)
                self.assertEqual([c[0] for c in chunks], list(range(0, mask.shape[0], size)))
                for start, stop, low, high, part in chunks:
                    block = mask[start:stop]
                    self.assertTrue(block[:, low].any() and block[:, high - 1].any(), "the run is not the shortest")
                    rebuilt[start:stop, low:high] = True if part is None else np.array(part)
                with self.subTest(mask=name, size=size):
                    self.assertTrue(np.array_equal(rebuilt, mask))
        with self.assertRaisesRegex(ValueError, "attention row 1 reads no key"):
            _chunks(np.array([[True, False], [False, False]]), 2)
        with self.assertRaisesRegex(ValueError, r"attention row 0 reads no key \(the first of 3 such rows\)"):
            _chunks(np.zeros((3, 2), dtype=bool), 2)


@unittest.skipIf(SKIP, SKIP)
class RotaryFrequencies(unittest.TestCase):
    def test_global_rope_leaves_most_of_the_head_unrotated(self):
        from karpo_apple_silicon.mlx import rotary_frequencies

        frequencies = rotary_frequencies(512, 1_000_000.0, 0.25)
        self.assertEqual(len(frequencies), 256)
        self.assertEqual(frequencies[0], 1.0)
        self.assertGreater(frequencies[63], 0.0)
        self.assertEqual(frequencies[64:], [0.0] * 192)

    def test_frequencies_equal_transformers_exactly_for_the_31b(self):
        """The 31B's two layer types (sliding: head 256, theta 10,000, the whole head; global: head 512, theta
        1,000,000, a quarter of the head, ``proportional``): every frequency is the float32 transformers holds,
        and the tables built from them match its own to the last bit or two of a float32 cosine."""
        try:
            from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
            from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
        except ImportError as error:
            self.skipTest(f"needs transformers' Gemma 4: {error}")
        import mlx.core as mx
        import torch

        from karpo_apple_silicon.mlx import _rotary_tables, rotary_frequencies

        config = Gemma4TextConfig(hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                  num_key_value_heads=1, head_dim=256, global_head_dim=512,
                                  layer_types=["sliding_attention", "full_attention"], vocab_size=8)
        self.assertEqual(config.rope_parameters["sliding_attention"], {"rope_type": "default", "rope_theta": 10_000.0})
        self.assertEqual(config.rope_parameters["full_attention"],
                         {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1_000_000.0})
        rotary = Gemma4TextRotaryEmbedding(config)
        positions = torch.tensor([0, 1, 2, 1023, 1024, 8191, 32767])
        for kind, head, theta, fraction in (("sliding_attention", 256, 10_000.0, 1.0),
                                            ("full_attention", 512, 1_000_000.0, 0.25)):
            with self.subTest(kind=kind):
                theirs = getattr(rotary, f"{kind}_inv_freq")
                self.assertEqual(theirs.dtype, torch.float32)
                self.assertEqual(rotary_frequencies(head, theta, fraction), theirs.tolist())
                want = rotary(torch.zeros(1), positions[None], kind)
                got = _rotary_tables(mx.array(positions.numpy()), head, theta, fraction, mx.float32)
                for ours, upstream in zip(got, want, strict=True):
                    self.assertLess(float(np.abs(np.array(ours) - upstream[0].numpy()).max()), 5e-7)


if __name__ == "__main__":
    unittest.main()
