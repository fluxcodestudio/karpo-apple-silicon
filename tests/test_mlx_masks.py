# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""The MLX path's masks and rotary frequencies. Needs mlx, mlx-lm and karpo-decision; no weights."""

from __future__ import annotations

import importlib.util
import unittest

NEEDED = ("mlx", "mlx_lm", "karpo_decision")


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in NEEDED), f"needs {', '.join(NEEDED)}")
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
        for qa, qb, ka, kb in zip(plan.q_cu[:-1], plan.q_cu[1:], s_cu[:-1], s_cu[1:]):
            local = window_mask(qb - qa, kb - ka, window - 1)
            for i in range(qb - qa):
                want[qa + i, rows[ka:kb][local[i]]] = True
        self.assertEqual(got.tolist(), want.tolist())

    def test_global_rope_leaves_most_of_the_head_unrotated(self):
        from karpo_apple_silicon.mlx import rotary_frequencies

        frequencies = rotary_frequencies(512, 1_000_000.0, 0.25)
        self.assertEqual(len(frequencies), 256)
        self.assertEqual(frequencies[0], 1.0)
        self.assertGreater(frequencies[63], 0.0)
        self.assertEqual(frequencies[64:], [0.0] * 192)


if __name__ == "__main__":
    unittest.main()
