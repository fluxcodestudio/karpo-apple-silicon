# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Tiny Gemma 4 configuration and block layouts follow upstream's Apache-2.0
tests/test_gemma_blocks.py, without importing its test helpers.
"""

from __future__ import annotations

import contextlib
import types
import unittest
from unittest.mock import patch

import torch

import karpo_apple_silicon
from karpo_apple_silicon.reference import dense_attend

try:
    from karpo_decision.encoding import Encoded
    from karpo_decision.families.gemma import packed, sliding_attention
    from karpo_decision.model.packed import pack
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
except ImportError:
    AVAILABLE = False
else:
    AVAILABLE = True

WINDOW = 8
LAYOUTS = (
    ((6, ()),),
    ((WINDOW - 1, (3,)), (WINDOW, (6, 2)), (WINDOW + 1, (10, 1, 4))),
    ((3, (12, 15)),),
    ((13, ()),),
)


def sequence(shared: int, branches: tuple[int, ...]) -> Encoded:
    """Build one shared block with optional branches and their final-token markers."""
    lengths = [shared, *branches]
    input_ids = torch.randint(3, 97, (sum(lengths),))
    positions = list(range(shared))
    segments = [0] * shared
    markers = []
    at = shared
    for number, length in enumerate(branches, start=1):
        positions.extend(range(shared, shared + length))
        segments.extend([number] * length)
        at += length
        markers.append(at - 1)
    if not branches:
        markers.append(shared - 1)
    return Encoded(keys=tuple(str(index) for index in range(max(1, len(branches)))),
                   counts=(2,) * max(1, len(branches)), input_ids=input_ids,
                   markers=torch.tensor(markers), positions=torch.tensor(positions),
                   segment_ids=torch.tensor(segments, dtype=torch.int16))


def model(dtype: torch.dtype) -> Gemma4TextModel:
    torch.manual_seed(0)
    layer_types = ["sliding_attention", "sliding_attention", "full_attention",
                   "sliding_attention", "full_attention"]
    config = Gemma4TextConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
                              num_hidden_layers=len(layer_types), num_attention_heads=4, num_key_value_heads=2,
                              head_dim=8, global_head_dim=16, num_global_key_value_heads=1,
                              attention_k_eq_v=True, sliding_window=WINDOW, layer_types=layer_types,
                              hidden_size_per_layer_input=0, final_logit_softcapping=30.0)
    config._attn_implementation = "sdpa"
    language_model = Gemma4TextModel(config).to(dtype=dtype)
    with torch.no_grad():
        for parameter in language_model.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
        for layer in language_model.layers:
            layer.layer_scalar.fill_(0.9)
    return language_model.eval()


@unittest.skipUnless(AVAILABLE, "karpo-decision and transformers.models.gemma4 are required")
class PackedForward(unittest.TestCase):
    def setUp(self):
        self.addCleanup(karpo_apple_silicon.uninstall)
        karpo_apple_silicon.install()

    def forward(self, language_model, layout, attention, device="cpu"):
        sequences = [sequence(shared, branches) for shared, branches in layout]
        packed_inputs = pack(sequences, None, torch.device(device))
        with torch.no_grad():
            return packed.Packed(types.SimpleNamespace(language_model=language_model), attention)(packed_inputs)

    def test_cpu_forward_runs(self):
        language_model = model(torch.float32)
        for attention in ("bidirectional", "causal"):
            for layout in LAYOUTS:
                with self.subTest(attention=attention, layout=layout):
                    result = self.forward(language_model, layout, attention)
                    self.assertEqual(result.shape, (sum(max(1, len(branches)) for _, branches in layout), 32))
                    self.assertTrue(torch.isfinite(result).all())

    def test_banded_matches_dense_in_float64(self):
        language_model = model(torch.float64)
        for attention in ("bidirectional", "causal"):
            for layout in LAYOUTS:
                torch.manual_seed(1)
                with patch.object(karpo_apple_silicon, "CHUNK", 3):
                    result = self.forward(language_model, layout, attention)
                torch.manual_seed(1)
                with patch.object(sliding_attention, "attend", dense_attend):
                    reference = self.forward(language_model, layout, attention)
                torch.testing.assert_close(result, reference, atol=1e-9, rtol=1e-9)

    def test_chunked_global_matches_one_call_per_sequence_in_float64(self):
        language_model = model(torch.float64)
        upstream_global = karpo_apple_silicon._originals["global"]
        for attention in ("bidirectional", "causal"):
            for layout in LAYOUTS:
                torch.manual_seed(1)
                with patch.object(karpo_apple_silicon, "CHUNK", 3):
                    result = self.forward(language_model, layout, attention)
                torch.manual_seed(1)
                with patch.object(karpo_apple_silicon, "CHUNK", 3), \
                     patch.object(packed.Packed, "_global", upstream_global), \
                     patch.object(packed, "sdpa_kernel", lambda *_: contextlib.nullcontext()):
                    reference = self.forward(language_model, layout, attention)
                torch.testing.assert_close(result, reference, atol=1e-9, rtol=1e-9)

    def test_both_global_call_sites_reach_the_replacement(self):
        spans = []

        def recorded(query, key, value, q_bounds, k_bounds, lens, causal, scale):
            spans.append((q_bounds[-1] - q_bounds[0], k_bounds[-1] - k_bounds[0]))
            return karpo_apple_silicon.global_attend(query, key, value, q_bounds, k_bounds, lens, causal, scale)

        layout = LAYOUTS[1]
        with patch.object(packed.Packed, "_global", staticmethod(recorded)):
            self.forward(model(torch.float32), layout, "bidirectional")
        tokens = sum(shared + sum(branches) for shared, branches in layout)
        readouts = sum(max(1, len(branches)) for _, branches in layout)
        keys = sum(shared * (1 + len(branches)) + sum(branches) for shared, branches in layout)
        # The model has two global layers: every row asks in the first, only the readout rows in the last, and
        # in both each branch reads its shared block as well as itself.
        self.assertEqual(spans, [(tokens, keys), (readouts, keys)])

    def test_uninstall_restores_original_objects(self):
        original_attend = karpo_apple_silicon._originals["attend"]
        original_global = karpo_apple_silicon._originals["global"]
        self.assertIsInstance(original_global, staticmethod)
        self.assertIs(packed.Packed._global, karpo_apple_silicon.global_attend)
        karpo_apple_silicon.uninstall()
        self.assertIs(sliding_attention.attend, original_attend)
        self.assertIs(packed.Packed.__dict__["_global"], original_global)
        karpo_apple_silicon.install()
        karpo_apple_silicon.install()
        karpo_apple_silicon.uninstall()
        self.assertIs(sliding_attention.attend, original_attend)
        self.assertIs(packed.Packed.__dict__["_global"], original_global)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_matches_cpu(self):
        for attention in ("bidirectional", "causal"):
            for dtype, tolerance in ((torch.float32, 1e-4), (torch.float16, 0.2), (torch.bfloat16, 0.2)):
                with self.subTest(attention=attention, dtype=dtype):
                    language_model = model(torch.float32)
                    torch.manual_seed(1)
                    cpu = self.forward(language_model, LAYOUTS[1], attention)
                    language_model.to(device="mps", dtype=dtype)
                    torch.manual_seed(1)
                    gpu = self.forward(language_model, LAYOUTS[1], attention, "mps")
                    self.assertTrue(torch.isfinite(gpu).all())
                    relative = (gpu.cpu().float() - cpu).abs().max() / cpu.abs().max()
                    self.assertLess(float(relative), tolerance)
