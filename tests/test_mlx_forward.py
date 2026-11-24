# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""The MLX forward pass against karpo-decision's own packed forward, on a tiny random Gemma 4: float32,
unquantized, the same weights on both sides, nothing downloaded. Needs mlx, mlx-lm, karpo-decision and
transformers' Gemma 4. Skipped where MLX cannot start (no Metal device: sandboxes, some CI runners).

karpo-decision's forward runs on CPU here: its sliding-window kernel is swapped for this package's ``attend`` and
its global layers' CUDA kernel selection is lifted, both for the length of one call, so nothing stays patched.
"""

from __future__ import annotations

import contextlib
import importlib.util
import types
import unittest
from unittest import mock

import numpy as np


def unavailable() -> str | None:
    """Why the MLX path cannot be tested here, or None."""
    for name in ("mlx", "mlx_lm", "karpo_decision", "transformers.models.gemma4"):
        try:
            found = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            found = None
        if found is None:
            return f"needs {name}"
    try:
        import mlx.core  # noqa: F401
    except Exception as error:  # MLX raises on import when there is no Metal device
        return f"MLX cannot start: {error}"
    return None


SKIP = unavailable()
WINDOW = 8
TYPES = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "sliding_attention",
         "full_attention"]
#: query heads, key heads of the sliding and of the global layers, head sizes of both. MLX's attention takes
#: another internal route from head size 256 up, so the 31B's own attention shapes are run too.
SHAPES = {
    "8 heads over 4 and 1, heads of 8 and 16": dict(heads=8, kv=4, global_kv=1, head=8, global_head=16),
    "the 31B's: 32 heads over 16 and 4, heads of 256 and 512": dict(heads=32, kv=16, global_kv=4, head=256,
                                                                    global_head=512),
}


def tiny(heads: int, kv: int, global_kv: int, head: int, global_head: int):
    """A random Gemma 4 text model in transformers and the ``text_config`` the converter would write for it."""
    import torch
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    torch.manual_seed(0)
    config = Gemma4TextConfig(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=len(TYPES),
                              num_attention_heads=heads, num_key_value_heads=kv, head_dim=head,
                              global_head_dim=global_head, num_global_key_value_heads=global_kv, attention_k_eq_v=True,
                              sliding_window=WINDOW, layer_types=TYPES, hidden_size_per_layer_input=0,
                              final_logit_softcapping=30.0)
    model = Gemma4TextModel(config).float().eval()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:  # norm weights away from 1 and layer scalars away from 1, so a wrong rule shows
                parameter.add_(0.2 * torch.randn_like(parameter))
            if ".q_norm." in name or ".k_norm." in name:  # smaller scores: a softmax that is not nearly one-hot
                parameter.mul_(0.25)                      # does not amplify float32 rounding
        for index, layer in enumerate(model.layers):
            layer.layer_scalar.fill_(0.8 + 0.05 * index)
    text = {**config.to_dict(), "global_head_dim": global_head, "num_global_key_value_heads": global_kv}
    return model, text


def sequence(shared: int, branches: list[int]):
    """A block layout: one block of ``shared`` tokens, or a shared block and one branch per entry."""
    import torch
    from karpo_decision.encoding import Encoded

    positions, segments, markers, at = list(range(shared)), [0] * shared, [], shared
    for number, rows in enumerate(branches, start=1):
        positions += range(shared, shared + rows)
        segments += [number] * rows
        at += rows
        markers.append(at - 1)
    count = max(1, len(branches))
    return Encoded(keys=tuple(str(i) for i in range(count)), counts=(2,) * count, input_ids=torch.randint(3, 97, (at,)),
                   markers=torch.tensor(markers or [shared - 1]), positions=torch.tensor(positions),
                   segment_ids=torch.tensor(segments, dtype=torch.int16))


def upstream_states(model, sequences) -> np.ndarray:
    """karpo-decision's ``Packed`` forward on CPU: the final-normed hidden states at the readout rows."""
    import torch
    from karpo_decision.families.gemma import packed, sliding_attention
    from karpo_decision.model.packed import pack

    import karpo_apple_silicon

    with (mock.patch.object(packed, "sdpa_kernel", lambda *_: contextlib.nullcontext()),
          mock.patch.object(sliding_attention, "attend", karpo_apple_silicon.attend), torch.inference_mode()):
        forward = packed.Packed(types.SimpleNamespace(language_model=model), "bidirectional")
        return forward(pack(sequences, None, torch.device("cpu"))).numpy()


def mlx_model(model, text: dict):
    """mlx-lm's model for ``text`` with ``model``'s weights (every one of them: the load is strict), and the
    rotary parameters, as ``MLXDecider`` holds them."""
    import mlx.core as mx

    from karpo_apple_silicon import mlx as karpo_mlx

    backbone = karpo_mlx._Backbone(karpo_mlx._arguments(text))
    backbone.load_weights([("language_model." + name, mx.array(value.numpy()))
                           for name, value in model.state_dict().items()], strict=True)
    return backbone.language_model, karpo_mlx._rope(text, TYPES)


@unittest.skipIf(SKIP, SKIP)
class Forward(unittest.TestCase):
    def test_readout_states_match_karpo_decision(self):
        """A shared block longer than the window with three branches, a one-block sequence and a branch longer
        than the window, packed together: within 1e-4 of upstream, in one attention call per layer and in
        chunks of five query rows, which cross every block and sequence boundary."""
        import torch

        from karpo_apple_silicon import mlx as karpo_mlx

        for label, shape in SHAPES.items():
            model, text = tiny(**shape)
            language_model, rope = mlx_model(model, text)
            torch.manual_seed(1)
            sequences = [sequence(13, [5, 7, 3]), sequence(11, []), sequence(3, [12])]
            want = upstream_states(model, sequences)
            self.assertEqual(want.shape, (5, 32))
            for chunk in (karpo_mlx.CHUNK, 5):
                with self.subTest(shape=label, chunk=chunk), mock.patch.object(karpo_mlx, "CHUNK", chunk):
                    got = np.array(karpo_mlx._marker_states(language_model, sequences, WINDOW, rope))
                    self.assertEqual(got.shape, want.shape)
                    self.assertLess(float(np.abs(got - want).max() / np.abs(want).max()), 1e-4)
            with self.subTest(shape=label, check="a window one too wide is told apart"):
                wrong = np.array(karpo_mlx._marker_states(language_model, sequences, WINDOW + 1, rope))
                self.assertGreater(float(np.abs(wrong - want).max() / np.abs(want).max()), 1e-2)

    def test_chunks_of_five_rows_are_crossed(self):
        """The pack above is 54 rows: in chunks of five, eleven attention calls per layer, not one."""
        from karpo_apple_silicon import mlx as karpo_mlx

        _, plan = karpo_mlx._plan_dict([sequence(13, [5, 7, 3]), sequence(11, []), sequence(3, [12])], WINDOW)
        masks = karpo_mlx.attention_masks(plan, WINDOW)
        self.assertEqual({name: len(karpo_mlx._chunks(mask, 5)) for name, mask in masks.items()},
                         {"global": 11, "sliding": 11, "last": 1})
        self.assertEqual({name: len(karpo_mlx._chunks(mask, karpo_mlx.CHUNK)) for name, mask in masks.items()},
                         {"global": 1, "sliding": 1, "last": 1})


if __name__ == "__main__":
    unittest.main()
