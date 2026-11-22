# Copyright 2026 the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental: Karpo Decision 31B, text-only requests, with the transformer layers in MLX.

For Macs that cannot hold the 62.5 GB bfloat16 checkpoint: ``scripts/convert_mlx.py`` writes an 8-bit
(about 32 GB) or 4-bit (about 18 GB) copy of the text layers, and ``load(path).decide(request)`` answers
with it. Quantized answers move a little; see the README for measured differences.

karpo-decision's own encoder and block plan are reused unchanged, on CPU: they produce the token ids,
positions, readout rows and the virtual sequences each row may read. This module turns those virtual
sequences into dense masks and runs the layers and the readout with MLX.

Not supported: images, checkpoints whose ``attention`` is not ``bidirectional``, the Qwen-family
checkpoints. Needs ``mlx`` and ``mlx-lm`` (``pip install "karpo-apple-silicon[mlx]"``).
"""

import json
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from transformers import AutoTokenizer

from karpo_decision.answers import answer
from karpo_decision.encoding import Encoder
from karpo_decision.families.gemma import Processor
from karpo_decision.families.gemma.packed import block_plan
from karpo_decision.model.packed import pack
from karpo_decision.templates import template
from mlx_lm.models.gemma4_text import Gemma4TextModel, ModelArgs


def rotary_frequencies(head_dim, base, fraction):
    """Return Gemma 4's half-head frequencies, including the global no-op tail."""
    rotated = int(fraction * head_dim // 2)
    frequencies = [base ** (-2 * index / head_dim) for index in range(rotated)]
    frequencies.extend([0.0] * (head_dim // 2 - rotated))
    return frequencies


def attention_masks(plan, window):
    """Dense (query row, key row) masks over a pack's rows, from karpo-decision's virtual sequences.

    ``window`` is the model's ``sliding_window``: a query reads itself and the ``window - 1`` keys before
    it, counted along its virtual sequence's keys, aligned at the sequence's end.
    """
    query_bounds = plan["query_bounds"]
    global_bounds = plan["global_bounds"]
    sliding_bounds = plan["sliding_bounds"]
    global_rows = plan["global_rows"]
    sliding_rows = plan["sliding_rows"]
    marker_rows = plan["marker_rows"]
    length = query_bounds[-1]
    global_mask = np.zeros((length, length), dtype=bool)
    sliding_mask = np.zeros((length, length), dtype=bool)
    last_mask = np.zeros((len(marker_rows), length), dtype=bool)
    marker_lookup = {row: index for index, row in enumerate(marker_rows)}

    # Shared rows and each branch use their own Karpo virtual sequence.
    for block in range(len(query_bounds) - 1):
        q_start = query_bounds[block]
        q_end = query_bounds[block + 1]
        g_start = global_bounds[block]
        g_end = global_bounds[block + 1]
        s_start = sliding_bounds[block]
        s_end = sliding_bounds[block + 1]
        gathered_global = global_rows[g_start:g_end]
        gathered_sliding = sliding_rows[s_start:s_end]
        shift = len(gathered_sliding) - (q_end - q_start)
        for query in range(q_start, q_end):
            global_mask[query, gathered_global] = True
            aligned = query - q_start + shift
            low = max(0, aligned - window + 1)
            sliding_mask[query, gathered_sliding[low:aligned + 1]] = True
            if query in marker_lookup:
                last_mask[marker_lookup[query], gathered_global] = True
    return {"global": global_mask, "sliding": sliding_mask, "last": last_mask}


def _plan_dict(sequences, window):
    packed = pack(sequences, None, torch.device("cpu"))
    planned = block_plan(packed, window)
    length = packed.tokens
    global_rows = planned.g_rows.tolist() if planned.g_rows is not None else list(range(length))
    sliding_rows = planned.s_rows.tolist() if planned.s_rows is not None else list(range(length))
    return packed, {
        "query_bounds": planned.q_cu,
        "global_rows": global_rows,
        "global_bounds": planned.g_cu,
        "sliding_rows": sliding_rows,
        "sliding_bounds": planned.s_cu_t.tolist(),
        "marker_rows": packed.markers.tolist(),
    }


def _rms_norm(x, weight, eps):
    values = x.astype(mx.float32)
    variance = mx.mean(mx.square(values), axis=-1, keepdims=True)
    result = values * mx.rsqrt(variance + eps)
    if weight is not None:
        result = result * weight.astype(mx.float32)
    return result.astype(x.dtype)


def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    swapped = mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    return x * cos + swapped * sin


def _rotary_tables(positions, head_dim, base, fraction, dtype):
    frequencies = mx.array(rotary_frequencies(head_dim, base, fraction), dtype=mx.float32)
    angles = positions.astype(mx.float32)[:, None] * frequencies[None, :]
    repeated = mx.concatenate((angles, angles), axis=-1)
    return mx.cos(repeated).astype(dtype), mx.sin(repeated).astype(dtype)


def _project_qkv(attention, x, positions, rope, marker_rows=None):
    length = x.shape[0]
    dim = attention.head_dim
    query = attention.q_proj(x).reshape(length, attention.n_heads, dim)
    key = attention.k_proj(x).reshape(length, attention.n_kv_heads, dim)
    raw_value = key if attention.use_k_eq_v else attention.v_proj(x).reshape(length, attention.n_kv_heads, dim)
    query = _rms_norm(query, attention.q_norm.weight, attention.q_norm.eps)
    key = _rms_norm(key, attention.k_norm.weight, attention.k_norm.eps)
    value = _rms_norm(raw_value, None, attention.v_norm.eps)
    base, fraction = rope
    cos, sin = _rotary_tables(positions, dim, base, fraction, query.dtype)
    query = _rotate(query, cos[:, None, :], sin[:, None, :])
    key = _rotate(key, cos[:, None, :], sin[:, None, :])
    if marker_rows is not None:
        query = query[mx.array(marker_rows)]
    return query, key, value


def _attention(layer, h, positions, mask, rope, marker_rows=None):
    attention = layer.self_attn
    source = _rms_norm(h, layer.input_layernorm.weight, layer.input_layernorm.eps)
    query, key, value = _project_qkv(attention, source, positions, rope, marker_rows)
    query = query.transpose(1, 0, 2)[None]
    key = key.transpose(1, 0, 2)[None]
    value = value.transpose(1, 0, 2)[None]
    # Gemma 4 norms queries and keys, so its attention scale is 1.
    attended = mx.fast.scaled_dot_product_attention(query, key, value, scale=1.0, mask=mask)
    attended = attended[0].transpose(1, 0, 2).reshape(query.shape[2], -1)
    return attention.o_proj(attended)


def _feedforward(layer, h):
    x = _rms_norm(h, layer.pre_feedforward_layernorm.weight, layer.pre_feedforward_layernorm.eps)
    gate = nn.gelu_approx(layer.mlp.gate_proj(x))
    mlp = layer.mlp.down_proj(gate * layer.mlp.up_proj(x))
    return _rms_norm(mlp, layer.post_feedforward_layernorm.weight, layer.post_feedforward_layernorm.eps)


def _layer(layer, h, positions, mask, rope, marker_rows=None):
    attended = _attention(layer, h, positions, mask, rope, marker_rows)
    source = h if marker_rows is None else h[mx.array(marker_rows)]
    post = _rms_norm(attended, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps)
    residual = source + post
    return (residual + _feedforward(layer, residual)) * layer.layer_scalar.astype(residual.dtype)


class _Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.language_model = Gemma4TextModel(ModelArgs.from_dict(config))


class MLXDecider:
    def __init__(self, path, token_budget=4096):
        self.path = Path(path)
        self.token_budget = token_budget
        metadata = json.loads((self.path / "mlx_config.json").read_text())
        self.bits = metadata["bits"]
        text = metadata["text_config"]
        if text.get("hidden_size_per_layer_input") or text.get("num_kv_shared_layers") or text.get("enable_moe_block"):
            raise ValueError("per-layer embeddings, shared key-value layers and expert blocks are not supported")
        self.window = text["sliding_window"]
        self.logit_cap = text.get("final_logit_softcapping")
        self.rope = {kind: (float(values["rope_theta"]), float(values.get("partial_rotary_factor", 1.0)))
                     for kind, values in text["rope_parameters"].items()}
        self.model = _Backbone(metadata["text_config"])
        nn.quantize(
            self.model,
            group_size=metadata["group_size"],
            bits=self.bits,
            class_predicate=lambda name, module: isinstance(module, nn.Linear),
        )
        for weight_file in sorted(self.path.glob("text-*.safetensors")):
            self.model.load_weights(list(mx.load(str(weight_file)).items()), strict=False)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.readout = mx.load(str(self.path / "readout.safetensors"))["weight"].astype(mx.float32)
        self.config = json.loads((self.path / "decision_config.json").read_text())
        if self.config["attention"] != "bidirectional":
            raise ValueError("only bidirectional checkpoints are supported (the global layers read both ways "
                             "and the last layer runs at the readout rows only)")
        if self.model.language_model.layers[-1].layer_type != "full_attention":
            raise ValueError("the last layer must be a global layer")
        tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        processor = Processor(tokenizer, None)
        self.encoder = Encoder(
            processor,
            self.config["codes"],
            template(self.config["template"]),
            self.config["marker"],
        )

    def _forward(self, sequences):
        packed, plan = _plan_dict(sequences, self.window)
        masks = attention_masks(plan, self.window)
        sliding = mx.array(masks["sliding"])
        global_mask = mx.array(masks["global"])
        last = mx.array(masks["last"])
        ids = mx.array(packed.input_ids.numpy())
        positions = mx.array(packed.seqpos.numpy())
        tokens = self.model.language_model.embed_tokens(ids)
        h = tokens * mx.array(math.sqrt(tokens.shape[-1]), dtype=tokens.dtype)
        for layer in self.model.language_model.layers[:-1]:
            mask = sliding if layer.layer_type == "sliding_attention" else global_mask
            h = _layer(layer, h, positions, mask, self.rope[layer.layer_type])
        final_layer = self.model.language_model.layers[-1]
        h = _layer(final_layer, h, positions, last, self.rope[final_layer.layer_type], plan["marker_rows"])
        norm = self.model.language_model.norm
        h = _rms_norm(h, norm.weight, norm.eps)
        logits = h.astype(mx.float32) @ self.readout.T
        if self.logit_cap:
            logits = self.logit_cap * mx.tanh(logits / self.logit_cap)
        mx.eval(logits)
        result = np.array(logits)
        # MLX keeps freed buffers for reuse, but requests of different lengths rarely reuse them, so the
        # cache would grow by gigabytes per long request until the system kills the process.
        del logits, h, tokens, sliding, global_mask, last
        mx.clear_cache()
        return result

    def decide(self, request):
        return self.decide_many([request])[0]

    def _packed_logits(self, sequences):
        token_budget = self.token_budget
        groups = []
        current = []
        used = 0
        for sequence in sequences:
            if current and used + sequence.length > token_budget:
                groups.append(current)
                current = []
                used = 0
            current.append(sequence)
            used += sequence.length
        if current:
            groups.append(current)
        return np.concatenate([self._forward(group) for group in groups], axis=0)

    def decide_many(self, requests):
        plans = [self.encoder.plan(request, 32768) for request in requests]
        sequences = [part for parts in plans for part in parts]
        if not sequences:
            return []
        if any(sequence.images for sequence in sequences):
            raise ValueError("the MLX path reads text-only requests")
        logits = self._packed_logits(sequences)
        results = []
        cursor = 0
        for request, parts in zip(requests, plans, strict=True):
            answers = {}
            for part in parts:
                for key, count in zip(part.keys, part.counts, strict=True):
                    question = request["questions"][key]
                    selected = logits[cursor, :count] / self.config["temperatures"][question["type"]]
                    shifted = selected - np.max(selected)
                    probabilities = np.exp(shifted) / np.exp(shifted).sum()
                    answers[key] = answer(question, probabilities.tolist())
                    cursor += 1
            usage = {
                "input_tokens": sum(part.length for part in parts),
                "output_tokens": 0,
                "sequences": len(parts),
            }
            results.append({"answers": answers, "usage": usage})
        return results


def load(path, token_budget=4096):
    """A decider over a folder written by ``scripts/convert_mlx.py``. ``token_budget``: tokens per forward
    pass when several requests are answered together (``decide_many``)."""
    started = time.perf_counter()
    model = MLXDecider(path, token_budget)
    model.load_seconds = time.perf_counter() - started
    return model
