# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Experimental: Karpo Decision 31B, text-only requests, with the transformer layers in MLX.

For Macs that cannot hold the 62.6 GB bfloat16 checkpoint: ``scripts/convert_mlx.py`` writes an 8-bit
(34.0 GB) or 4-bit (19.3 GB) copy of the text layers, and ``load(path).decide(request)`` answers
with it. Quantized answers move a little; see the README for measured differences.

karpo-decision's own encoder and block plan are reused unchanged, on CPU: they produce the token ids,
positions, readout rows and the virtual sequences each row may read. This module turns those virtual
sequences into masks and runs the layers and the readout with MLX. Attention runs ``CHUNK`` query rows at
a time, each chunk over the shortest run of rows that holds every key its rows may read.

Not supported: images, checkpoints whose ``attention`` is not ``bidirectional``, the Qwen-family
checkpoints. Needs ``mlx`` and ``mlx-lm`` (this package's ``mlx`` extra) and karpo-decision, which is
installed separately; the README has the commands.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from karpo_decision.answers import answer
from karpo_decision.config import DecisionConfig
from karpo_decision.decider import DEFAULT_MAX_LENGTH
from karpo_decision.encoding import Encoded, Encoder
from karpo_decision.families import gemma
from karpo_decision.families.gemma.packed import block_plan
from karpo_decision.model.packed import Pack, pack
from karpo_decision.protocol import Request
from karpo_decision.templates import template
from mlx.utils import tree_flatten
from mlx_lm.models.gemma4_text import Attention, DecoderLayer, Gemma4TextModel, ModelArgs
from transformers import AutoTokenizer

__all__ = ["CHUNK", "MLXDecider", "attention_masks", "decision_config", "expected_tensors", "load",
           "rotary_frequencies"]

SLIDING, GLOBAL = "sliding_attention", "full_attention"
#: The MLP gate ``_feedforward`` computes (``nn.gelu_approx``), under transformers' name for it.
ACTIVATION = "gelu_pytorch_tanh"
#: Query rows per attention call; each call reads only the shortest run of key rows that holds every key its
#: rows may read. On an Apple GPU, with random bfloat16 tensors of the 31B's shapes, the attention of one sliding
#: layer over 8,192 rows took 316 ms as one call over every row and 53 ms in chunks of 256. Of 128, 256, 512 and
#: 1,024, 256 was the fastest or within 3 ms or 1% of it, for one long sequence and for twenty packed ones. In a
#: whole request most of a layer's time is its quantized projections, so chunking saves memory more than time.
CHUNK = 256
#: Bytes of attention scores (heads x query rows x key rows) above which a chunk is computed at once instead of
#: with the rest of the forward pass, so that its scores are released before the next chunk's are made. Only a
#: long sequence's global layers pass it (more than 4,096 key rows for the 31B); a sliding layer's chunks (21 MB)
#: were a third slower computed one at a time. For one global layer's attention alone at 8,192 rows the peak was
#: 0.5 GB this way and 2.9 GB left to the end. In a whole 7,464-token request on the 4-bit 31B the peak was the
#: same with and without it (20.9 GB and 20.7 GB) and no difference in time could be measured; it is kept as a
#: bound for longer sequences, which were not measured.
SCORES_AT_ONCE = 64 * 1024 * 1024
#: ``decision_config.json``'s ``readout_dtype`` values.
READOUT_DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16}

#: (first query row, one past the last, first key row, one past the last, mask over them or None for no mask).
Chunk = tuple[int, int, int, int, mx.array | None]


def rotary_frequencies(head_dim: int, base: float, fraction: float) -> list[float]:
    """Gemma 4's rotary frequencies for half a head: the first ``fraction`` of them rotate, the rest are 0.

    They are computed in float32 with torch, the way transformers computes them
    (``Gemma4TextRotaryEmbedding``: its default type for the sliding layers, ``proportional`` for the global
    ones), so they are upstream's frequencies bit for bit. Each value is a float32 held in a float.
    """
    rotated = int(fraction * head_dim // 2)
    exponents = torch.arange(0, 2 * rotated, 2, dtype=torch.int64).to(torch.float) / head_dim
    return [*(1.0 / (base ** exponents)).tolist(), *[0.0] * (head_dim // 2 - rotated)]


def attention_masks(plan: dict[str, Any], window: int) -> dict[str, np.ndarray]:
    """Boolean masks over a pack's rows, True where a query row may read a key row, from karpo-decision's
    virtual sequences: ``global`` and ``sliding`` are (rows, rows), ``last`` is (readout rows, rows).

    ``plan`` holds, per virtual sequence, the bounds of its query rows (``query_bounds``) and of its key rows in
    ``global_rows`` and ``sliding_rows`` (``global_bounds``, ``sliding_bounds``), and the pack's ``marker_rows``.
    A global query reads every key of its sequence. ``window`` is the model's ``sliding_window``: a sliding
    query reads itself and the ``window - 1`` keys before it, counted along its sequence's keys, aligned at the
    sequence's end. ``last`` is the global mask at the readout rows.
    """
    query_bounds, global_bounds, sliding_bounds = plan["query_bounds"], plan["global_bounds"], plan["sliding_bounds"]
    global_rows = np.asarray(plan["global_rows"], dtype=np.int64)
    sliding_rows = np.asarray(plan["sliding_rows"], dtype=np.int64)
    length = query_bounds[-1]
    global_mask = np.zeros((length, length), dtype=bool)
    sliding_mask = np.zeros((length, length), dtype=bool)
    # Shared rows and each branch are their own virtual sequence; every row is a query of exactly one.
    for block in range(len(query_bounds) - 1):
        start, end = query_bounds[block], query_bounds[block + 1]
        global_keys = global_rows[global_bounds[block]:global_bounds[block + 1]]
        sliding_keys = sliding_rows[sliding_bounds[block]:sliding_bounds[block + 1]]
        shift = len(sliding_keys) - (end - start)
        if shift < 0:
            raise ValueError(f"sliding attention: {end - start} queries over {len(sliding_keys)} keys leaves "
                             "queries without any key")
        global_mask[start:end, _columns(global_keys)] = True
        aligned = np.arange(end - start)[:, None] + shift
        position = np.arange(len(sliding_keys))[None, :]
        sliding_mask[start:end, _columns(sliding_keys)] = (position <= aligned) & (position > aligned - window)
    last_mask = global_mask[np.asarray(plan["marker_rows"], dtype=np.int64)]
    return {"global": global_mask, "sliding": sliding_mask, "last": last_mask}


def _columns(rows: np.ndarray) -> slice | np.ndarray:
    """``rows`` as a slice when they are one unbroken ascending run (a block's own rows), which numpy fills
    several times faster than a list of rows; otherwise ``rows`` itself (a branch: the shared rows, then its own)."""
    if len(rows) and bool((np.diff(rows) == 1).all()):
        return slice(int(rows[0]), int(rows[-1]) + 1)
    return rows


def decision_config(path: str | Path) -> DecisionConfig:
    """``decision_config.json`` of ``path``, read and validated by karpo-decision's own ``DecisionConfig``, and
    refused (``ValueError``) when this module cannot answer with it: the global layers here read both ways and
    the last layer runs at the readout rows only, over a Gemma block layout."""
    config = DecisionConfig.load(path)
    if config.attention != "bidirectional":
        raise ValueError("only bidirectional checkpoints are supported (the global layers read both ways "
                         "and the last layer runs at the readout rows only)")
    if config.template not in gemma.TEMPLATES:
        raise ValueError(f"template {config.template!r} is not a Gemma block layout ({', '.join(gemma.TEMPLATES)})")
    return config


def expected_tensors(text_config: dict[str, Any], bits: int, group_size: int) -> dict[str, tuple[int, ...]]:
    """Name and shape of every tensor ``load`` requires in a folder's ``text-*.safetensors`` files, for a
    checkpoint's ``text_config`` quantized to ``bits`` bits in groups of ``group_size``.

    ``scripts/convert_mlx.py`` checks what it wrote against this. The model is built the way ``load`` builds
    it and nothing is computed, so it costs no memory. Raises ``ValueError`` for a configuration this module
    does not compute.
    """
    return {name: tuple(value.shape) for name, value in tree_flatten(_backbone(text_config, bits, group_size)
                                                                     .parameters())}


class _Backbone(nn.Module):
    """mlx-lm's Gemma 4 text model under the name the checkpoint stores it by, ``language_model``."""

    def __init__(self, arguments: ModelArgs) -> None:
        super().__init__()
        self.language_model = Gemma4TextModel(arguments)


def _arguments(text: dict[str, Any]) -> ModelArgs:
    """mlx-lm's model arguments for a ``text_config``, refusing what this module does not compute.

    The checks read the built arguments, not the dictionary: a key the file leaves out takes mlx-lm's default
    (256 for the per-layer embedding size, 20 shared key-value layers), which a check of the dictionary alone
    would let through.
    """
    missing = [key for key in ("layer_types", "rope_parameters", "sliding_window") if not text.get(key)]
    if missing:
        raise ValueError(f"text_config has no {', '.join(missing)}")
    arguments = ModelArgs.from_dict(text)
    if len(arguments.layer_types) != arguments.num_hidden_layers:  # mlx-lm builds num_hidden_layers of them
        raise ValueError(f"{len(arguments.layer_types)} layer types for {arguments.num_hidden_layers} layers")
    if arguments.hidden_size_per_layer_input or arguments.num_kv_shared_layers or arguments.enable_moe_block:
        raise ValueError("per-layer embeddings, shared key-value layers and expert blocks are not supported")
    activation = text.get("hidden_activation", ACTIVATION)
    if activation != ACTIVATION:
        raise ValueError(f"the MLP computed here is {ACTIVATION}, not {activation}")
    unknown = set(arguments.layer_types) - {SLIDING, GLOBAL}
    if unknown:
        raise ValueError(f"unknown layer types: {', '.join(sorted(unknown))}")
    if arguments.layer_types[-1] != GLOBAL:
        raise ValueError("the last layer must be a global layer")
    if text.get("use_bidirectional_attention") == "all":
        raise ValueError("use_bidirectional_attention 'all' is not supported: it changes the sliding window")
    return arguments


def _rope(text: dict[str, Any], layer_types: Sequence[str]) -> dict[str, tuple[float, float]]:
    """Per layer type, the rotary base and the fraction of each half head that rotates."""
    rope = {}
    for kind, values in text["rope_parameters"].items():
        rope_type = values.get("rope_type", "default")
        if rope_type not in ("default", "proportional") or values.get("factor", 1.0) != 1.0:
            raise ValueError(f"the {kind} layers' rotary type {rope_type!r} is not supported")
        # transformers' default type rotates the whole head, whatever partial_rotary_factor says.
        fraction = float(values.get("partial_rotary_factor", 1.0)) if rope_type == "proportional" else 1.0
        rope[kind] = (float(values["rope_theta"]), fraction)
    missing = set(layer_types) - rope.keys()
    if missing:
        raise ValueError(f"rope_parameters has no entry for {', '.join(sorted(missing))}")
    return rope


def _backbone(text: dict[str, Any], bits: int, group_size: int) -> _Backbone:
    """The text layers as mlx-lm builds them for a ``text_config``, every linear projection quantized.
    Building computes nothing: the weights are filled in by ``_load_weights``."""
    model = _Backbone(_arguments(text))
    nn.quantize(model, group_size=group_size, bits=bits,
                class_predicate=lambda _, module: isinstance(module, nn.Linear))
    return model


def _load_weights(model: _Backbone, path: Path) -> None:
    """Every ``text-*.safetensors`` of ``path`` into ``model``, all or nothing: a missing file or tensor, an
    extra tensor, a wrong shape or a file that cannot be read (cut short, say) is an error, never a model left
    partly at its random initial values."""
    files = sorted(path.glob("text-*.safetensors"))
    weights: dict[str, mx.array] = {}
    repeated: set[str] = set()
    try:
        for file in files:
            shard = mx.load(str(file))  # RuntimeError for a file that is cut short or is not safetensors
            repeated |= weights.keys() & shard.keys()
            weights.update(shard)
        if repeated:
            raise ValueError(f"{len(repeated)} tensors are in more than one file: {', '.join(sorted(repeated))}")
        model.load_weights(list(weights.items()), strict=True)
    except (ValueError, RuntimeError) as error:
        reason = " ".join(str(error).split())
        reason = reason if len(reason) <= 200 else reason[:200] + " ..."
        raise ValueError(f"{path} is incomplete or was not written by scripts/convert_mlx.py: it holds "
                         f"{len(weights)} tensors in {len(files)} text-*.safetensors files. {reason}") from error


def _plan_dict(sequences: Sequence[Encoded], window: int) -> tuple[Pack, dict[str, Any]]:
    """``sequences`` packed end to end by karpo-decision on CPU, and its block plan as ``attention_masks``
    reads it."""
    packed = pack(sequences, None, torch.device("cpu"))
    planned = block_plan(packed, window)
    own = np.arange(packed.tokens)  # without branches every virtual sequence's keys are its own rows
    return packed, {
        "query_bounds": planned.q_cu,
        "global_rows": own if planned.g_rows is None else planned.g_rows.numpy(),
        "global_bounds": planned.g_cu,
        "sliding_rows": own if planned.s_rows is None else planned.s_rows.numpy(),
        "sliding_bounds": planned.s_cu_t.tolist(),
        "marker_rows": packed.markers.tolist(),
    }


def _chunks(mask: np.ndarray, size: int) -> list[Chunk]:
    """``mask`` (query rows, key rows) cut into runs of ``size`` query rows, each with the shortest run of key
    rows that holds every key any of its rows may read and the mask over that run (None when every row of the
    chunk reads the whole run)."""
    if not mask.any(axis=1).all():
        raise ValueError("an attention row reads no key")
    chunks = []
    for start in range(0, mask.shape[0], size):
        stop = min(start + size, mask.shape[0])
        columns = np.flatnonzero(mask[start:stop].any(axis=0))
        low, high = int(columns[0]), int(columns[-1]) + 1
        part = mask[start:stop, low:high]
        chunks.append((start, stop, low, high, None if part.all() else mx.array(part)))
    return chunks


def _rms_norm(x: mx.array, weight: mx.array | None, eps: float) -> mx.array:
    """Gemma 4's RMS norm: computed in float32, a direct weight (None for the value norm), back in ``x``'s
    type."""
    values = x.astype(mx.float32)
    variance = mx.mean(mx.square(values), axis=-1, keepdims=True)
    result = values * mx.rsqrt(variance + eps)
    if weight is not None:
        result = result * weight.astype(mx.float32)
    return result.astype(x.dtype)


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    swapped = mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)
    return x * cos + swapped * sin


def _rotary_tables(positions: mx.array, head_dim: int, base: float, fraction: float,
                   dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
    """(rows, head) cosines and sines of each row's position times the head's frequencies, angles in float32."""
    frequencies = mx.array(rotary_frequencies(head_dim, base, fraction), dtype=mx.float32)
    angles = positions.astype(mx.float32)[:, None] * frequencies[None, :]
    repeated = mx.concatenate((angles, angles), axis=-1)
    return mx.cos(repeated).astype(dtype), mx.sin(repeated).astype(dtype)


def _keys_values(attention: Attention, x: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
    """(1, key heads, rows, head) keys (normed, rotated) and values (normed, no weight) of every row. A global
    layer of the 31B has one projection for both: its value is the raw key projection, before the key norm."""
    rows, dim = x.shape[0], attention.head_dim
    raw = attention.k_proj(x).reshape(rows, attention.n_kv_heads, dim)
    value = raw if attention.use_k_eq_v else attention.v_proj(x).reshape(rows, attention.n_kv_heads, dim)
    key = _rotate(_rms_norm(raw, attention.k_norm.weight, attention.k_norm.eps), cos[:, None, :], sin[:, None, :])
    value = _rms_norm(value, None, attention.v_norm.eps)
    return key.transpose(1, 0, 2)[None], value.transpose(1, 0, 2)[None]


def _attend(query: mx.array, key: mx.array, value: mx.array, chunks: Sequence[Chunk]) -> mx.array:
    """(1, heads, queries, head) queries over (1, key heads, rows, head) keys and values, one call per chunk;
    query head h reads key head ``h // (heads // key heads)``. Gemma 4 norms queries and keys, so its
    attention scale is 1. A chunk whose scores pass ``SCORES_AT_ONCE`` is computed before the next is built."""
    parts = []
    for start, stop, low, high, mask in chunks:
        parts.append(mx.fast.scaled_dot_product_attention(query[:, :, start:stop], key[:, :, low:high],
                                                          value[:, :, low:high], scale=1.0, mask=mask))
        if query.shape[1] * (stop - start) * (high - low) * query.dtype.size > SCORES_AT_ONCE:
            mx.eval(parts[-1])
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)


def _feedforward(layer: DecoderLayer, h: mx.array) -> mx.array:
    x = _rms_norm(h, layer.pre_feedforward_layernorm.weight, layer.pre_feedforward_layernorm.eps)
    gate = nn.gelu_approx(layer.mlp.gate_proj(x))
    mlp = layer.mlp.down_proj(gate * layer.mlp.up_proj(x))
    return _rms_norm(mlp, layer.post_feedforward_layernorm.weight, layer.post_feedforward_layernorm.eps)


def _layer(layer: DecoderLayer, h: mx.array, tables: tuple[mx.array, mx.array], chunks: Sequence[Chunk],
           markers: mx.array | None = None) -> mx.array:
    """One decoder layer over ``h`` (rows, hidden). With ``markers``, the queries, the residual and the MLP are
    those rows only (the last layer of a bidirectional model); the keys and values are still every row's."""
    attention = layer.self_attn
    cos, sin = tables
    source = _rms_norm(h, layer.input_layernorm.weight, layer.input_layernorm.eps)
    key, value = _keys_values(attention, source, cos, sin)
    if markers is not None:
        h, source, cos, sin = h[markers], source[markers], cos[markers], sin[markers]
    query = attention.q_proj(source).reshape(source.shape[0], attention.n_heads, attention.head_dim)
    query = _rotate(_rms_norm(query, attention.q_norm.weight, attention.q_norm.eps), cos[:, None, :], sin[:, None, :])
    attended = _attend(query.transpose(1, 0, 2)[None], key, value, chunks)
    attended = attention.o_proj(attended[0].transpose(1, 0, 2).reshape(query.shape[0], -1))
    residual = h + _rms_norm(attended, layer.post_attention_layernorm.weight, layer.post_attention_layernorm.eps)
    return (residual + _feedforward(layer, residual)) * layer.layer_scalar.astype(residual.dtype)


def _marker_states(model: Gemma4TextModel, sequences: Sequence[Encoded], window: int,
                   rope: dict[str, tuple[float, float]]) -> mx.array:
    """The final-normed hidden states at the readout rows of ``sequences`` packed end to end, (readout rows,
    hidden), in pack order. Sequences never read each other; every layer but the last runs on every row."""
    packed, plan = _plan_dict(sequences, window)
    chunks = {name: _chunks(mask, CHUNK) for name, mask in attention_masks(plan, window).items()}
    positions = mx.array(packed.seqpos.numpy())
    tokens = model.embed_tokens(mx.array(packed.input_ids.numpy()))
    h = tokens * mx.array(math.sqrt(tokens.shape[-1]), dtype=tokens.dtype)
    tables: dict[str, tuple[mx.array, mx.array]] = {}
    for layer in model.layers:  # one pair of tables per layer type, in the hidden state's type as upstream's
        if layer.layer_type not in tables:
            tables[layer.layer_type] = _rotary_tables(positions, layer.self_attn.head_dim, *rope[layer.layer_type],
                                                      h.dtype)
    for layer in model.layers[:-1]:
        h = _layer(layer, h, tables[layer.layer_type], chunks["sliding" if layer.layer_type == SLIDING else "global"])
    final = model.layers[-1]
    h = _layer(final, h, tables[final.layer_type], chunks["last"], mx.array(plan["marker_rows"]))
    return _rms_norm(h, model.norm.weight, model.norm.eps)


class MLXDecider:
    """Answers text-only requests from a folder written by ``scripts/convert_mlx.py``, as karpo-decision's
    ``Decider`` does from the original checkpoint.

    ``token_budget`` is the number of tokens per forward pass when several sequences are answered together; a
    longer sequence runs alone. ``max_length`` is the longest single sequence: a longer request is split into
    parts of its questions, each with the full state, and one question that cannot fit raises karpo-decision's
    ``LengthError``. Attention memory grows with the length, so a smaller ``max_length`` bounds it.

    The folder is checked before the text layers' weights are read, and those load all or nothing:
    ``ValueError`` for a configuration this module does not compute and for a folder that is incomplete.
    """

    def __init__(self, path: str | Path, token_budget: int = 4096, max_length: int = DEFAULT_MAX_LENGTH) -> None:
        self.path = Path(path)
        self.token_budget = token_budget
        self.max_length = max_length
        metadata = json.loads((self.path / "mlx_config.json").read_text())
        self.bits: int = metadata["bits"]
        text = metadata["text_config"]
        arguments = _arguments(text)
        self.window: int = arguments.sliding_window
        self.logit_cap: float | None = text.get("final_logit_softcapping")
        self.rope = _rope(text, arguments.layer_types)
        self.config = decision_config(self.path)
        tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        for code, token_id in zip(self.config.codes, self.config.token_ids, strict=True):
            if tokenizer.encode(code, add_special_tokens=False) != [token_id]:
                raise ValueError(f"code {code!r} is not token {token_id} of this tokenizer")
        self.encoder = Encoder(gemma.Processor(tokenizer, None), self.config.codes, template(self.config.template),
                               self.config.marker)
        readout = mx.load(str(self.path / "readout.safetensors")).get("weight")
        if readout is None or readout.shape != (len(self.config.codes), arguments.hidden_size):
            found = "no weight" if readout is None else f"a weight of {readout.shape}"
            raise ValueError(f"readout.safetensors holds {found}, not one of ({len(self.config.codes)} codes, "
                             f"{arguments.hidden_size} hidden)")
        self.readout = readout.astype(READOUT_DTYPES[self.config.readout_dtype])
        # Everything above is cheap; the weights are read only for a folder that passed.
        self.model = _backbone(text, self.bits, metadata["group_size"])
        _load_weights(self.model, self.path)
        self.model.eval()
        mx.eval(self.model.parameters())

    def decide(self, requests: Request | Sequence[Request]) -> dict[str, Any] | list[dict[str, Any]]:
        """One response per request (``{"answers": {...}, "usage": {...}}``), in input order: a single response
        for a single request, a list for a list, as karpo-decision's ``Decider.decide``. A request with images
        raises karpo-decision's ``TextOnlyError``."""
        if isinstance(requests, dict):
            return self.decide_many([requests])[0]
        return self.decide_many(list(requests))

    def decide_many(self, requests: Sequence[Request]) -> list[dict[str, Any]]:
        """One response per request, in input order. Every request is encoded before any forward pass, so one
        that cannot be answered fails the call instead of returning part of the answers."""
        plans = [self.encoder.plan(request, self.max_length) for request in requests]
        sequences = [part for parts in plans for part in parts]
        if not sequences:
            return []
        logits = self._packed_logits(sequences)
        results = []
        cursor = 0
        for request, parts in zip(requests, plans, strict=True):
            answers = {}
            for part in parts:
                for key, count in zip(part.keys, part.counts, strict=True):
                    question = request["questions"][key]
                    selected = logits[cursor, :count] / self.config.temperatures[question["type"]]
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

    def _packed_logits(self, sequences: Sequence[Encoded]) -> np.ndarray:
        """Readout logits of every sequence's questions, in order: consecutive sequences filling at most
        ``token_budget`` tokens share a forward pass (always at least one sequence), as ``Decider._packs``."""
        groups: list[list[Encoded]] = []
        current: list[Encoded] = []
        used = 0
        for sequence in sequences:
            if current and used + sequence.length > self.token_budget:
                groups.append(current)
                current = []
                used = 0
            current.append(sequence)
            used += sequence.length
        if current:
            groups.append(current)
        return np.concatenate([self._forward(group) for group in groups], axis=0)

    def _forward(self, sequences: Sequence[Encoded]) -> np.ndarray:
        """(readout rows, codes) float32 logits of one pack, soft-capped as the backbone's own head caps."""
        hidden = _marker_states(self.model.language_model, sequences, self.window, self.rope)
        logits = (hidden.astype(self.readout.dtype) @ self.readout.T).astype(mx.float32)
        if self.logit_cap:
            logits = self.logit_cap * mx.tanh(logits / self.logit_cap)
        mx.eval(logits)
        result = np.array(logits)
        # MLX keeps freed buffers for reuse, but requests of different lengths rarely reuse them, so the
        # cache would grow by gigabytes per long request until the system kills the process.
        del logits, hidden
        mx.clear_cache()
        return result


def load(path: str | Path, token_budget: int = 4096, max_length: int = DEFAULT_MAX_LENGTH) -> MLXDecider:
    """A decider over a folder written by ``scripts/convert_mlx.py``. ``token_budget``: tokens per forward
    pass when several requests are answered together. ``max_length``: the longest single sequence; a longer
    request is split into parts of its questions (``MLXDecider``)."""
    return MLXDecider(path, token_budget, max_length)
