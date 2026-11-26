# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Dense statements of the two attention rules, independent of the chunked implementations.

These slower, quadratic-memory oracles test ``attend`` and ``global_attend`` and support the
wrong-on-purpose controls of the comparison script and the upstream suite check. ``install()`` never
uses them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from karpo_apple_silicon import window_mask


def dense_attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_cu: torch.Tensor, k_cu: torch.Tensor,
                 max_q: int, max_k: int, scale: float, window: int, *, mask=window_mask) -> torch.Tensor:
    """Apply one full end-aligned window mask per packed sequence as a plain oracle for ``attend``."""
    q_bounds, k_bounds = q_cu.tolist(), k_cu.tolist()
    if len(q_bounds) != len(k_bounds):
        raise ValueError(f"{len(q_bounds) - 1} query sequences but {len(k_bounds) - 1} key sequences")
    if query.shape[1] % key.shape[1]:
        raise ValueError(f"{query.shape[1]} query heads are not a multiple of {key.shape[1]} key/value heads")
    outs = []
    for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:], strict=True):
        lq, lk = qb - qa, kb - ka
        if lq == 0:
            continue
        if lk < lq:
            raise ValueError(f"sliding attention: {lq} queries over {lk} keys leaves queries without any key")
        q = query[qa:qb].transpose(0, 1)[None]
        k = key[ka:kb].transpose(0, 1)[None]
        v = value[ka:kb].transpose(0, 1)[None]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask(lq, lk, window, query.device), scale=scale,
                                             enable_gqa=key.shape[1] != query.shape[1])
        outs.append(out[0].transpose(0, 1))
    if not outs:
        return query.new_empty(query.shape)
    return torch.cat(outs)


# Wrong-on-purpose controls for the comparison script and upstream differential checks.
def wrong_no_window(lq: int, lk: int, window: int, device: torch.device | str) -> torch.Tensor:
    """Use an effectively unbounded causal window."""
    return window_mask(lq, lk, lq + lk, device)


def wrong_top_left(lq: int, lk: int, window: int, device: torch.device | str) -> torch.Tensor:
    """Align the causal diagonal at the start rather than the end."""
    i = torch.arange(lq, device=device)[:, None]
    j = torch.arange(lk, device=device)[None, :]
    return (j <= i) & (j >= i - window)


def wrong_window_minus_one(lq: int, lk: int, window: int, device: torch.device | str) -> torch.Tensor:
    """Omit one key from the permitted window."""
    return window_mask(lq, lk, window - 1, device)


def wrong_window_plus_one(lq: int, lk: int, window: int, device: torch.device | str) -> torch.Tensor:
    """Add one key to the permitted window."""
    return window_mask(lq, lk, window + 1, device)


def dense_global(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_bounds: list[int],
                 k_bounds: list[int], lens: torch.Tensor | None, causal: bool, scale: float, *,
                 hidden: int = 0) -> torch.Tensor:
    """One call per sequence over all of its keys, as karpo-decision makes it without CUDA: a plain oracle
    for ``global_attend``. ``hidden`` keys at the end of each sequence are left out (wrong on purpose)."""
    outs = []
    for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:], strict=True):
        lq, lk = qb - qa, kb - ka - hidden
        if lq == 0:
            continue
        q = query[:, qa:qb].transpose(1, 2)
        k = key[:, ka:ka + lk].transpose(1, 2)
        v = value[:, ka:ka + lk].transpose(1, 2)
        mask = window_mask(lq, lk, lk, query.device) if causal else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale,
                                             enable_gqa=key.shape[2] != query.shape[2])
        outs.append(out[0].transpose(0, 1))
    if not outs:
        return query.new_empty((0, query.shape[2] * query.shape[3]))
    out = torch.cat(outs)
    return out.reshape(out.shape[0], -1)


def wrong_global_last_key_hidden(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_bounds: list[int],
                                 k_bounds: list[int], lens: torch.Tensor | None, causal: bool,
                                 scale: float) -> torch.Tensor:
    """Hide each sequence's last key from its queries."""
    return dense_global(query, key, value, q_bounds, k_bounds, lens, causal, scale, hidden=1)
