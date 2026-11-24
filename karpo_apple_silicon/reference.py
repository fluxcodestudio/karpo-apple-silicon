# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Dense statement of the attention rule, independent of the banded implementation.

This slower, quadratic-memory oracle tests ``attend`` and supports the comparison script's
wrong-on-purpose controls. ``install()`` never uses it.
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
