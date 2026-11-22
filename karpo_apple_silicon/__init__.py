# Copyright 2026 the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Run Karpo Decision's Gemma-family checkpoints (12B, 31B) without an NVIDIA GPU.

karpo-decision's sliding-window layers call flash attention's variable-length kernel
(``torch.nn.attention.varlen.varlen_attn``), which exists only for CUDA. ``install()`` replaces that
one call (``karpo_decision.families.gemma.sliding_attention.attend``) with a dense, masked
scaled-dot-product attention that computes the same thing on any device.

The rule, for one virtual sequence with ``Lq`` queries and ``Lk`` keys and ``d = Lk - Lq``:

    query i reads key j  when  i + d - window <= j <= i + d

That is flash attention's ``window_size=(window, 0)``: causal, aligned at the END of each sequence
(the last query sits on the last key), each query reading itself and at most ``window`` keys before
it. karpo-decision passes ``window = sliding_window - 1`` (1,023 for Gemma 4), so a query reads
1,024 keys in all.

Nothing in site-packages is edited; the function is swapped at run time.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

__version__ = "0.1.0"
__all__ = ["TESTED_UPSTREAM", "attend", "install", "window_mask"]

#: The karpo-decision release this was written against and checked with.
TESTED_UPSTREAM = "0.1.0"


def window_mask(lq: int, lk: int, window: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """(lq, lk) boolean mask, True where query i may read key j: ``i + d - window <= j <= i + d``
    with ``d = lk - lq``."""
    i = torch.arange(lq, device=device)[:, None]
    j = torch.arange(lk, device=device)[None, :]
    diagonal = i + (lk - lq)
    return (j <= diagonal) & (j >= diagonal - window)


def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_cu: torch.Tensor, k_cu: torch.Tensor,
           max_q: int, max_k: int, scale: float, window: int) -> torch.Tensor:
    """Drop-in for ``karpo_decision.families.gemma.sliding_attention.attend``.

    ``query`` is (T, heads, D); ``key`` and ``value`` are (Tk, kv heads, D), with ``heads`` a multiple
    of ``kv heads`` (query head h reads key/value head ``h // (heads // kv heads)``). ``q_cu`` and
    ``k_cu`` are the cumulative boundaries of the packed sequences, one more entry than sequences.
    ``max_q`` and ``max_k`` are accepted for signature compatibility and not needed. Returns
    (T, heads, D).
    """
    q_bounds, k_bounds = q_cu.tolist(), k_cu.tolist()
    if len(q_bounds) != len(k_bounds):
        raise ValueError(f"{len(q_bounds) - 1} query sequences but {len(k_bounds) - 1} key sequences")
    if query.shape[1] % key.shape[1]:
        raise ValueError(f"{query.shape[1]} query heads are not a multiple of {key.shape[1]} key/value heads")
    repeat = query.shape[1] // key.shape[1]
    outs = []
    for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:]):
        lq, lk = qb - qa, kb - ka
        if lq == 0:
            continue
        if lk < lq:
            # The first lq - lk queries would have no key at all. karpo-decision never builds such a
            # sequence (a sequence's keys always include its own rows), so this is a caller error.
            raise ValueError(f"sliding attention: {lq} queries over {lk} keys leaves queries without any key")
        q = query[qa:qb].transpose(0, 1)[None]                                  # (1, heads, lq, D)
        k = key[ka:kb].transpose(0, 1).repeat_interleave(repeat, dim=0)[None]   # (1, heads, lk, D)
        v = value[ka:kb].transpose(0, 1).repeat_interleave(repeat, dim=0)[None]
        mask = window_mask(lq, lk, window, query.device)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        outs.append(out[0].transpose(0, 1))                                     # (lq, heads, D)
    if not outs:
        return query.new_empty(query.shape)
    return torch.cat(outs)


def install(force: bool = False) -> None:
    """Swap karpo-decision's sliding-window attention for the dense one in this process.

    Call it once, before ``Decider.decide``. On a machine with CUDA the original kernel works and is
    faster, so nothing is changed there unless ``force`` is true (useful for comparing the two).
    """
    from importlib.metadata import PackageNotFoundError, version

    from karpo_decision.families.gemma import sliding_attention

    try:
        found = version("karpo-decision")
    except PackageNotFoundError:  # an editable or vendored copy: nothing to compare
        found = None
    if found is not None and found != TESTED_UPSTREAM:
        warnings.warn(f"karpo-apple-silicon was checked against karpo-decision {TESTED_UPSTREAM}; "
                      f"{found} is installed and its attention rule may differ.", stacklevel=2)
    if not callable(getattr(sliding_attention, "attend", None)):
        raise RuntimeError("karpo_decision.families.gemma.sliding_attention has no attend(); "
                           "this karpo-decision version is not supported")
    if torch.cuda.is_available() and not force:
        return
    sliding_attention.attend = attend
