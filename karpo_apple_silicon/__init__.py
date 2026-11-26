# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Run Karpo Decision's Gemma-family checkpoints (12B, 31B) without an NVIDIA GPU.

karpo-decision's sliding-window layers call flash attention's variable-length kernel
(``torch.nn.attention.varlen.varlen_attn``), which exists only for CUDA. ``install()`` replaces that
call (``karpo_decision.families.gemma.sliding_attention.attend``) with masked scaled-dot-product
attention that computes the same thing on any device. Queries are answered ``CHUNK`` at a time, each
chunk reading only the keys its window can reach, so a long sequence costs its length times the
window and not its length squared.

On a machine without CUDA, ``install()`` also replaces the global layers' attention
(``karpo_decision.families.gemma.packed.Packed._global``). There karpo-decision makes one call over a
whole sequence, whose score table is heads x length x length, inside a kernel selection that leaves a
CPU with no kernel at all. ``global_attend`` makes the same call ``CHUNK`` query rows at a time, so
no table grows with the square of the length.

The rule, for one virtual sequence with ``Lq`` queries and ``Lk`` keys and ``d = Lk - Lq``:

    query i reads key j  when  i + d - window <= j <= i + d

That is flash attention's ``window_size=(window, 0)``: causal, aligned at the END of each sequence
(the last query sits on the last key), each query reading itself and at most ``window`` keys before
it. karpo-decision passes ``window = sliding_window - 1`` (1,023 for Gemma 4), so a query reads
1,024 keys in all.

Nothing in site-packages is edited; the two names are swapped at run time, and ``uninstall()`` puts
them back. On a machine with CUDA nothing is swapped unless forced, and the global layers never are.
"""

from __future__ import annotations

import inspect
import json
import warnings
from importlib.metadata import PackageNotFoundError, distribution, version
from typing import Any

import torch
import torch.nn.functional as F

__version__ = "0.2.0"
__all__ = ["CHUNK", "TESTED_UPSTREAM", "TESTED_UPSTREAM_COMMIT", "TESTED_UPSTREAM_COMMITS", "attend", "global_attend",
           "install", "uninstall", "window_mask"]

#: The karpo-decision release this was written against and checked with.
TESTED_UPSTREAM = "0.1.0"

#: The karpo-decision commits this was checked with, newest first. Upstream's version has been 0.1.0 at every
#: commit so far, so the commit is what ``install()`` compares when pip recorded one. The two have the same
#: source files: the newer one changes only licence and Docker text.
TESTED_UPSTREAM_COMMITS = ("c52d3507a254d3ad8423726c6bc6568c99113e7d", "6600011eb5cf46232d056ec0717dc726146d29de")

#: The commit the install instructions name.
TESTED_UPSTREAM_COMMIT = TESTED_UPSTREAM_COMMITS[0]

#: Queries answered per attention call. In a sliding-window layer a chunk reads at most ``CHUNK + window``
#: keys, where one call per sequence would read them all. On Apple GPUs, times were flat from 128 to 256 and
#: worse at 64 and 512. In a global layer a chunk reads all of its sequence's keys, or, when causal, the keys up
#: to its last row's own.
CHUNK = 256

#: What ``install()`` replaced in this process, by name, for ``uninstall()``.
_originals: dict[str, Any] = {}


def window_mask(lq: int, lk: int, window: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """(lq, lk) boolean mask, True where query i may read key j: ``i + d - window <= j <= i + d``
    with ``d = lk - lq``.

    ``window`` is flash attention's ``window_size``, which is the model's ``sliding_window`` minus one, so a
    query reads up to ``window + 1`` keys; ``mlx.attention_masks`` takes ``sliding_window`` itself as its
    ``window``, so a query there reads up to ``window`` keys.
    """
    i = torch.arange(lq, device=device)[:, None]
    j = torch.arange(lk, device=device)[None, :]
    diagonal = i + (lk - lq)
    return (j <= diagonal) & (j >= diagonal - window)


def _check_sequences_and_heads(q_bounds: list[int], k_bounds: list[int], heads: int, kv_heads: int) -> None:
    """``ValueError`` unless the query and key boundaries mark the same number of sequences and the query heads
    are a multiple of the key/value heads. The two functions here and their oracles in ``reference`` all call
    it, so they refuse the same input in the same words."""
    if len(q_bounds) != len(k_bounds):
        raise ValueError(f"{len(q_bounds) - 1} query sequences but {len(k_bounds) - 1} key sequences")
    if heads % kv_heads:
        raise ValueError(f"{heads} query heads are not a multiple of {kv_heads} key/value heads")


def attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_cu: torch.Tensor, k_cu: torch.Tensor,
           max_q: int, max_k: int, scale: float, window: int) -> torch.Tensor:
    """Drop-in for ``karpo_decision.families.gemma.sliding_attention.attend``.

    ``query`` is (T, heads, D); ``key`` and ``value`` are (Tk, kv heads, D), with ``heads`` a multiple
    of ``kv heads`` (query head h reads key/value head ``h // (heads // kv heads)``). ``q_cu`` and
    ``k_cu`` are the cumulative boundaries of the packed sequences, one more entry than sequences.
    ``max_q`` and ``max_k`` are accepted for signature compatibility and not needed. Returns
    (T, heads, D).

    Each sequence's queries are answered ``CHUNK`` at a time over only the keys that chunk can reach,
    which is the same attention as one full mask per sequence (``reference.dense_attend``) with the
    keys every query would discard left out.
    """
    q_bounds, k_bounds = q_cu.tolist(), k_cu.tolist()
    _check_sequences_and_heads(q_bounds, k_bounds, query.shape[1], key.shape[1])
    enable_gqa = key.shape[1] != query.shape[1]
    outs = []
    for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:], strict=True):
        lq, lk = qb - qa, kb - ka
        if lq == 0:
            continue
        if lk < lq:
            # The first lq - lk queries would have no key at all. karpo-decision never builds such a
            # sequence (a sequence's keys always include its own rows), so this is a caller error.
            raise ValueError(f"sliding attention: {lq} queries over {lk} keys leaves queries without any key")
        q = query[qa:qb].transpose(0, 1)[None]                                  # (1, heads, lq, D)
        k = key[ka:kb].transpose(0, 1)[None]                                    # (1, kv heads, lk, D)
        v = value[ka:kb].transpose(0, 1)[None]
        shift = lk - lq
        for a in range(0, lq, CHUNK):
            b = min(a + CHUNK, lq)
            lo = max(0, a + shift - window)
            hi = b + shift
            # The slice ends on the chunk's last readable key, so the same end-aligned rule holds on it:
            # its shift is (hi - lo) - (b - a) = a + shift - lo.
            mask = window_mask(b - a, hi - lo, window, query.device)
            out = F.scaled_dot_product_attention(q[:, :, a:b], k[:, :, lo:hi], v[:, :, lo:hi], attn_mask=mask,
                                                 scale=scale, enable_gqa=enable_gqa)
            outs.append(out[0].transpose(0, 1))                                 # (b - a, heads, D)
    if not outs:
        return query.new_empty(query.shape)
    return torch.cat(outs)


def global_attend(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, q_bounds: list[int],
                  k_bounds: list[int], lens: torch.Tensor | None, causal: bool, scale: float) -> torch.Tensor:
    """Drop-in for ``karpo_decision.families.gemma.packed.Packed._global`` on a device without CUDA.

    ``query`` is (1, T, heads, D); ``key`` and ``value`` are (1, Tk, kv heads, D), with ``heads`` a multiple
    of ``kv heads``. Virtual sequence ``n`` has query rows ``[q_bounds[n], q_bounds[n + 1])`` and key rows
    ``[k_bounds[n], k_bounds[n + 1])``. A query reads every key of its sequence, or, when ``causal``, the
    keys up to its own, the last query sitting on the last key. ``lens`` is accepted for signature
    compatibility and not needed. Returns (query rows, heads x D).

    One difference from the original: a causal sequence with more queries than keys, whose first queries
    have no key to read, is a ``ValueError`` here. karpo-decision never builds one (a sequence's keys
    include its own rows), and the original answers such rows without an error.

    karpo-decision makes one call per sequence here. Its score table is heads x queries x keys, which
    for a long request is more than a Mac's memory holds; the 12B's answers to a 23,627-token request
    came back as noise that way on a 96 GB Mac. This makes the same call ``CHUNK`` query rows at a time,
    which is the same attention (``reference.dense_global``) with a table that stays small.
    """
    _check_sequences_and_heads(q_bounds, k_bounds, query.shape[2], key.shape[2])
    enable_gqa = key.shape[2] != query.shape[2]
    outs = []
    for qa, qb, ka, kb in zip(q_bounds[:-1], q_bounds[1:], k_bounds[:-1], k_bounds[1:], strict=True):
        lq, lk = qb - qa, kb - ka
        if lq == 0:
            continue
        if causal and lk < lq:
            raise ValueError(f"global attention: {lq} causal queries over {lk} keys leaves queries without any key")
        q = query[:, qa:qb].transpose(1, 2)                                     # (1, heads, lq, D)
        k = key[:, ka:kb].transpose(1, 2)                                       # (1, kv heads, lk, D)
        v = value[:, ka:kb].transpose(1, 2)
        for a in range(0, lq, CHUNK):
            b = min(a + CHUNK, lq)
            if causal:
                # Rows a..b read keys up to their own, so nothing past the chunk's last readable key is
                # needed, and on that slice the end-aligned rule is a window that never clips.
                hi = b + lk - lq
                out = F.scaled_dot_product_attention(q[:, :, a:b], k[:, :, :hi], v[:, :, :hi],
                                                     attn_mask=window_mask(b - a, hi, hi, query.device),
                                                     scale=scale, enable_gqa=enable_gqa)
            else:
                out = F.scaled_dot_product_attention(q[:, :, a:b], k, v, scale=scale, enable_gqa=enable_gqa)
            outs.append(out[0].transpose(0, 1))                                 # (b - a, heads, D)
    if not outs:
        return query.new_empty((0, query.shape[2] * query.shape[3]))
    out = torch.cat(outs)
    return out.reshape(out.shape[0], -1)


#: The parameter names ``install()`` requires of the two functions it replaces: the replacements' own, in order.
_ATTEND_PARAMETERS = tuple(inspect.signature(attend).parameters)
_GLOBAL_PARAMETERS = tuple(inspect.signature(global_attend).parameters)


def _installed_commit() -> str | None:
    """The git commit karpo-decision was installed from, when pip recorded one (``pip install`` from a
    git URL does; a local or editable install does not)."""
    try:
        direct_url = distribution("karpo-decision").read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id") if direct_url else None


def install(force: bool = False) -> None:
    """Swap karpo-decision's sliding-window attention for ``attend`` in this process and, on a machine
    without CUDA, its global layers' attention for ``global_attend``.

    Call it once, before ``Decider.decide``. On a machine with CUDA the original kernels work and are
    faster, so nothing is changed there unless ``force`` is true (useful for comparing the two); the
    global layers are never changed there. Calling it again changes nothing.

    Warns when karpo-decision is not a commit this was checked with, and raises ``RuntimeError``, before
    anything is swapped, when a function to be replaced no longer takes the arguments its replacement
    was written for.
    """
    try:
        from karpo_decision.families.gemma import packed, sliding_attention
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("karpo_decision"):
            raise ModuleNotFoundError(
                "karpo-decision is installed separately: pip install \"karpo-decision @ "
                f"git+https://github.com/machinepulse-ai/karpo-decision@{TESTED_UPSTREAM_COMMIT}\""
            ) from error
        raise

    try:
        found = version("karpo-decision")
    except PackageNotFoundError:  # a source tree not installed as a distribution: no version to compare
        found = None
    commit = _installed_commit()
    if commit is not None and commit not in TESTED_UPSTREAM_COMMITS:
        warnings.warn(f"karpo-apple-silicon was checked against karpo-decision commit "
                      f"{TESTED_UPSTREAM_COMMIT[:12]}; {commit[:12]} is installed and its attention rule may differ.",
                      stacklevel=2)
    elif commit is None and found is not None and found != TESTED_UPSTREAM:
        warnings.warn(f"karpo-apple-silicon was checked against karpo-decision {TESTED_UPSTREAM}; "
                      f"{found} is installed and its attention rule may differ.", stacklevel=2)
    original_attend = _originals.get("attend", getattr(sliding_attention, "attend", None))
    if not callable(original_attend):
        raise RuntimeError("karpo_decision.families.gemma.sliding_attention has no attend(); "
                           "this karpo-decision version is not supported")
    if tuple(inspect.signature(original_attend).parameters) != _ATTEND_PARAMETERS:
        raise RuntimeError("this karpo-decision is not supported: attend() has a different signature")
    has_cuda = torch.cuda.is_available()
    if has_cuda and not force:
        return
    original_global = None
    if not has_cuda:
        # The class attribute itself (a staticmethod), so that uninstall() puts back the same object.
        packed_class = getattr(packed, "Packed", None)
        class_attribute = None if packed_class is None else vars(packed_class).get("_global")
        original_global = _originals.get("global", class_attribute)
        if not isinstance(original_global, staticmethod):
            raise RuntimeError("karpo_decision.families.gemma.packed.Packed has no static _global(); "
                               "this karpo-decision version is not supported")
        if tuple(inspect.signature(original_global.__func__).parameters) != _GLOBAL_PARAMETERS:
            raise RuntimeError("this karpo-decision is not supported: Packed._global() has a different signature")
    _originals.setdefault("attend", original_attend)
    sliding_attention.attend = attend
    if original_global is not None:
        _originals.setdefault("global", original_global)
        packed.Packed._global = staticmethod(global_attend)


def uninstall() -> None:
    """Put back what ``install()`` replaced in this process. Does nothing if nothing is installed."""
    if not _originals:
        return
    from karpo_decision.families.gemma import packed, sliding_attention

    if "attend" in _originals:
        sliding_attention.attend = _originals["attend"]
    if "global" in _originals:
        packed.Packed._global = _originals["global"]
    _originals.clear()
