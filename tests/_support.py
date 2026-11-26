# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""What the test modules share. Not a test module itself: unittest collects ``test*.py`` only. torch, MLX and
karpo-decision are imported inside the functions, so a module whose tests are skipped can still import this."""

from __future__ import annotations

import contextlib
import importlib.util
import os
import unittest
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    from karpo_decision.encoding import Encoded


def mlx_unavailable(*also: str) -> str | None:
    """Why the MLX path cannot be tested here, or None: mlx, mlx-lm, karpo-decision or a module of ``also`` is
    missing, or MLX cannot start."""
    for name in ("mlx", "mlx_lm", "karpo_decision", *also):
        try:
            found = importlib.util.find_spec(name)
        except ModuleNotFoundError:  # a dotted name whose parent package is missing
            found = None
        if found is None:
            return f"needs {name}"
    try:
        import mlx.core  # noqa: F401
    except Exception as error:  # MLX raises on import when there is no Metal device
        return f"MLX cannot start: {error}"
    return None


def needs_apple_gpu(test):
    """Skip a test or a test class where PyTorch has no Apple GPU (MPS), and where ``KARPO_TEST_NO_MPS=1`` asks
    for these tests to be left out, as on a Mac whose GPU is busy. It does not cover the MLX tests, which reach
    the GPU through MLX and not through PyTorch."""
    import torch

    if os.environ.get("KARPO_TEST_NO_MPS") == "1":
        return unittest.skip("KARPO_TEST_NO_MPS=1")(test)
    return unittest.skipUnless(torch.backends.mps.is_available(), "MPS is not available")(test)


def sequence(shared: int, branches: Sequence[int]) -> Encoded:
    """One sequence in karpo-decision's block layout with random token ids: a block of ``shared`` tokens, or a
    shared block and one branch per entry of ``branches``. A branch continues the shared block's positions and
    ends on its readout marker; without branches the block's last token is the marker."""
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


@contextlib.contextmanager
def attention_calls() -> Iterator[list[tuple[int, int]]]:
    """Record (query rows, key rows) of every ``scaled_dot_product_attention`` call made through the package's
    ``F`` while the block runs; each call still goes to the real function. ``reference`` calls through the same
    ``F``, so compute an oracle outside the block."""
    import karpo_apple_silicon

    real = karpo_apple_silicon.F.scaled_dot_product_attention
    calls: list[tuple[int, int]] = []

    def recorded(query, key, *arguments, **options):
        calls.append((query.shape[-2], key.shape[-2]))
        return real(query, key, *arguments, **options)

    with mock.patch.object(karpo_apple_silicon.F, "scaled_dot_product_attention", recorded):
        yield calls
