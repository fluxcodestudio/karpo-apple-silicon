# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Replace upstream's sliding-window and global attention during its CPU tests and count calls."""

from __future__ import annotations

import os
from functools import partial
from unittest.mock import patch

import pytest
from karpo_decision.families.gemma import packed, sliding_attention

import karpo_apple_silicon
from karpo_apple_silicon.reference import (
    dense_attend,
    wrong_global_last_key_hidden,
    wrong_no_window,
    wrong_top_left,
    wrong_window_minus_one,
    wrong_window_plus_one,
)

VARIANTS = {"top-left": wrong_top_left, "window-1": wrong_window_minus_one,
            "window+1": wrong_window_plus_one, "no-window": wrong_no_window}
GLOBAL_VARIANTS = {"global-1": wrong_global_last_key_hidden}
variant = os.environ.get("KARPO_PORT_VARIANT", "port")
if variant not in ("port", *VARIANTS, *GLOBAL_VARIANTS):
    raise ValueError(f"unknown KARPO_PORT_VARIANT: {variant}")
attend_calls = 0
global_calls = 0


@pytest.fixture(autouse=True)
def use_port(monkeypatch):
    """Exercise short upstream sequences across the band and chunk boundaries. A wrong-on-purpose variant
    changes one of the two functions and leaves the port's own in the other."""
    implementation = (partial(dense_attend, mask=VARIANTS[variant]) if variant in VARIANTS
                      else karpo_apple_silicon.attend)
    whole = GLOBAL_VARIANTS.get(variant, karpo_apple_silicon.global_attend)

    def counted(*args, **kwargs):
        global attend_calls
        attend_calls += 1
        return implementation(*args, **kwargs)

    def counted_global(*args, **kwargs):
        global global_calls
        global_calls += 1
        return whole(*args, **kwargs)

    monkeypatch.setattr(sliding_attention, "attend", counted)
    monkeypatch.setattr(packed.Packed, "_global", staticmethod(counted_global))
    with patch.object(karpo_apple_silicon, "CHUNK", 3):
        yield


def pytest_sessionfinish(session, exitstatus):
    """Show that the tests reached the replacements rather than only upstream code."""
    print(f"variant={variant} attend calls={attend_calls} global calls={global_calls}")
