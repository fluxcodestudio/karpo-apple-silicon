# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Replace upstream's attention kernel during its CPU tests and count calls."""

from __future__ import annotations

import os
from functools import partial
from unittest.mock import patch

import pytest
from karpo_decision.families.gemma import sliding_attention

import karpo_apple_silicon
from karpo_apple_silicon.reference import (
    dense_attend,
    wrong_no_window,
    wrong_top_left,
    wrong_window_minus_one,
    wrong_window_plus_one,
)

VARIANTS = {"top-left": wrong_top_left, "window-1": wrong_window_minus_one,
            "window+1": wrong_window_plus_one, "no-window": wrong_no_window}
variant = os.environ.get("KARPO_PORT_VARIANT", "port")
if variant not in ("port", *VARIANTS):
    raise ValueError(f"unknown KARPO_PORT_VARIANT: {variant}")
attend_calls = 0


@pytest.fixture(autouse=True)
def use_port(monkeypatch):
    """Exercise short upstream sequences across the band boundary."""
    global attend_calls

    implementation = (karpo_apple_silicon.attend if variant == "port"
                      else partial(dense_attend, mask=VARIANTS[variant]))

    def counted(*args, **kwargs):
        global attend_calls
        attend_calls += 1
        return implementation(*args, **kwargs)

    monkeypatch.setattr(sliding_attention, "attend", counted)
    if variant == "port":
        with patch.object(karpo_apple_silicon, "CHUNK", 3):
            yield
    else:
        yield


def pytest_sessionfinish(session, exitstatus):
    """Show that the tests reached the replacement rather than only upstream code."""
    print(f"variant={variant} attend calls={attend_calls}")
