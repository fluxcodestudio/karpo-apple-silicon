# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Time a checkpoint on this Mac: one request at a time and in batches, in bfloat16 and float16.

    python scripts/bench.py path/to/karpo-decision-31B

The requests are invented product-support notes of roughly 200 tokens each. Close other heavy programs
first; timings on a busy machine are not a hardware limit.
"""

from __future__ import annotations

import argparse
import random
import statistics
import subprocess
import time

import torch

import karpo_apple_silicon

PRODUCTS = ("blender", "desk lamp", "kettle", "bicycle helmet", "garden hose", "wireless mouse", "tent", "toaster")
EVENTS = ("arrived with a cracked casing", "stopped working after two weeks", "was delivered to the wrong address",
          "is missing its power cable", "works but is louder than expected", "was charged twice on the card",
          "came in the wrong colour", "has not arrived after twelve days")
WISHES = ("asks for a refund", "asks for a replacement", "asks how to repair it", "asks to cancel the order",
          "asks for a discount on the next order", "asks to speak to a supervisor")
QUESTIONS = {"refund": {"type": "noul", "instructions": "Is the customer asking for their money back?"}}


def requests(count: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for number in range(count):
        notes = [f"Order {1000 + number}: the {rng.choice(PRODUCTS)} {rng.choice(EVENTS)}. The customer "
                 f"{rng.choice(WISHES)}." for _ in range(1)]
        notes += [f"Earlier contact {i + 1}: about a {rng.choice(PRODUCTS)} that {rng.choice(EVENTS)}; the customer "
                  f"{rng.choice(WISHES)[5:]} was noted and the case was closed after {rng.randint(1, 9)} days."
                  for i in range(4)]
        out.append({"state": " ".join(notes), "questions": QUESTIONS})
    return out


def timed(fn):
    torch.mps.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.mps.synchronize()
    return result, time.perf_counter() - started


def measure(decider, batch_of: list[dict], sizes: tuple[int, ...]) -> None:
    decider.decide(batch_of[0])  # warm-up
    singles = [timed(lambda r=r: decider.decide(r)) for r in batch_of[:12]]
    tokens = statistics.median(out["usage"]["input_tokens"] for out, _ in singles)
    print(f"  one at a time: median {1000 * statistics.median(t for _, t in singles):.0f} ms per request "
          f"(median {tokens:.0f} tokens)", flush=True)
    for size in sizes:
        groups = [batch_of[i:i + size] for i in range(0, 32, size)]
        total = sum(timed(lambda g=g: decider.decide(g))[1] for g in groups)
        print(f"  batches of {size}: {1000 * total / 32:.0f} ms per request", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    karpo_apple_silicon.install()
    from karpo_decision.decider import Decider

    print("load average:", subprocess.check_output(["sysctl", "-n", "vm.loadavg"], text=True).strip())
    started = time.perf_counter()
    decider = Decider.from_pretrained(args.checkpoint, device="mps")
    print(f"loaded in {time.perf_counter() - started:.0f} s")
    batch = requests(32)
    print("bfloat16 (as shipped)")
    measure(decider, batch, (4, 16))
    decider.ckpt.backbone.to(torch.float16)
    print("float16")
    measure(decider, batch, (16,))


if __name__ == "__main__":
    main()
