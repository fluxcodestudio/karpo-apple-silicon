# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Compare this port with MachinePulse's published per-request results for the Decision Index public suite.

You need three things, none of which ship with this repository:

  --checkpoint   a Karpo Decision Gemma-family checkpoint folder (for example karpo-decision-31B)
  --reference    results.jsonl.gz from the model's Decision Index run, published by MachinePulse at
                 huggingface.co/datasets/MachinePulseAI/karpo-decision-31B-decision-index
  --rows         suite rows (selected-rows.jsonl or .jsonl.gz) rebuilt with the Decision Index kit,
                 github.com/apolinario/decision-index (the suite is not redistributable; a partial rebuild
                 with ``suite rebuild --edition 0.1 --only ...`` is enough)

Requests are joined on run id, their payload hash must match the published one, and the token count the
port reports must equal the published ``usage.input_tokens`` before any probability is compared. Requests
are run one at a time, as the published run did. Only aggregates are written to --out; a file beside it
(--out plus .progress.jsonl: ids and differences, no request text) lets a stopped run resume. Delete it to
start again, and always after changing the code or the checkpoint.

    python scripts/compare_reference.py --checkpoint CKPT --reference results.jsonl.gz \
        --rows selected-rows.jsonl.gz --out comparison.json

``--variants`` runs more passes over the same sample in one process (one model load):
  port        the port as shipped, in the checkpoint's bfloat16
  no-window   wrong on purpose: causal, but the window never clips
  top-left    wrong on purpose: the causal diagonal aligned at each sequence's start instead of its end
  window-1    wrong on purpose: one key fewer in the window
  float16     the port with the backbone converted to float16 (run it last: the conversion is not undone)
The wrong-on-purpose passes show how large a difference this comparison would report for a wrong mask.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

import karpo_apple_silicon

VARIANTS = ("port", "no-window", "top-left", "window-1", "float16")
CELLS = ("single-short", "single-long", "multi-short", "multi-long")


def read_jsonl(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def cell_of(response) -> str:
    questions = "multi" if len(response["answers"]) > 1 else "single"
    return f"{questions}-{'long' if response['usage']['input_tokens'] > 1024 else 'short'}"


def select(rows_path, reference_path, per_cell, max_tokens, seed):
    """A deterministic sample: per cell, tracks in turn, requests in hash order."""
    reference = {}
    for record in read_jsonl(reference_path):
        if record["status"] == "ok" and record["response"]["usage"]["sequences"] == 1:
            reference[record["run_id"]] = record
    pools = defaultdict(lambda: defaultdict(list))
    seen = mismatched = 0
    for row in read_jsonl(rows_path):
        meta = row["_evaluation"]
        record = reference.get(meta["run_id"])
        if record is None:
            continue
        seen += 1
        if record["payload_sha256"] != meta["payload_sha256"]:
            mismatched += 1
            continue
        tokens = record["response"]["usage"]["input_tokens"]
        if tokens > max_tokens:
            continue
        order = hashlib.sha256(f"{seed}:{meta['run_id']}".encode()).hexdigest()
        pools[cell_of(record["response"])][meta["track"]].append(
            (order, meta["run_id"], {"state": row["state"], "questions": row["questions"]}))
    if mismatched:
        print(f"warning: {mismatched} of {seen} joined rows have a payload hash that differs from the published "
              "one and were dropped", flush=True)
    sample = []
    for cell in CELLS:
        tracks = {track: sorted(items) for track, items in sorted(pools[cell].items())}
        wanted = per_cell[cell]
        while wanted > 0 and any(tracks.values()):
            for track in list(tracks):
                if tracks[track] and wanted > 0:
                    _, run_id, request = tracks[track].pop(0)
                    sample.append((cell, track, run_id, request, reference[run_id]["response"]))
                    wanted -= 1
    return sample


def probabilities(answer) -> dict[str, float]:
    if answer["type"] == "noul":
        return {"true": answer["noul"], "false": 1.0 - answer["noul"]}
    return answer["probabilities"]


def top(answer) -> str:
    return ("true" if answer["noul"] >= 0.5 else "false") if answer["type"] == "noul" else answer["choice"]


def set_variant(name: str, decider) -> None:
    original = karpo_apple_silicon._original_window_mask
    if name in ("port", "float16"):
        karpo_apple_silicon.window_mask = original
    elif name == "no-window":
        karpo_apple_silicon.window_mask = lambda lq, lk, window, device="cpu": original(lq, lk, lq + lk, device)
    elif name == "window-1":
        karpo_apple_silicon.window_mask = lambda lq, lk, window, device="cpu": original(lq, lk, window - 1, device)
    elif name == "top-left":
        def top_left(lq, lk, window, device="cpu"):
            i = torch.arange(lq, device=device)[:, None]
            j = torch.arange(lk, device=device)[None, :]
            return (j <= i) & (j >= i - window)
        karpo_apple_silicon.window_mask = top_left
    if name == "float16":
        decider.ckpt.backbone.to(torch.float16)


def run(decider, sample, synchronize, progress_path, variant):
    """Answer ``sample`` and summarize. Each finished request is appended to ``progress_path`` (ids, token
    counts and differences only, never request text), and requests already there are not run again, so a
    stopped run resumes."""
    done = {}
    if progress_path.exists():
        for record in read_jsonl(progress_path):
            if record["variant"] == variant:
                done[record["run_id"]] = record
    if done:
        print(f"  resuming: {len(done)} requests already done", flush=True)
    stats = defaultdict(lambda: {"requests": 0, "questions": 0, "token_mismatches": 0, "top_changed": 0, "diffs": [],
                                 "tokens": [], "seconds": []})
    with progress_path.open("a", encoding="utf-8") as progress:
        for index, (cell, track, run_id, request, published) in enumerate(sample):
            record = done.get(run_id)
            if record is None:
                synchronize()
                started = time.perf_counter()
                response = decider.decide(request)
                synchronize()
                record = {"variant": variant, "run_id": run_id, "seconds": time.perf_counter() - started,
                          "tokens": response["usage"]["input_tokens"], "diffs": None, "top_changed": 0}
                if record["tokens"] == published["usage"]["input_tokens"]:
                    record["diffs"] = []
                    for name, answer in response["answers"].items():
                        theirs = published["answers"][name]
                        ours, ref = probabilities(answer), probabilities(theirs)
                        record["diffs"].append(max(abs(ours[option] - ref[option]) for option in ref))
                        record["top_changed"] += top(answer) != top(theirs)
                progress.write(json.dumps(record) + "\n")
                progress.flush()
            for key in (cell, f"track:{track}", "all"):
                entry = stats[key]
                entry["requests"] += 1
                entry["tokens"].append(record["tokens"])
                entry["seconds"].append(record["seconds"])
                if record["diffs"] is None:
                    entry["token_mismatches"] += 1
                    continue
                entry["questions"] += len(record["diffs"])
                entry["diffs"] += record["diffs"]
                entry["top_changed"] += record["top_changed"]
            if (index + 1) % 20 == 0:
                print(f"  {index + 1}/{len(sample)}", flush=True)
    summary = {}
    for key, entry in sorted(stats.items()):
        diffs = sorted(entry["diffs"])
        summary[key] = {
            "requests": entry["requests"], "questions": entry["questions"],
            "token_mismatches": entry["token_mismatches"], "top_answers_changed": entry["top_changed"],
            "max_abs_probability_difference": max(diffs) if diffs else None,
            "mean_abs_probability_difference": statistics.fmean(diffs) if diffs else None,
            "p99_abs_probability_difference": diffs[min(len(diffs) - 1, int(0.99 * len(diffs)))] if diffs else None,
            "questions_differing_by_more_than_0.05": sum(d > 0.05 for d in diffs),
            "median_input_tokens": statistics.median(entry["tokens"]), "max_input_tokens": max(entry["tokens"]),
            "median_seconds_per_request": statistics.median(entry["seconds"]),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--backend", choices=("torch", "mlx"), default="torch",
                        help="torch: karpo-decision with this port's patch; mlx: karpo_apple_silicon.mlx over a "
                             "folder written by scripts/convert_mlx.py")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--variants", default="port", help=f"comma-separated, from: {', '.join(VARIANTS)}")
    parser.add_argument("--per-cell", default="200,40,60,40",
                        help="requests for: one question up to 1,024 tokens; one question, longer; several "
                             "questions up to 1,024 tokens; several questions, longer")
    parser.add_argument("--control-cells", default="single-long,multi-short,multi-long",
                        help="cells the wrong-on-purpose variants run on")
    parser.add_argument("--control-limit", type=int, default=15,
                        help="requests per cell for every variant other than port")
    parser.add_argument("--max-tokens", type=int, default=9000, help="skip longer requests (dense attention memory)")
    parser.add_argument("--seed", default="0")
    args = parser.parse_args()
    variants = args.variants.split(",")
    if any(v not in VARIANTS for v in variants):
        parser.error(f"variants must be from {VARIANTS}")
    per_cell = dict(zip(CELLS, (int(n) for n in args.per_cell.split(",")), strict=True))

    sample = select(args.rows, args.reference, per_cell, args.max_tokens, args.seed)
    print(f"sample: {len(sample)} requests", flush=True)

    from importlib.metadata import version

    started = time.perf_counter()
    if args.backend == "mlx":
        if variants != ["port"]:
            parser.error("the mlx backend runs the port variant only")
        import mlx.core as mx

        from karpo_apple_silicon import mlx as karpo_mlx

        decider = karpo_mlx.load(args.checkpoint)
        synchronize = mx.synchronize
        names = ("karpo-decision", "torch", "transformers", "mlx", "mlx-lm")
    else:
        from karpo_decision.decider import Decider

        karpo_apple_silicon._original_window_mask = karpo_apple_silicon.window_mask
        karpo_apple_silicon.install()
        synchronize = torch.mps.synchronize if torch.device(args.device).type == "mps" else (lambda: None)
        decider = Decider.from_pretrained(args.checkpoint, device=args.device)
        names = ("karpo-decision", "torch", "transformers")
    print(f"loaded in {time.perf_counter() - started:.0f} s", flush=True)
    decider.decide(sample[0][3])  # warm-up

    report = {"checkpoint": Path(args.checkpoint).name, "backend": args.backend,
              "device": "mlx" if args.backend == "mlx" else args.device, "seed": args.seed,
              "versions": {name: version(name) for name in names},
              "port_version": karpo_apple_silicon.__version__, "sample_requests": len(sample), "variants": {}}
    progress_path = Path(str(args.out) + ".progress.jsonl")
    controls = set(args.control_cells.split(","))
    for variant in variants:
        if variant == "port":
            subset = sample
        else:
            cells = CELLS if variant == "float16" else [c for c in CELLS if c in controls]
            subset = [s for cell in cells for s in [x for x in sample if x[0] == cell][:args.control_limit]]
        print(f"variant {variant}: {len(subset)} requests", flush=True)
        if args.backend == "torch":
            set_variant(variant, decider)
        report["variants"][variant] = run(decider, subset, synchronize, progress_path, variant)
        overall = report["variants"][variant]["all"]
        print(json.dumps({k: overall[k] for k in ("requests", "questions", "token_mismatches", "top_answers_changed",
                                                  "max_abs_probability_difference",
                                                  "mean_abs_probability_difference")}), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
