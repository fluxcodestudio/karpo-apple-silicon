# karpo-apple-silicon

[![Tests](https://github.com/fluxcodestudio/karpo-apple-silicon/actions/workflows/tests.yml/badge.svg)](https://github.com/fluxcodestudio/karpo-apple-silicon/actions/workflows/tests.yml)

Run the [Karpo Decision](https://huggingface.co/MachinePulseAI/karpo-decision-31B) 31B and 12B models on a Mac
with Apple Silicon.

**How do I run Karpo Decision on a Mac?** Install MachinePulse's `karpo-decision`, install this package, and
call `karpo_apple_silicon.install()` once before the first `decide()`. That replaces the CUDA-only attention
call that makes the model fail on Apple Silicon with
`'aten::_flash_attention_forward' is not currently implemented for the MPS device`, and the 31B and 12B then run
on the Mac's GPU. For Macs with less memory there are ready-made 8-bit and 4-bit MLX copies of the 31B.

This is an **unofficial community port**. It is not affiliated with, endorsed by or supported by MachinePulse
or Google. The models and the inference code they need are MachinePulse's
([karpo-decision](https://github.com/machinepulse-ai/karpo-decision), Apache-2.0); this repository adds one
small patch on top and ships no weights.

> Powered by karpo-decision (MachinePulse): <https://huggingface.co/MachinePulseAI/karpo-decision-31B>

## At a glance

| | |
|---|---|
| What it is | A small Python package that makes MachinePulse's Karpo Decision models run on macOS |
| Models | `karpo-decision-31B` and `karpo-decision-12B` (the Gemma 4 family checkpoints) |
| Hardware | Apple Silicon Macs (M1, M2, M3, M4 and later), using the Metal GPU through PyTorch `mps` |
| Also included | An experimental MLX path with 8-bit and 4-bit quantized weights, for Macs with less memory |
| Not covered | The Qwen-family Karpo models (0.8B to 9B and 27B-act), images, and NVIDIA-only server features |
| Weights | None in this repository: the originals and ready-made MLX 8-bit and 4-bit copies are on Hugging Face |
| Licence | Apache-2.0, free to use and change |

What each route needed on a 96 GB MacBook Pro (M2 Max), in the units macOS reports. Macs with less memory were
not tried, so leave room for macOS and your other apps when you choose:

| Route | Memory it used | Longest single sequence it was run with (karpo-decision's default limit is 32,768 tokens) |
|---|---|---|
| 31B as shipped, through PyTorch | 60 GB of weights on the GPU, plus working memory | 31,489 tokens; practical to about 10,000 tokens on 96 GB, slow beyond |
| 31B, 8-bit MLX copy | 32 GB of weights; peaks of 32 to 42 GB | 31,489 tokens, in under seven minutes |
| 31B, 4-bit MLX copy | 18 GB of weights; peaks of 19 to 28 GB | 31,489 tokens, in about seven minutes |
| 12B as shipped, through PyTorch | 23 GB of weights on the GPU, plus working memory | 31,489 tokens, in about three minutes |

**Does Karpo Decision run on a Mac?** Not as published: it needs an NVIDIA GPU. With this package it does.
If you loaded a Karpo Decision checkpoint with `device="mps"` (or on CPU) and the first `decide()` call failed
inside `torch.nn.attention.varlen.varlen_attn`, flash attention or `sliding_attention`, this is the fix.

## Contents

- [The problem](#the-problem) and [the fix](#the-fix)
- [Errors this fixes](#errors-this-fixes), including
  [wrong answers on long requests](#long-requests-wrong-answers-without-an-error)
- [Install](#install) and [use](#use)
- [Speed](#speed): [the 31B on a 96 GB Mac](#the-31b-on-a-96-gb-mac-version-020) and
  [long requests, up to 31,489 tokens](#long-requests-up-to-31489-tokens-version-020)
- [What is verified, and how](#what-is-verified-and-how)
- [Experimental: MLX, for Macs with less memory](#experimental-mlx-for-macs-with-less-memory)
- [Known limits](#known-limits)
- [Tests](#tests)
- [Questions people ask](#questions-people-ask)
- [Reporting a mismatch](#reporting-a-mismatch)
- [Licence and credit](#licence-and-credit)

## The problem

Karpo Decision's code is written for NVIDIA GPUs. On a Mac the checkpoint loads without complaint, and then
the first request fails: fifty of the 31B's sixty layers and forty of the 12B's forty-eight layers
("sliding-window" layers) compute their attention
with a flash-attention kernel that only exists for CUDA. The remaining "global" layers do run on Apple's GPU as
written, but they compute their attention over a whole request in one piece, and for long requests that went
wrong without any error (see "Long requests: wrong answers without an error" below).

## The fix

`karpo_apple_silicon.install()` replaces the sliding-window attention call, at run time, with ordinary masked
attention from PyTorch that computes the same result on any device, to rounding. Queries are answered 256 at
a time, each chunk reading only the keys its window can reach. Without CUDA, it also replaces the global
layers' attention call with one that computes the same result 256 query rows at a time, which keeps long
requests correct and lets the model run on a CPU. Nothing in the installed karpo-decision package is edited,
and on a machine with an NVIDIA GPU `install()` does nothing unless called with `force=True`.

The rule the replacement follows, for one packed sequence with `Lq` queries, `Lk` keys and `d = Lk - Lq`:
query `i` reads key `j` when `i + d - 1023 <= j <= i + d`. In words: each token reads itself and the 1,023
tokens before it, and when a question's branch reads the shared part of the request, the two are lined up at
their ends.

## Errors this fixes

On a Mac, karpo-decision loads the checkpoint and then the first `decide()` call stops with one of these
(the text comes from PyTorch):

```text
NotImplementedError: The operator 'aten::_flash_attention_forward' is not currently implemented for the MPS device.
```

```text
NotImplementedError: Could not run 'aten::_flash_attention_forward' with arguments from the 'CPU' backend.
```

The global layers can also stop on CPU with this error, which version 0.1.0 of this port did not fix:

```text
RuntimeError: No viable backend for scaled_dot_product_attention was found.
```

Setting `PYTORCH_ENABLE_MPS_FALLBACK=1`, as the first message suggests, does not help: the operator does not
exist for the CPU either. The call comes from `karpo_decision.families.gemma.sliding_attention.attend`, which
uses `torch.nn.attention.varlen.varlen_attn`. `karpo_apple_silicon.install()` replaces that call and, without
CUDA, the global layers' attention call as well, which is where the last error comes from (it asks PyTorch for
a kernel that exists only for CUDA). On CPU the real 12B was run on an Apple Silicon Mac: with
version 0.1.0 it stopped with the last error above; with version 0.2.0 it gave the same top answers as on the
Apple GPU, with probabilities within 0.00003, for a 249-token and a 1,049-token request.

### Long requests: wrong answers without an error

Version 0.1.0 of this port had a second failure, one that raises nothing. Without CUDA, karpo-decision's global
layers make one attention call over a whole sequence, and its table of scores has heads x tokens x tokens
entries. On a 96 GB MacBook Pro (M2 Max) the real 12B needed 43 GB on top of its weights for a 16,170-token
request that way. A 23,627-token request came back as noise: of 9 answers that can be checked against facts in
the request 2 were right, every probability sat near an even split, and macOS reported critical memory pressure
while it ran. Nothing was raised and every number was finite.

Version 0.2.0 makes the same call 256 query rows at a time. On the same Mac the 23,627-token request is then
answered correctly (9 of 9) in 122 seconds instead of 404, the 16,170-token request gives the same
probabilities as before with half the GPU memory, and requests of up to 31,489 tokens, just under
karpo-decision's default limit of 32,768 per sequence, are answered correctly (see "Long requests, up to
31,489 tokens" under Speed).

Why the single call returned noise instead of failing was not established. The same call on random tensors of
that size was correct when nothing else was loaded, and at 16,170 tokens it was correct even with 30 GB of
other GPU memory held, so neither the size alone nor a mild shortage explains it. The one time it happened,
memory pressure was critical. With version 0.2.0 the 31B passed through a moment of critical pressure on a
31,489-token request and still answered it correctly, but until the cause is known, answers produced under
critical memory pressure deserve a second look.

## Install

You need a Mac with Apple Silicon and macOS 14 or later, Python 3.12 or later, and enough memory for the weights:
the 31B takes 62.6 GB (so a 96 GB or larger Mac in practice), the 12B 24.0 GB.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install "karpo-decision @ git+https://github.com/machinepulse-ai/karpo-decision@c52d3507a254d3ad8423726c6bc6568c99113e7d"
pip install "karpo-apple-silicon @ git+https://github.com/fluxcodestudio/karpo-apple-silicon@v0.2.0"
```

The third line installs the newest karpo-decision commit this port was checked with (version 0.1.0). The
measurements in this README were made at that commit and at the one before it, 6600011eb5cf; the pinned commit
changes only licence and Docker text, so the two have the same source files, and upstream's tests pass against
this port at both. A newer one may work, but has not been checked. The fourth line installs this port's
release 0.2.0; leave out `@v0.2.0` for the newest commit.

To run the example, benchmark, tests or converter from this repository, clone it and install it in editable mode
instead of the second `pip install` line above:

```bash
git clone https://github.com/fluxcodestudio/karpo-apple-silicon
cd karpo-apple-silicon
pip install -e .
```

Download a checkpoint (62.6 GB for the 31B, 24.0 GB for the 12B):

```bash
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("MachinePulseAI/karpo-decision-12B"))'
```

## Use

```python
import karpo_apple_silicon
karpo_apple_silicon.install()          # once, before the first decide()

from karpo_decision.decider import Decider

decider = Decider.from_pretrained("path/to/karpo-decision-12B", device="mps")
response = decider.decide({
    "state": "Order 1182: the customer asks to return a blender bought 45 days ago. Policy: returns within 30 days.",
    "questions": {
        "eligible": {"type": "noul", "instructions": "Is the return eligible under the policy?"},
        "action": {"type": "choice", "instructions": "What should the agent do?",
                   "criteria": {"refund": "Issue a full refund", "decline": "Politely decline the return",
                                "escalate": "Escalate to a supervisor"}},
    },
})
print(response["answers"])
```

The request and response formats are karpo-decision's own; see its README. A complete script is in
[examples/decide.py](examples/decide.py).

### Two things that make it faster

1. **float16.** The checkpoint is stored in bfloat16, which Apple's GPU handles slowly. Converting once after
   loading is faster and moved answers very little in our checks (numbers below):

   ```python
   import torch
   decider.ckpt.backbone.to(torch.float16)
   ```

2. **Batches.** Pass a list of requests to `decide()` instead of calling it once per request. karpo-decision
   packs them into shared forward passes.

On CPU (`device="cpu"`), convert to float32 instead, with `decider.ckpt.backbone.to(torch.float32)`: bfloat16
is far slower there, and float32 takes twice the memory (48 GB for the 12B). On an M2 Max a 1,049-token request
to the 12B took 14 seconds in float32 on eight threads and 250 seconds in bfloat16 on four; the answers differed
by 0.00002 at most.

## Speed

Measured with version 0.1.0 on one machine: a Mac Studio, M2 Ultra, 192 GB. **The machine was busy with other jobs during every
measurement** (load average above 20), so treat these as "at least this fast", not as what the hardware can
do. One yes/no question per request; the 31B requests were about 206 tokens long, the 12B requests (from
[scripts/bench.py](scripts/bench.py)) 294 tokens.

| Model and setting | Per request |
|---|---:|
| 31B, bfloat16 as shipped, one at a time | 2.1 s |
| 31B, bfloat16, batches of 16 | 1.4 s |
| 31B, float16, one at a time | 1.4 s |
| 31B, float16, batches of 16 | 0.86 s |
| 12B, bfloat16 as shipped, one at a time | 0.95 s |
| 12B, bfloat16, batches of 16 | 0.74 s |
| 12B, float16, one at a time | 0.58 s |
| 12B, float16, batches of 16 | 0.44 s |
| 31B, MLX 8-bit (experimental, below), one at a time | 1.37 s |
| 31B, MLX 4-bit (experimental, below), one at a time | 1.35 s |

For comparison, MachinePulse reports a median of 94 ms per request for the 31B on one datacentre NVIDIA GPU.
A Mac is roughly ten to twenty times slower, and with version 0.1.0 time grew quickly with length: an 8,000-token request with
forty questions took about 55 seconds (34 in float16). Run [scripts/bench.py](scripts/bench.py) to measure
your own.

### Long requests (version 0.2.0)

One sliding-window layer, random tensors of the model's shape, no weights, M2 Max, machine busy (load average
about 10), median of 5:

| One sliding-window layer | 0.1.0 (full table) | 0.2.0 (banded) |
|---|---:|---:|
| 31B shape, float16, 2,000 tokens | 32 ms | 18 ms |
| 31B shape, float16, 4,000 tokens | 121 ms | 40 ms |
| 31B shape, float16, 8,000 tokens | 493 ms | 87 ms |
| 31B shape, bfloat16, 8,000 tokens | 1,060 ms | 167 ms |
| 31B shape, float16, 7,000 shared tokens and 40 questions | 443 ms | 126 ms |
| 12B shape, float16, 8,000 tokens | 250 ms | 45 ms |
| 31B shape, float16, 206 tokens | 3 ms | 3 ms |

End to end the gain is much smaller, because most of a request's time is spent elsewhere. The real 12B, one
request at a time, on a MacBook Pro (M2 Max, 38-core GPU, 96 GB): seconds per request, the fastest of 6 runs of
each version, taken in alternation in two sessions of 3.

| Request | Tokens | bfloat16, 0.1.0 | bfloat16, 0.2.0 | float16, 0.1.0 | float16, 0.2.0 |
|---|---:|---:|---:|---:|---:|
| The example under Use, 2 questions | 249 | 1.18 | 1.13 | 0.69 | 0.69 |
| 1 question | 1,049 | 4.36 | 4.31 | 2.97 | 2.84 |
| 1 question | 1,949 | 8.38 | 7.90 | 5.54 | 5.26 |
| 1 question | 3,767 | 18.0 | 15.2 | 11.8 | 10.4 |
| 1 question | 7,464 | 43.4 | 34.2 | 29.4 | 22.9 |
| 40 questions | 10,196 | 58.0 | 48.3 | 37.6 | 32.0 |

Version 0.2.0 is about 1.3 times as fast at 7,464 tokens, 1.14 to 1.18 times at 3,767, about 1.05 times at
1,949, and about the same at 1,049 and below. The 40-question request gains about 1.2 times in bfloat16; in
float16 the two sessions gave 1.07 and 1.17. The machine was not idle in either session: in the first a file
copy was running and some repetitions differed by up to 29%, in the second they differed by 13% at most. For
the single-question requests of 1,049 tokens and more the two sessions' ratios agree within 0.03, except the
3,767-token bfloat16 row, which was noisy in the first.

Both columns of that table, and the breakdown below, were measured before the global layers were answered in
chunks, so they compare the two sliding-window implementations and nothing else. With the released code,
single runs of the last five rows took 4.33, 7.88, 15.3, 31.6 and 46.0 seconds in bfloat16 and 2.73, 5.01,
10.1, 21.9 and 29.5 in float16: the same as the 0.2.0 columns or a little less.

Where the time goes inside the 7,464-token request, measured by draining the GPU queue around each part:

| Part of the request | bfloat16, 0.1.0 | bfloat16, 0.2.0 | float16, 0.1.0 | float16, 0.2.0 |
|---|---:|---:|---:|---:|
| Sliding-window attention (the function this port supplies) | 12.8 s (29%) | 3.0 s (9%) | 8.4 s (30%) | 2.1 s (9%) |
| Feed-forward layers | 20.2 s (46%) | 20.3 s (60%) | 11.9 s (42%) | 12.0 s (54%) |
| Attention projections, norms and rotation | 6.9 s (16%) | 6.6 s (19%) | 4.4 s (15%) | 4.4 s (20%) |
| Global attention | 3.3 s (8%) | 3.3 s (10%) | 2.6 s (9%) | 2.7 s (12%) |
| Everything else | 0.9 s (2%) | 0.9 s (3%) | 1.0 s (3%) | 0.9 s (4%) |

The sliding-window attention itself became about four times faster, but with 0.2.0 it is under a tenth of a
long request; the rest is karpo-decision's and PyTorch's own code, which this port does not change. At 1,949
tokens the sliding-window attention was 11 to 12% of the request with 0.1.0 and is 7 to 8% with 0.2.0.
Converting to float16, above, does more for speed than this change. The per-request table at the top of this
section was measured with 0.1.0.

### The 31B on a 96 GB Mac (version 0.2.0)

The 31B was run with version 0.2.0 on a MacBook Pro (M2 Max, 38-core GPU, 96 GB) with most other apps closed:
once as shipped (bfloat16) and once converted to float16. It loaded in about 110 seconds, with the files
already in the disk cache, and held about 60 GB on the GPU. Swap in use stayed between 9.6 and 11.5 GB through
each session. Available memory after a request fell with its length, from 20 GB after the shortest to 8 GB
(bfloat16) and 9 GB (float16) after the 10,196-token one. Memory here and in the MLX section is in the
units macOS reports it in (1 GB is 2^30 bytes, so the 62.6 GB checkpoint is 58.3 GB of memory). Seconds per
request, one run of each: a rough guide, not a measurement (load average about 6). The first request of each
session, the 249-token example, is left out as a warm-up.

| Request | Tokens | bfloat16 | float16 |
|---|---:|---:|---:|
| 1 question | 414 | 4.8 | 3.2 |
| 1 question | 1,049 | 11.1 | 6.6 |
| 1 question | 1,949 | 20.7 | 12.5 |
| 12 questions | 2,910 | 31.2 | 18.8 |
| 1 question | 3,767 | 39.8 | 24.2 |
| 12 questions | 4,574 | 49.7 | 29.9 |
| 1 question | 7,464 | 83.3 | 51.3 |
| 40 questions | 10,196 | 128 | 107 |

- **It works, and requests up to about 10,000 tokens are practical.** All 71 questions of the nine requests
  were answered in both precisions, and all 51 answers that can be checked against facts in the requests were
  right in both.
- **float16 gives the same top answers, faster.** All 71 top answers matched bfloat16; the largest difference in
  a probability was 0.028 and none exceeded 0.05. The single runs were 1.5 to 1.7 times as fast up to 7,464
  tokens and 1.2 times at 10,196.
- **Answering the global layers in chunks is what made long requests practical here.** Before that change the
  same Mac took 240 seconds (bfloat16) and 159 (float16) for the 7,464-token request and 370 and 287 for the
  10,196-token one, with swap growing to 23 or 24 GB: the single table of scores used the memory. The answers
  did not move: on all 71 questions, in both precisions, no probability differed by as much as 0.00005.
- **Same top answers as 0.1.0's sliding-window attention.** Before the global layers were changed, version
  0.1.0's function was swapped in after 0.2.0's on each of the first eight requests in bfloat16 (31 questions):
  no top answer differed. Only three of those answers were not near-certain (a top probability below 0.98);
  they were in the 4,574-token request, where the largest difference in a probability was 0.037. In the other
  seven requests it was 0.00003 or less.
- **About the same time as the 8-bit MLX copy, in twice the memory.** That copy (below) took 79 seconds for the
  7,464-token request and 132 for the 10,196-token one on this Mac, one run of each.

### Long requests, up to 31,489 tokens (version 0.2.0)

The same kind of invented support log, longer: 12 questions over 13,140 to 31,489 tokens, 40 questions over
15,974, and 40 questions over a log of about 30,000 tokens, which is more than one sequence can hold, so
karpo-decision splits it into two of about 31,600 tokens each. Same MacBook Pro, version 0.2.0, seconds for one
run of each (a rough guide). Each request holds facts that 7 to 28 of its answers can be checked against.

| Tokens | 12B, bfloat16 | 12B, float16 | 31B, bfloat16 | 31B, 8-bit MLX | 31B, 4-bit MLX | Checked answers right |
|---:|---:|---:|---:|---:|---:|---|
| 13,140 | 61 | 38 | 258 | 123 | not run | 7 of 7 in each |
| 16,170 | 76 | 48 | 304 | 165 | 178 | 9 of 9 in each |
| 15,974, 40 questions | 75 | 48 | not run | 182 | not run | 28 of 28 in each |
| 23,627 | 122 | 80 | 1,263 | 278 | not run | 9 of 9 in each |
| 31,489 | 177 | 124 | 2,175 | 399 | 432 | 9 of 9 in each |
| 63,269 in two sequences, 40 questions | 364 | not run | not run | 788 | not run | 27 of 27 in each |

- **Every answer that could be checked was right**: 89 of 89 for the 12B in bfloat16 and for the 8-bit MLX
  copy, 62 of 62 for the 12B in float16, 34 of 34 for the 31B, 18 of 18 for the 4-bit MLX copy. On the same
  check the nine shorter requests above score 51 of 51 on every route.
- **The 12B has room to spare.** Memory pressure stayed normal throughout, and float16 was about 1.5 times as
  fast as bfloat16.
- **The 31B as shipped fits, with little room to work in.** Beyond about 10,000 tokens swap grew to 31 GB and
  time rose steeply: 21 minutes at 23,627 tokens and 36 at 31,489, where macOS reported critical memory
  pressure for a moment. The answers stayed right. On a 96 GB Mac the MLX copies are the practical route for
  requests this long: memory peaked at 36 to 42 GB with the 8-bit copy, and at 23.6 and 28.3 GB with the 4-bit
  copy, which gave the same top answers as the 8-bit one on both of its requests (largest difference 0.035).
- **Version 0.1.0 cannot do this on the PyTorch route**: see "Long requests: wrong answers without an error".
  On the 8-bit MLX copy, where 0.1.0 is correct, it gave the same top answers as 0.2.0 at 13,140 and 16,170
  tokens (largest differences 0.014 and 0.062) with peaks of 45.2 and 51.4 GB against 36.1 and 37.2 GB.
- **A request whose text alone exceeds the limit is refused**, with karpo-decision's `LengthError`: every
  sequence repeats the text, so there is nothing to split (checked on the MLX route).

## What is verified, and how

Four kinds of evidence, in rising order of how much they prove.

**1. Reading the source.** For the sliding-window layers karpo-decision 0.1.0 calls the CUDA kernel with
`window_size=(1023, 0)`, the model's own scale, and grouped key/value heads; the replacement implements that
rule and nothing else. For the global layers karpo-decision already has a branch for devices without CUDA,
and the replacement makes that branch's call 256 query rows at a time.

**2. Unit tests** (below), which need no weights. They prove the replacement computes the rule as written.
They cannot prove the rule was read correctly from the CUDA kernel, which does not run on a Mac.

**3. Upstream's own tests.** With this port's two functions in place of upstream's, the three Gemma test files
at the pinned karpo-decision commit pass (35 tests, with 138 calls into the sliding-window replacement and 68
into the global one). They compare the packed forward with an unmodified Transformers forward under explicit
masks, with shared blocks one short of, at and one past the window. Wrong-on-purpose variants fail: the
sliding mask aligned at the start, 15 tests; its window one key short, one key long, or absent, 12 tests
each; the global layers with each sequence's last key hidden, 26 tests.
`scripts/check_upstream_suite.py` repeats this check. Upstream states the sliding-window contract
in `tests/test_gemma_blocks.py` (`varlen_reference`) and checks its CUDA kernels against it in
`tests/gpu/test_packed_gemma.py`, which cannot be run on a Mac.

**4. Reproducing MachinePulse's published answers.** MachinePulse publishes the probability the 31B gave for
every request of the [Decision Index](https://github.com/apolinario/decision-index) public suite, run on an
NVIDIA GPU. The same requests were rebuilt here from the public datasets and run on the Mac with this port.

Sample: 340 requests holding 1,606 questions from 11 tracks of the suite, chosen by a fixed rule before
anything was run: 200 short single-question requests, 40 single-question requests longer than the 1,024-token
window, 60 requests with several questions, and 40 with several questions and longer than the window (up to
8,072 tokens and 43 questions). The script compares the hash recorded in the rebuilt suite row with the
published one and drops rows that differ; it compares token counts after running.

The comparison was run twice on that sample: with version 0.1.0's full-table attention on a Mac Studio (M2
Ultra, 192 GB, Darwin kernel 25.3), and with version 0.2.0's banded attention on a MacBook Pro (M2 Max, 96 GB,
Darwin kernel 27.0). For the second run the suite rows were rebuilt from scratch and came out byte-identical
to the first run's, the script chose the same 340 requests, and every request's token count matched the
published one.

The 0.2.0 rows were measured during development: the PyTorch rows with the global-layer change in place
(commits e071fb1 and 50e21ac), the MLX rows just before it (commit 6c32a38; that change does not touch the MLX
route). The commits after those changed error handling, types, tests and packaging, not arithmetic. As a
check, the last commit that changes the package (57a245f) was run again on nine requests of 249 to 10,196
tokens (71 questions): the 12B in bfloat16 and in float16, the 31B in bfloat16 and both MLX copies each gave
the answers of the measured code, with no probability differing by as much as 0.000001. The 12B on CPU was
run again on two of those requests, with no difference to five decimal places. The 31B in float16 was not run
again.

| 31B on the Mac, against the published NVIDIA results | Questions | Top answer changed | Mean difference | Largest difference |
|---|---:|---:|---:|---:|
| Version 0.2.0, bfloat16 as shipped (340 requests) | 1,606 | 1 | 0.0012 | 0.090 |
| Version 0.2.0, float16 (60 of the requests) | 524 | 0 | 0.0009 | 0.024 |
| Version 0.2.0, bfloat16, on the same 60 requests | 524 | 0 | 0.0010 | 0.042 |
| Version 0.1.0, bfloat16 as shipped (340 requests) | 1,606 | 1 | 0.0011 | 0.103 |
| Version 0.1.0, float16 (60 of the requests) | 524 | 0 | 0.0009 | 0.027 |
| Version 0.1.0, bfloat16, on the same 60 requests | 524 | 0 | 0.0009 | 0.035 |

"Difference" is the absolute difference in a probability, on a scale of 0 to 1. Two questions out of 1,606
differed by more than 0.05 in each run.

Version 0.2.0's banded attention is also tested against 0.1.0's implementation (kept as
`karpo_apple_silicon.reference.dense_attend`) and agrees to rounding: 1e-12 in float64 and one unit in the
last place in half precision on random tensors on the Apple GPU.

With version 0.1.0, the same check was run with the mask made wrong on purpose, on 45 of the longer and
multi-question requests (509 questions; on these 45 the port itself scored mean 0.0009, largest 0.035 and no
changed answers with version 0.1.0, and mean 0.0010, largest 0.042 and no changed answers with version 0.2.0):

| Mask made wrong on purpose (version 0.1.0 run) | Top answer changed | Mean difference | Largest difference |
|---|---:|---:|---:|
| Branches lined up with the shared part at the start, not the end | 118 | 0.30 | 0.98 |
| No window at all (every earlier token readable) | 0 | 0.0028 | 0.165 |
| Window one token too short | 0 | 0.0009 | 0.032 |

What this does and does not show, plainly:

- The end alignment is confirmed beyond doubt: the wrong alignment changes 118 answers.
- That a window exists is supported: removing it triples the mean difference and gives eight questions over
  0.05 where the port has none on the same requests.
- **The exact window size is not confirmed by the published-results comparison.** A window one token too short is
  indistinguishable from the port there. Upstream's tests above pin it at 1,024 tokens (the token itself and
  1,023 before it).
- Most of the differences that remain look like rounding, not a wrong rule. The one changed answer is the same
  question in both runs: its published probability is 0.49 and it came out at 0.51, with either version and in
  either precision.
- **In bfloat16, an answer far from 0 and 1 can move by a few hundredths with the Mac and with the version.**
  The largest difference of the first run was one question of 39 in a 7,377-token request: 0.312 published,
  0.416 with version 0.1.0 on the Mac Studio. On the MacBook Pro the same 0.1.0 function gave 0.376 and version
  0.2.0 gave 0.338, on the same loaded weights. In float16 the three figures were 0.295, 0.291 and 0.290.
- **Requests shorter than the window give the same answers with both versions.** Seven requests of the sample
  were answered on the MacBook Pro with each version's function in turn on the same loaded weights, in both
  precisions. On the three shorter than the window (162 to 839 tokens) no probability differed by as much as
  0.00005. On the four longer ones (1,345 to 8,064 tokens) the versions differed by up to 0.037 in bfloat16,
  and by up to 0.007 in float16 on the three where both ran (the fourth lacked free memory for 0.1.0's table).
  Those four were chosen for holding the largest differences from the published figures, three in the 0.2.0
  run and one in the 0.1.0 run, and in bfloat16 each version was the further one on the requests chosen as its
  own worst, as that choice alone would predict. The table above is the fair comparison of the versions.
- **One question sits above the published figure on both Macs.** One question in an 839-token request (0.379
  published) came out between 0.43 and 0.44 on the Mac Studio with version 0.1.0, in bfloat16, in float16 and
  through the MLX route below. On the MacBook Pro it came out at 0.47 in bfloat16 and 0.42 in float16, with
  either version. Each figure is 0.04 to 0.09 above the published one, and they differ among themselves by up
  to 0.05, so this question is sensitive to rounding as well. The
  request is shorter than the window, so the window rule cannot be the cause; the cause is not known. One possible
  cause is the published run's torch 2.14.1+cu130 and transformers 5.21.0 on an RTX PRO 6000 Blackwell (its
  `environment.json`), against torch 2.15.0 and transformers 5.23.0 here; this has not been tested. The top
  answer did not change.
- Batching was checked separately, with version 0.1.0: 72 of the requests (182 questions) sent in batches of 8
  and of 24 gave no changed answer, and differed from the same requests sent one at a time on the same Mac by
  0.0002 on average (largest 0.004).
- Not compared: the 12B (it runs through the same code and answers the example sensibly, but no per-request
  results are published for it), any request longer than 8,072 tokens, and images.

To repeat the comparison on your own Mac, see the header of
[scripts/compare_reference.py](scripts/compare_reference.py). It needs the checkpoint, MachinePulse's published
results file and a rebuild of the suite; none of the three ships with this repository. On the Apple GPU the
script empties PyTorch's cache of freed buffers after every request. An earlier run without that grew to
103 GB of memory and 40 GB of swap on the 96 GB MacBook Pro after about 300 requests; with it, the 340
requests took 74 minutes there and swap stayed near 10 GB.

## Experimental: MLX, for Macs with less memory

The PyTorch route above needs the whole 62.6 GB checkpoint in memory. `karpo_apple_silicon.mlx` is a second,
separate route for the 31B that runs the model's layers with Apple's [MLX](https://github.com/ml-explore/mlx)
on a compressed ("quantized") copy of the weights: a 34.0 GB download at 8 bits and 19.3 GB at 4 bits (31.6
and 18.0 GB once in memory, in the units macOS reports). It is more code than the patch above, it is newer,
and it covers less: **text requests only, the 31B only**. Treat it as experimental.

Ready-made copies are on Hugging Face, so you do not need the full 62.6 GB checkpoint to use this route:

| Copy | Download size | Download |
|---|---|---|
| 8-bit | 34.0 GB | [jonnyArt/karpo-decision-31B-mlx-8bit](https://huggingface.co/jonnyArt/karpo-decision-31B-mlx-8bit) |
| 4-bit | 19.3 GB | [jonnyArt/karpo-decision-31B-mlx-4bit](https://huggingface.co/jonnyArt/karpo-decision-31B-mlx-4bit) |

```bash
pip install "karpo-decision @ git+https://github.com/machinepulse-ai/karpo-decision@c52d3507a254d3ad8423726c6bc6568c99113e7d"
pip install "karpo-apple-silicon[mlx] @ git+https://github.com/fluxcodestudio/karpo-apple-silicon@v0.2.0"
hf download jonnyArt/karpo-decision-31B-mlx-8bit --local-dir karpo-decision-31B-mlx-8bit
```

Or convert your own copy from the full checkpoint. Run the script from a clone of this repository. It imports
`karpo_apple_silicon.mlx`, so install karpo-decision as above and the package with its `mlx` extra, which
brings `safetensors` for reading the source. The package installs `torch`, which is also needed to read the
checkpoint tensors.

```bash
pip install -e ".[mlx]"
```

The converter accepts `model-*.safetensors` shards or one `model.safetensors`, and only 8 or 4 bits. It checks
the source tensor names before writing anything. It writes to `<target>.partial`, checks the copied tensor names
and shapes against the loader, then writes `mlx_config.json` last and renames the folder to `<target>`. If it
stops part-way, remove `<target>.partial` before retrying; the next run will refuse to overwrite it. The copied
`NOTICE` gains a modification notice. The vision tower is not copied.

```bash
python scripts/convert_mlx.py path/to/karpo-decision-31B path/to/karpo-decision-31B-mlx-8bit --bits 8
```

```python
from karpo_apple_silicon import mlx as karpo_mlx

decider = karpo_mlx.load("path/to/karpo-decision-31B-mlx-8bit")
response = decider.decide(request)                        # one request returns one response
responses = decider.decide([request, another_request])    # a list returns a list
```

`decide_many`, from version 0.1.0, still works and takes a list. `load` has two optional limits. `token_budget`
(4,096 by default) is the number of tokens per forward pass when several sequences are answered together; a
longer sequence runs alone. `max_length` (karpo-decision's default, 32,768) is the longest single sequence: a
longer request is split by its questions, and one question that cannot fit raises karpo-decision's
`LengthError`. Attention memory grows with the length, so a smaller `max_length` bounds it. Before loading
text-layer weights, the loader checks `mlx_config.json`, karpo-decision's decision configuration (including
bidirectional attention and a Gemma block layout), each answer code against one tokenizer token, the layer type
count and final global layer, and the readout weight shape. Text-layer weights load all or nothing: missing,
extra or wrong-shaped tensors, and cut-short or unreadable weight files, raise an error saying the folder is
incomplete or was not written by the converter. A folder with no `mlx_config.json`, such as the original
checkpoint, is refused as not being a converted copy, and a tensor found in two weight files is reported as
that.

Attention runs in chunks of 256 query rows, each reading only the key rows its window can reach. The last
layer runs only on marker rows. Images are refused; the vision tower is neither converted nor run. The Qwen
models and checkpoints without bidirectional attention are not supported by this route.

The reworked loader and forward pass were checked against karpo-decision on small random models. File headers
in both published 8-bit and 4-bit folders hold exactly the tensors the loader requires, by name and shape.

The published 4-bit copy was then run with version 0.2.0 and with version 0.1.0 of this route on the same
weights: nine invented requests of 249 to 10,196 tokens, 71 questions, on a MacBook Pro (M2 Max, 96 GB) that
was in use.

- **Answers.** No top answer changed. Four answers differed by more than 0.01 and the largest difference was
  0.061 (mean 0.002), on questions in long requests whose answers were not near-certain. This route computes in
  bfloat16; on the same requests the PyTorch route's full-table and banded attention differed by up to 0.056 in
  bfloat16 and 0.007 in float16 on the 12B.
- **Memory.** MLX's peak fell from 23.3 GB to 20.9 GB for a 7,464-token request and from 26.8 GB to 22.0 GB for
  a 10,196-token request with 40 questions. Below about 3,000 tokens it is unchanged (19 GB).
- **Time.** No measurable change. The fastest of three runs of each version differed by 10% or less for the
  seven requests up to 4,574 tokens. A 7,464-token request took 71 to 122 seconds over six runs of version
  0.2.0 and 98 to 123 seconds over four runs of version 0.1.0; that spread is the machine's other work, since
  the same code in the same process gave both ends of it. Most of a layer's time is its quantized projections,
  not its attention.
- **Against the full checkpoint.** The full bfloat16 31B, run through the PyTorch route with version 0.2.0 on
  the same Mac, gave the same top answer on all 71 questions, 66 of which it answered with a top probability of
  0.98 or more. The largest difference in a probability was 0.159 (mean 0.009); five answers differed by more
  than 0.05, all in the 10,196-token request with 40 questions.

The published 8-bit copy was run the same way on the same Mac, one run of each version (0.2.0 first), with a
load average of 5 to 10:

- **Answers.** No top answer changed between the versions on the 71 questions. Three answers differed by more
  than 0.01 and the largest difference was 0.040 (mean 0.001).
- **Memory.** The peak fell from 40.4 GB to 35.6 GB for the 10,196-token request and from 37.0 GB to 34.5 GB
  for the 7,464-token one. Below about 3,000 tokens it is unchanged (32 to 33 GB; the weights take 31.6 GB).
- **Against the full checkpoint.** The same top answer on all 71 questions. The largest difference in a
  probability was 0.052 (mean 0.002), the only one over 0.05; the 4-bit copy's were 0.159 and 0.009, with five
  over 0.05.
- **Time.** With version 0.2.0 the 1,049-token request took 9.1 seconds, the 7,464-token one 79 and the
  10,196-token one 132. Version 0.1.0, run afterwards, took 11.4, 104 and 155; one run of each in one order is
  not a comparison (for the 4-bit copy, alternated runs showed no difference).

Both copies were also compared with MachinePulse's published results, on the sample described under "What is
verified, and how": with version 0.1.0 on the Mac Studio, and with version 0.2.0 on the MacBook Pro, using
the copies as downloaded from Hugging Face. In the 0.2.0 runs every token count matched the published one.

| 31B with MLX, against the published NVIDIA results | Questions | Top answer changed | Mean difference | Largest difference |
|---|---:|---:|---:|---:|
| 8-bit, version 0.2.0 (the same 340 requests) | 1,606 | 2 | 0.0014 | 0.078 |
| 8-bit, version 0.1.0 (the same 340 requests) | 1,606 | 2 | 0.0016 | 0.079 |
| 4-bit, version 0.2.0 (145 of the requests) | 681 | 3 | 0.0067 | 0.176 |
| 4-bit, version 0.1.0 (the same 145) | 681 | 3 | 0.0067 | 0.183 |

At 8 bits, two of the 1,606 questions differed by more than 0.05 with version 0.2.0 and six with version 0.1.0
(two for the PyTorch route in each run). At 4 bits, 23 of the 681 did with version 0.2.0. On those 145
requests the 4-bit copy's mean difference is about five times that of the 8-bit copy and of the full
checkpoint (0.0067 against 0.0013 for each, with version 0.2.0); prefer 8 bits when it fits. The MLX timings
in the Speed table were measured with version 0.1.0. During the version 0.1.0 comparison, memory in use peaked
at 24 GB for 8,000-token requests with the 4-bit copy (18 GB of weights in memory); it was not recorded for
the 8-bit copy in that run. For the version 0.1.0 results, the converter was checked against the folders used
for these numbers on a sample of tensors, not re-run in full. The converted folder is a derivative of
MachinePulse's weights: if you share one, keep their LICENSE and NOTICE with it.
This repository itself ships no weights; the converted copies linked above are hosted on Hugging Face with
MachinePulse's LICENSE and NOTICE.

## Known limits

- **Gemma family only (31B and 12B).** The Qwen-family Karpo Decision models (0.8B to 9B and 27B-act) use a different
  base model whose code calls the CUDA kernel in another place and needs a second CUDA-only package. This port
  does not touch that path, and those models do not run with it.
- **Only text requests have been run with real weights.** An image request goes through both replaced
  functions and then through karpo-decision's own image code. Upstream's image test on a small random model
  passes with the port in place, on CPU; no image request has been run with a real checkpoint or on the Apple
  GPU. Treat images as untested.
- **The MLX route is text-only.** It refuses image requests and does not convert or run the vision tower. Its
  reworked loader and forward pass have been run with the real 4-bit and 8-bit weights.
- **PyTorch inference with `Decider.decide` only.** karpo-decision's server, its FP8 and CUDA-graph fast paths, its
  Triton attention and anything to do with training are NVIDIA-only or untested here.
- **Long requests need working memory beyond the weights.** No table grows with the square of the length any
  more: sliding-window layers read a window per chunk, and the global layers (10 of the 31B's 60, 8 of the
  12B's 48) read the whole request 256 query rows at a time. Requests of up to 31,489 tokens, just under
  karpo-decision's default limit of 32,768 per sequence, have been run and checked (see Speed), but the 31B as
  shipped leaves a 96 GB Mac little room, and beyond about 10,000 tokens it swaps and slows down sharply. The
  published-results comparison goes up to 8,072 tokens.
- **Answers under critical memory pressure deserve a second look.** The one silent failure seen, with version
  0.1.0's whole-table global attention, happened while macOS reported critical memory pressure, and its cause
  was not established (see "Long requests: wrong answers without an error"). Version 0.2.0 passed through a
  moment of critical pressure on the 31B's 31,489-token request and answered it correctly.
- **A long-lived process keeps freed GPU memory.** PyTorch caches the Apple GPU buffers it frees, and requests
  of different lengths leave different buffers behind. Where the weights nearly fill the Mac's memory, call
  `torch.mps.empty_cache()` between requests (see the end of "What is verified, and how" for what happened
  without it).
- **Slower than an NVIDIA GPU**, by roughly ten to twenty times (see Speed).
- **Tied to the pinned karpo-decision commit.** `install()` warns when pip records a git commit other than the
  two it was checked with, or when no commit is recorded and the version differs from 0.1.0. It refuses an
  `attend` or `Packed._global` whose signature it does not recognise, before it swaps either. Upstream's
  version has not changed between commits so far, which is why the commit is compared.
- **Small differences from the published numbers are expected.** Different hardware adds up numbers in a
  different order. Here probabilities differed by 0.001 on average, and by 0.03 to 0.10 in a handful of
  questions whose probability was far from 0 and 1 (see the numbers above). An answer whose probability sits
  close to the decision boundary can therefore come out on the other side.

## Tests

Install the development dependencies from the cloned repository:

```bash
pip install -e ".[dev]"
```

The attention tests need PyTorch and no weights; the packed-forward tests use karpo-decision and Transformers.
The three MLX test modules need karpo-decision, `mlx` and `mlx-lm`; they skip where MLX cannot start.
To run them from a clone, install `pip install -e ".[dev,mlx]"` as well as karpo-decision above.

```bash
python -m unittest discover -s tests -v
```

They check the mask against a brute-force statement of the rule, both attention functions against a
hand-written softmax per query (grouped key/value heads, packed sequences of unequal length, causal and not),
`global_attend` also against karpo-decision's own whole-table call, half precision against float32, and the
packed forward against the dense references on a tiny random model. They also check that no attention call is
given more query rows than a chunk or more keys than its band, which the results alone cannot show, and that
`install()` swaps both functions, restores the same objects and refuses an upstream it was not written for.
Tests that need the Apple GPU skip where there is none; `KARPO_TEST_NO_MPS=1` leaves them out on a Mac whose
GPU is busy.
`test_mlx_masks.py` checks global, sliding and final-marker masks against per-row masks on upstream block plans,
the sliding mask against the PyTorch patch, the shortest readable key run per chunk and rotary frequencies
against Transformers. `test_mlx_forward.py` compares marker states from a small random unquantized MLX model
with karpo-decision's packed forward, including chunks that cross block boundaries and the 31B's attention
head shapes. `test_mlx_convert.py` converts and loads small random checkpoints, checks single and batched text
requests, length and image errors, incomplete or altered folders, pre-weight configuration checks, a single-file
source, an appended modification notice, refused source tensors and bit widths, and leftover partial folders.
It also compares the answers of `decide()` with a hand-written softmax of the readout under a logit cap and
per-question temperatures. `test_scripts.py` checks the arguments the comparison and suite scripts refuse.
These tests do not run the published quantized weights.

To run the pinned upstream CPU suite with the wrong-on-purpose controls, against a checkout of karpo-decision
at the pinned commit:

```bash
git clone https://github.com/machinepulse-ai/karpo-decision upstream
git -C upstream checkout c52d3507a254d3ad8423726c6bc6568c99113e7d
python scripts/check_upstream_suite.py upstream
```

## Questions people ask

**Can I run Karpo Decision on a MacBook Pro, Mac Studio or Mac mini?** Yes, if it has Apple Silicon and enough
memory: about 96 GB for the 31B as shipped, more than the 24.0 GB of weights for the 12B, and less for the 31B
with the quantized MLX copies described above.

**Does it work on Intel Macs?** PyTorch 2.15.0 has no Intel macOS build, so this package cannot be installed there.

**Does it work on Linux or Windows without an NVIDIA GPU (CPU, AMD, Intel Arc)?** The patch is plain PyTorch
and is not tied to macOS. On CPU the real 12B has been run on an Apple Silicon Mac only. On Linux the unit
tests and upstream's CPU suite run on small random models in this repository's automated tests, and no real
checkpoint has been run there. Windows, and AMD and Intel GPUs, have not been checked at all.

**Is there a GGUF, llama.cpp or Ollama version?** Not here. This port keeps MachinePulse's own inference code,
which Karpo Decision's answers depend on, and changes its attention at run time.

**Are the answers the same as on an NVIDIA GPU?** Very nearly: see "What is verified, and how" for the
measured differences and what was not checked.

**Do I need flash-attn or xformers?** No. Neither is needed with this patch.

**Which Karpo models does it cover?** `MachinePulseAI/karpo-decision-31B` and `MachinePulseAI/karpo-decision-12B`.
The Qwen-family Karpo models (0.8B to 9B and 27B-act) are not covered.

## Reporting a mismatch

If an answer on your Mac differs from the same request on an NVIDIA GPU by more than the differences reported
above, please open an issue here (not on MachinePulse's repository; they did not write this port) with:

- the request (state and questions), or a reduced one that still shows it;
- both responses, with their probabilities;
- which checkpoint, whether you converted to float16, your Mac model and memory;
- the output of `pip list | grep -i -E "karpo|torch|transformers|mlx"`.

## Licence and credit

This port is maintained by FluxCode Studio. The code in this repository is licensed under the
[Apache License, Version 2.0](LICENSE); see [NOTICE](NOTICE).

Karpo Decision, its weights and its inference code are the work of MachinePulse (Copyright 2026 MachinePulse
Pte. Ltd., Apache-2.0). karpo-decision-31B and karpo-decision-12B are modified versions of Google DeepMind's
Gemma 4 (Apache-2.0). This port only makes their code run on different hardware; the model's quality, and any
credit for it, is theirs.

Tested with karpo-decision 0.1.0 (commits 6600011eb5cf and c52d3507a254), torch 2.15.0 and transformers
5.23.0; the MLX path with mlx 0.32.4 and mlx-lm 0.32.2.
