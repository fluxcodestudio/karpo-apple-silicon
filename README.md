# karpo-apple-silicon

Run the [Karpo Decision](https://huggingface.co/MachinePulseAI/karpo-decision-31B) 31B and 12B models on a Mac
with Apple Silicon.

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
| Not covered | The Qwen-family Karpo models (0.8B to 9B), images, and NVIDIA-only server features |
| Weights | None shipped: you download them from Hugging Face |
| Licence | Apache-2.0, free to use and change |

**Does Karpo Decision run on a Mac?** Not as published: it needs an NVIDIA GPU. With this package it does.
If you loaded a Karpo Decision checkpoint with `device="mps"` (or on CPU) and the first `decide()` call failed
inside `torch.nn.attention.varlen.varlen_attn`, flash attention or `sliding_attention`, this is the fix.

## The problem

Karpo Decision's code is written for NVIDIA GPUs. On a Mac the checkpoint loads without complaint, and then
the first request fails: fifty of the model's sixty layers ("sliding-window" layers) compute their attention
with a flash-attention kernel that only exists for CUDA. Everything else in the model already runs on Apple's
GPU.

## The fix

`karpo_apple_silicon.install()` replaces that one function, at run time, with ordinary masked attention from
PyTorch that computes the same result on any device. Nothing in the installed karpo-decision package is
edited, and on a machine with an NVIDIA GPU `install()` does nothing.

The rule the replacement follows, for one packed sequence with `Lq` queries, `Lk` keys and `d = Lk - Lq`:
query `i` reads key `j` when `i + d - 1023 <= j <= i + d`. In words: each token reads itself and the 1,023
tokens before it, and when a question's branch reads the shared part of the request, the two are lined up at
their ends.

## Install

You need a Mac with Apple Silicon, Python 3.12 or later, and enough memory for the weights: the 31B takes
62.5 GB (so a 96 GB or larger Mac in practice), the 12B about 22 GB.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install "karpo-decision @ git+https://github.com/machinepulse-ai/karpo-decision@6600011eb5cf46232d056ec0717dc726146d29de"
pip install "karpo-apple-silicon @ git+https://github.com/fluxcodestudio/karpo-apple-silicon"
```

The first line installs the exact karpo-decision commit this port was checked with (version 0.1.0). A newer
one may work, but has not been checked.

Download a checkpoint (58 GB for the 31B, 22 GB for the 12B):

```bash
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("MachinePulseAI/karpo-decision-12B"))'
```

## Use

```python
import karpo_apple_silicon
karpo_apple_silicon.install()          # once, before the first decide()

from karpo_decision.decider import Decider

decider = Decider.from_pretrained("path/to/karpo-decision-31B", device="mps")
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

## Speed

Measured on one machine: a Mac Studio, M2 Ultra, 192 GB. **The machine was busy with other jobs during every
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
A Mac is roughly ten to twenty times slower, and time grows quickly with length: an 8,000-token request with
forty questions took about 55 seconds (34 in float16). Run [scripts/bench.py](scripts/bench.py) to measure
your own.

## What is verified, and how

Three things, in rising order of how much they prove.

**1. Reading the source.** karpo-decision 0.1.0 calls the CUDA kernel with `window_size=(1023, 0)`, the
model's own scale, and grouped key/value heads. The replacement implements that rule and nothing else.

**2. Unit tests** (below), which need no weights. They prove the replacement computes the rule as written.
They cannot prove the rule was read correctly from the CUDA kernel, which does not run on a Mac.

**3. Reproducing MachinePulse's published answers.** MachinePulse publishes the probability the 31B gave for
every request of the [Decision Index](https://github.com/apolinario/decision-index) public suite, run on an
NVIDIA GPU. The same requests were rebuilt here from the public datasets and run on the Mac with this port.

Sample: 340 requests holding 1,606 questions from 11 tracks of the suite, chosen by a fixed rule before
anything was run: 200 short single-question requests, 40 single-question requests longer than the 1,024-token
window, 60 requests with several questions, and 40 with several questions and longer than the window (up to
8,072 tokens and 43 questions). Every request's text matched the published hash and every token count matched
the published one.

| 31B on the Mac, against the published NVIDIA results | Questions | Top answer changed | Mean difference | Largest difference |
|---|---:|---:|---:|---:|
| The port, bfloat16 as shipped (340 requests) | 1,606 | 1 | 0.0011 | 0.103 |
| The port, float16 (60 of the requests) | 524 | 0 | 0.0009 | 0.027 |
| The port, bfloat16, on the same 60 requests | 524 | 0 | 0.0009 | 0.035 |

"Difference" is the absolute difference in a probability, on a scale of 0 to 1. Two questions out of 1,606
differed by more than 0.05.

The same check was run with the mask made wrong on purpose, on 45 of the longer and multi-question requests
(509 questions; the port itself scores mean 0.0009, largest 0.035, no changed answers on these 45):

| Mask made wrong on purpose | Top answer changed | Mean difference | Largest difference |
|---|---:|---:|---:|
| Branches lined up with the shared part at the start, not the end | 118 | 0.30 | 0.98 |
| No window at all (every earlier token readable) | 0 | 0.0028 | 0.165 |
| Window one token too short | 0 | 0.0009 | 0.032 |

What this does and does not show, plainly:

- The end alignment is confirmed beyond doubt: the wrong alignment changes 118 answers.
- That a window exists is supported: removing it triples the mean difference and gives eight questions over
  0.05 where the port has none on the same requests.
- **The exact window size is not confirmed by this comparison.** A window one token too short is
  indistinguishable from the port here. That the window is 1,024 tokens (the token itself and 1,023 before it)
  rests on reading the source and on the unit tests.
- Most of the differences that remain look like rounding, not a wrong rule. The one changed answer had a published
  probability of 0.49 and came out at 0.51. The largest difference (0.312 published, 0.416 here, one question
  of 39 in a 7,377-token request) came out at 0.295 when the same request was run in float16 on the same Mac:
  the two precisions on one machine disagree with each other by more than either disagrees with the published
  figure.
- **One difference is not rounding on the Mac's side.** One question in an 839-token request (0.379
  published) came out between 0.43 and 0.44 in bfloat16, in float16 and in the separate MLX route below. Three different
  calculations on the Mac agree with each other and differ from the published figure by 0.05 to 0.06. The
  request is shorter than the window, so the window rule cannot be the cause; the cause is not known. The top
  answer did not change.
- Batching was checked separately: 72 of the requests (182 questions) sent in batches of 8 and of 24 gave no
  changed answer, and differed from the same requests sent one at a time on the same Mac by 0.0002 on average
  (largest 0.004).
- Not compared: the 12B (it runs through the same code and answers the example sensibly, but its published
  results were not downloaded), any request longer than 8,072 tokens, and images.

To repeat the comparison on your own Mac, see the header of
[scripts/compare_reference.py](scripts/compare_reference.py). It needs the checkpoint, MachinePulse's published
results file and a rebuild of the suite; none of the three ships with this repository.

## Experimental: MLX, for Macs with less memory

The PyTorch route above needs the whole 62.5 GB checkpoint in memory. `karpo_apple_silicon.mlx` is a second,
separate route for the 31B that runs the model's layers with Apple's [MLX](https://github.com/ml-explore/mlx)
on a compressed ("quantized") copy of the weights: about 32 GB at 8 bits, about 18 GB at 4 bits. It is more
code than the patch above, it is newer, and it covers less: **text requests only, the 31B only**. Treat it as
experimental.

```bash
pip install "karpo-apple-silicon[mlx] @ git+https://github.com/fluxcodestudio/karpo-apple-silicon"
python scripts/convert_mlx.py path/to/karpo-decision-31B path/to/karpo-decision-31B-mlx-8bit --bits 8
```

```python
from karpo_apple_silicon import mlx as karpo_mlx

decider = karpo_mlx.load("path/to/karpo-decision-31B-mlx-8bit")
response = decider.decide(request)             # same request and answer format as above
responses = decider.decide_many([request, another_request])
```

The same comparison against MachinePulse's published results:

| 31B with MLX, against the published NVIDIA results | Questions | Top answer changed | Mean difference | Largest difference |
|---|---:|---:|---:|---:|
| 8-bit (the same 340 requests) | 1,606 | 2 | 0.0016 | 0.079 |
| 4-bit (145 of the requests) | 681 | 3 | 0.0067 | 0.183 |

Six of the 1,606 questions differed by more than 0.05 at 8 bits (two for the PyTorch route). The 4-bit copy
moves answers about six times as much as the full checkpoint does; prefer 8 bits when it fits. Memory in use
peaked at 24 GB for 8,000-token requests with the 4-bit copy (18 GB of weights); the 8-bit copy was not
measured this way. The converter
was checked against the folders used for these numbers on a sample of tensors, not re-run in full. The converted
folder is a derivative of MachinePulse's weights: if you share one, keep their LICENSE and NOTICE with it.
This repository ships no converted weights.

## Known limits

- **Gemma family only (31B and 12B).** The smaller Karpo Decision models (0.8B to 9B) are built on a different
  base model whose code calls the CUDA kernel in another place and needs a second CUDA-only package. This port
  does not touch that path, and those models do not run with it.
- **Text requests only have been run.** Requests with images go through code this patch does not change and
  that looks device-independent, but no image request has been run on a Mac. Treat images as untested.
- **Inference with `Decider.decide` only.** karpo-decision's server, its FP8 and CUDA-graph fast paths, its
  Triton attention and anything to do with training are NVIDIA-only or untested here.
- **Memory grows with the square of the request length.** The replacement builds a full attention table per
  layer, where the NVIDIA kernel does not. The longest request run here was 8,072 tokens; very long requests
  (the model accepts up to 32,768 tokens) may run out of memory or be very slow.
- **Slower than an NVIDIA GPU**, by roughly ten to twenty times (see Speed).
- **Tied to karpo-decision 0.1.0.** `install()` warns on any other version, because the rule it copies could
  change upstream.
- **Small differences from the published numbers are expected.** Different hardware adds up numbers in a
  different order. Here probabilities differed by 0.001 on average, and by 0.03 to 0.10 in a handful of
  questions whose probability was far from 0 and 1 (see the numbers above). An answer whose probability sits
  close to the decision boundary can therefore come out on the other side.

## Tests

The unit tests need PyTorch only (no weights, no GPU); two of them also use karpo-decision if it is installed,
and the MLX mask tests need `mlx` and `mlx-lm`.

```bash
python -m unittest discover -s tests -v
```

They check the mask against a brute-force statement of the rule, the attention against a hand-written softmax
per query (grouped key/value heads, several packed sequences of unequal length, half precision), the patch fed
by karpo-decision's own plan against the rule stated by token position, and that a query with no key to read
is an error rather than a silent wrong answer.

## Reporting a mismatch

If an answer on your Mac differs from the same request on an NVIDIA GPU by more than the differences reported
above, please open an issue here (not on MachinePulse's repository; they did not write this port) with:

- the request (state and questions), or a reduced one that still shows it;
- both responses, with their probabilities;
- which checkpoint, whether you converted to float16, your Mac model and memory;
- the output of `pip list | grep -i -E "karpo|torch|transformers|mlx"`.

## Licence and credit

The code in this repository is licensed under the [Apache License, Version 2.0](LICENSE); see [NOTICE](NOTICE).

Karpo Decision, its weights and its inference code are the work of MachinePulse (Copyright 2026 MachinePulse
Pte. Ltd., Apache-2.0). karpo-decision-31B and karpo-decision-12B are modified versions of Google DeepMind's
Gemma 4 (Apache-2.0). This port only makes their code run on different hardware; the model's quality, and any
credit for it, is theirs.

Tested with karpo-decision 0.1.0 (commit 6600011eb5cf), torch 2.15.0 and transformers 5.23.0; the MLX path with mlx 0.32.4 and
mlx-lm 0.32.2.
