# karpo-apple-silicon

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

**Does Karpo Decision run on a Mac?** Not as published: it needs an NVIDIA GPU. With this package it does.
If you loaded a Karpo Decision checkpoint with `device="mps"` (or on CPU) and the first `decide()` call failed
inside `torch.nn.attention.varlen.varlen_attn`, flash attention or `sliding_attention`, this is the fix.

## The problem

Karpo Decision's code is written for NVIDIA GPUs. On a Mac the checkpoint loads without complaint, and then
the first request fails: fifty of the 31B's sixty layers and forty of the 12B's forty-eight layers
("sliding-window" layers) compute their attention
with a flash-attention kernel that only exists for CUDA. Everything else in the model already runs on Apple's
GPU.

## The fix

`karpo_apple_silicon.install()` replaces the sliding-window attention call, at run time, with ordinary masked
attention from PyTorch that computes the same result on any device, to rounding. Queries are answered 256 at
a time, each chunk reading only the keys its window can reach. Without CUDA, it also lifts the global layers'
kernel restriction, which otherwise stops CPU execution. Nothing in the installed karpo-decision package is
edited, and on a machine with an NVIDIA GPU `install()` does nothing unless called with `force=True`.

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
CUDA, lifts the global layers' kernel restriction. On CPU the real 12B was run on an Apple Silicon Mac: with
version 0.1.0 it stopped with the last error above; with version 0.2.0 it gave the same top answers as on the
Apple GPU, with probabilities within 0.00003, for a 249-token and a 1,049-token request.

## Install

You need a Mac with Apple Silicon and macOS 14 or later, Python 3.12 or later, and enough memory for the weights:
the 31B takes 62.6 GB (so a 96 GB or larger Mac in practice), the 12B 24.0 GB.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install "karpo-decision @ git+https://github.com/machinepulse-ai/karpo-decision@6600011eb5cf46232d056ec0717dc726146d29de"
pip install "karpo-apple-silicon @ git+https://github.com/fluxcodestudio/karpo-apple-silicon"
```

The third line installs the exact karpo-decision commit this port was checked with (version 0.1.0). A newer
one may work, but has not been checked.

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
Converting to float16, above, does more for speed than this change. The 31B has not been timed end to end with
0.2.0, and the per-request table at the top of this section was measured with 0.1.0.

## What is verified, and how

Four kinds of evidence, in rising order of how much they prove.

**1. Reading the source.** karpo-decision 0.1.0 calls the CUDA kernel with `window_size=(1023, 0)`, the
model's own scale, and grouped key/value heads. The replacement implements that rule and nothing else.

**2. Unit tests** (below), which need no weights. They prove the replacement computes the rule as written.
They cannot prove the rule was read correctly from the CUDA kernel, which does not run on a Mac.

**3. Upstream's own tests.** With this port's function in place of the kernel, the three Gemma test files at the
pinned karpo-decision commit pass (35 tests, 138 calls into the replacement). They compare the packed forward
with an unmodified Transformers forward under explicit masks, with shared blocks one short of, at and one past
the window. Wrong-on-purpose masks fail: aligned at the start, 15 tests; window one key short, one key long,
or absent, 12 tests each. `scripts/check_upstream_suite.py` repeats this check. Upstream states the same contract
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

This comparison used 0.1.0's full-table attention. Version 0.2.0's banded attention is tested against that
implementation (kept as `karpo_apple_silicon.reference.dense_attend`) and agrees to rounding: 1e-12 in float64
and one unit in the last place in half precision on random tensors. The published-results comparison has not
been re-run with 0.2.0.

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
- **The exact window size is not confirmed by the published-results comparison.** A window one token too short is
  indistinguishable from the port there. Upstream's tests above pin it at 1,024 tokens (the token itself and
  1,023 before it).
- Most of the differences that remain look like rounding, not a wrong rule. The one changed answer had a published
  probability of 0.49 and came out at 0.51. The largest difference (0.312 published, 0.416 here, one question
  of 39 in a 7,377-token request) came out at 0.295 when the same request was run in float16 on the same Mac:
  the two precisions on one machine disagree with each other by more than either disagrees with the published
  figure.
- **One difference is not rounding on the Mac's side.** One question in an 839-token request (0.379
  published) came out between 0.43 and 0.44 in bfloat16, in float16 and in the 0.1.0 MLX route below. Three
  different calculations on the Mac agree with each other and differ from the published figure by 0.05 to 0.06. The
  request is shorter than the window, so the window rule cannot be the cause; the cause is not known. One possible
  cause is the published run's torch 2.14.1+cu130 and transformers 5.21.0 on an RTX PRO 6000 Blackwell (its
  `environment.json`), against torch 2.15.0 and transformers 5.23.0 here; this has not been tested. The top
  answer did not change.
- Batching was checked separately: 72 of the requests (182 questions) sent in batches of 8 and of 24 gave no
  changed answer, and differed from the same requests sent one at a time on the same Mac by 0.0002 on average
  (largest 0.004).
- Not compared: the 12B (it runs through the same code and answers the example sensibly, but no per-request
  results are published for it), any request longer than 8,072 tokens, and images.

To repeat the comparison on your own Mac, see the header of
[scripts/compare_reference.py](scripts/compare_reference.py). It needs the checkpoint, MachinePulse's published
results file and a rebuild of the suite; none of the three ships with this repository.

## Experimental: MLX, for Macs with less memory

The PyTorch route above needs the whole 62.6 GB checkpoint in memory. `karpo_apple_silicon.mlx` is a second,
separate route for the 31B that runs the model's layers with Apple's [MLX](https://github.com/ml-explore/mlx)
on a compressed ("quantized") copy of the weights: 34.0 GB at 8 bits, 19.3 GB at 4 bits. It is more
code than the patch above, it is newer, and it covers less: **text requests only, the 31B only**. Treat it as
experimental.

Ready-made copies are on Hugging Face, so you do not need the full 62.6 GB checkpoint to use this route:

| Copy | Size | Download |
|---|---|---|
| 8-bit | 34.0 GB | [jonnyArt/karpo-decision-31B-mlx-8bit](https://huggingface.co/jonnyArt/karpo-decision-31B-mlx-8bit) |
| 4-bit | 19.3 GB | [jonnyArt/karpo-decision-31B-mlx-4bit](https://huggingface.co/jonnyArt/karpo-decision-31B-mlx-4bit) |

```bash
pip install "karpo-decision @ git+https://github.com/machinepulse-ai/karpo-decision@6600011eb5cf46232d056ec0717dc726146d29de"
pip install "karpo-apple-silicon[mlx] @ git+https://github.com/fluxcodestudio/karpo-apple-silicon"
hf download jonnyArt/karpo-decision-31B-mlx-8bit --local-dir karpo-decision-31B-mlx-8bit
```

Or convert your own copy from the full checkpoint. Run the script from a clone of this repository. It imports
`karpo_apple_silicon.mlx`, so install karpo-decision as above, the package with its `mlx` extra, and `safetensors`
for reading the source. The package installs `torch`, which is also needed to read the checkpoint tensors.

```bash
pip install -e ".[mlx]" safetensors
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
incomplete or was not written by the converter.

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

The 8-bit copy has not been run with version 0.2.0, and the comparison with MachinePulse's published results
below has not been repeated. The MLX timings in the Speed table were measured with version 0.1.0.

The version 0.1.0 MLX comparison against MachinePulse's published results:

| 31B with MLX, against the published NVIDIA results | Questions | Top answer changed | Mean difference | Largest difference |
|---|---:|---:|---:|---:|
| 8-bit (the same 340 requests) | 1,606 | 2 | 0.0016 | 0.079 |
| 4-bit (145 of the requests) | 681 | 3 | 0.0067 | 0.183 |

Six of the 1,606 questions differed by more than 0.05 at 8 bits (two for the PyTorch route). The 4-bit copy
moves answers about six times as much as the full checkpoint does; prefer 8 bits when it fits. With version
0.1.0, memory in use peaked at 24 GB for 8,000-token requests with the 4-bit copy (19.3 GB of weights); the
8-bit copy was not measured this way. For those results, the converter was checked against the folders used for these numbers on a
sample of tensors, not re-run in full. The converted folder is a derivative of MachinePulse's weights: if you
share one, keep their LICENSE and NOTICE with it.
This repository itself ships no weights; the converted copies linked above are hosted on Hugging Face with
MachinePulse's LICENSE and NOTICE.

## Known limits

- **Gemma family only (31B and 12B).** The Qwen-family Karpo Decision models (0.8B to 9B and 27B-act) use a different
  base model whose code calls the CUDA kernel in another place and needs a second CUDA-only package. This port
  does not touch that path, and those models do not run with it.
- **Text requests only have been run.** Requests with images go through code this patch does not change and
  that looks device-independent, but no image request has been run on a Mac. Treat images as untested.
- **The MLX route is text-only.** It refuses image requests and does not convert or run the vision tower. Its
  reworked loader and forward pass have been run with the real 4-bit weights, not with the 8-bit ones.
- **PyTorch inference with `Decider.decide` only.** karpo-decision's server, its FP8 and CUDA-graph fast paths, its
  Triton attention and anything to do with training are NVIDIA-only or untested here.
- **Memory still grows with the square of the request length.** Sliding-window layers no longer build a full
  table; their cost grows with length times the window. The model's global layers (10 of the 31B's 60, 8 of
  the 12B's 48) are upstream's own code and read the whole request, so quadratic memory remains in fewer layers.
  The longest request run here was 8,072 tokens; very long requests (the model accepts up to 32,768 tokens)
  may run out of memory or be very slow.
- **Slower than an NVIDIA GPU**, by roughly ten to twenty times (see Speed).
- **Tied to the pinned karpo-decision commit.** `install()` warns when pip records a different git commit, or
  when no commit is recorded and the version differs from 0.1.0. It rejects an incompatible `attend` signature.
  Upstream's version has not changed between commits so far, which is why the commit is compared.
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
To run them from a clone, install `pip install -e ".[dev,mlx]" safetensors` as well as karpo-decision above.

```bash
python -m unittest discover -s tests -v
```

They check the mask against a brute-force statement of the rule, the attention against a hand-written softmax
per query (grouped key/value heads and packed sequences of unequal length), half precision against float32
`attend`, the packed forward against `dense_attend` on a tiny random model, the patch fed by karpo-decision's
own plan against token positions, and that a query with no key to read is an error. MPS tests skip when unavailable.
`test_mlx_masks.py` checks global, sliding and final-marker masks against per-row masks on upstream block plans,
the sliding mask against the PyTorch patch, the shortest readable key run per chunk and rotary frequencies
against Transformers. `test_mlx_forward.py` compares marker states from a small random unquantized MLX model
with karpo-decision's packed forward, including chunks that cross block boundaries and the 31B's attention
head shapes. `test_mlx_convert.py` converts and loads small random checkpoints, checks single and batched text
requests, length and image errors, incomplete or altered folders, pre-weight configuration checks, a single-file
source, an appended modification notice, refused source tensors and bit widths, and leftover partial folders.
These tests do not run the published quantized weights.

To run the pinned upstream CPU suite with wrong-mask controls:

```bash
python scripts/check_upstream_suite.py PATH_TO_KARPO_DECISION_CHECKOUT
```

## Questions people ask

**Can I run Karpo Decision on a MacBook Pro, Mac Studio or Mac mini?** Yes, if it has Apple Silicon and enough
memory: about 96 GB for the 31B as shipped, more than the 24.0 GB of weights for the 12B, and less for the 31B
with the quantized MLX copies described above.

**Does it work on Intel Macs?** PyTorch 2.15.0 has no Intel macOS build, so this package cannot be installed there.

**Does it work on Linux or Windows without an NVIDIA GPU (CPU, AMD, Intel Arc)?** The patch is plain PyTorch
and is not tied to macOS. On CPU the real 12B has been run on an Apple Silicon Mac only; Linux and Windows, and
AMD and Intel GPUs, have not been checked at all.

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

Tested with karpo-decision 0.1.0 (commit 6600011eb5cf), torch 2.15.0 and transformers 5.23.0; the MLX path with mlx 0.32.4 and
mlx-lm 0.32.2.
