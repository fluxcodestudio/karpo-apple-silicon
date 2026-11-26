# Changelog

All notable changes to this project are documented here.

## 0.2.0 (2026-11-25)

### Added

- Added `uninstall()`, `global_attend`, the independent `dense_attend` and `dense_global` references and a
  check against upstream's own Gemma tests.
- Added packed-forward and Apple GPU tests, Ruff checks and cross-platform CI, which fails on an unexpected
  skip and builds the package.
- Added tests that no attention call exceeds its chunk of rows or band of keys, and that the MLX answer step
  matches a hand-written calculation.
- Added a security policy and a release checklist.
- Added MLX mask, forward and converter checks on small random models, including strict folder validation.

### Changed

- Sliding-window attention now reads only the keys reachable by each band of queries. On the 12B a 7,464-token
  request is about 1.3 times as fast end to end; requests of about 1,000 tokens or fewer are unchanged.
- On the real 31B on a 96 GB MacBook Pro, 0.2.0 answers nine requests of up to 10,196 tokens in bfloat16 and in
  float16, and gives the same top answers as 0.1.0 on the eight requests both ran.
- Against MachinePulse's published 31B results, 0.2.0 reproduces 0.1.0's agreement on the same 340 requests
  (1,606 questions): one top answer changed through PyTorch (mean difference 0.0012, largest 0.090) and two
  with the 8-bit MLX copy (0.0014, 0.078). The 4-bit copy was run, as with 0.1.0, on 145 of those requests:
  three of 681 top answers changed (0.0067, 0.176).
- `scripts/compare_reference.py` empties PyTorch's Apple GPU buffer cache after every request; without that
  a 31B run on a 96 GB Mac grew to 103 GB of memory and 40 GB of swap.
- The installer checks the recorded upstream commit and the signatures of both functions it replaces, before
  it swaps either.
- `TESTED_UPSTREAM_COMMIT` and `TESTED_UPSTREAM_COMMITS` are public names.
- The converter's functions raise `ValueError` or `FileNotFoundError`; only its command line exits. The
  comparison script refuses a `--variants` list in which float16 is not last and malformed `--per-cell` or
  `--control-cells` values.
- The `mlx` extra includes `safetensors` and allows only the checked mlx-lm minor version. The source archive
  includes the scripts, the example and every test module.
- Pinned to karpo-decision commit c52d3507a254, upstream's newest. Commit 6600011eb5cf, which has the same
  source files, is still accepted without a warning.
- Corrected the documentation on memory sizes, test coverage and measured speed.
- MLX `load()` accepts `max_length`; `decide()` accepts one request or a list, and `decide_many()` remains.
- MLX attention reads reachable keys in chunks of 256 query rows; the last layer runs only on marker rows. On
  the 4-bit 31B a 7,464-token request peaks at 20.9 GB instead of 23.3 GB; its time and top answers are unchanged.
  On the 8-bit copy a 10,196-token request peaks at 35.6 GB instead of 40.4 GB, with the same top answers.
- The MLX converter accepts sharded or single-file sources, only 8 or 4 bits, and appends a modification notice.

### Fixed

- Long requests are answered correctly through PyTorch. Without CUDA, karpo-decision's global layers made one
  attention call over a whole sequence; on a 96 GB Mac the real 12B answered a 23,627-token request with noise
  that way, with no error, and 0.1.0 has this defect. The port now replaces that call and makes it 256 query
  rows at a time: the same request is answered correctly in 122 seconds instead of 404, and the 12B and the
  31B answer requests of up to 31,489 tokens correctly (karpo-decision's default limit is 32,768 per sequence).
- The 31B's long requests no longer run short of memory on a 96 GB Mac: a 7,464-token request takes 83 seconds
  as shipped instead of 240 and a 10,196-token one 128 instead of 370, and no probability moved by as much as
  0.00005.
- CPU forward now works, because the replaced global-layer call no longer asks PyTorch for a CUDA-only kernel.
  On the real 12B, 0.1.0 stopped on CPU and 0.2.0 matches the Apple GPU's answers.
- MLX loads refuse incomplete, altered or unreadable text weights instead of leaving a partial model.
- MLX `load()` says what is wrong when a folder is not a converted copy (the original checkpoint, a missing
  `mlx_config.json` entry, no readout file) instead of stopping with a bare file or key error.
- MLX conversion checks source names first and completes a validated partial folder before renaming it.

## 0.1.0 (2026-11-21)

- First release.
