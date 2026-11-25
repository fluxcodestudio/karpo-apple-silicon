# Changelog

All notable changes to this project are documented here.

## 0.2.0 (unreleased)

### Added

- Added `uninstall()`, the independent `dense_attend` reference and a check against upstream's own Gemma tests.
- Added packed-forward and Apple GPU tests, Ruff checks and cross-platform CI.
- Added MLX mask, forward and converter checks on small random models, including strict folder validation.

### Changed

- Sliding-window attention now reads only the keys reachable by each band of queries. On the 12B a 7,464-token
  request is about 1.3 times as fast end to end; requests of about 1,000 tokens or fewer are unchanged.
- On the real 31B on a 96 GB MacBook Pro, 0.2.0 answers nine requests of up to 10,196 tokens in bfloat16 and in
  float16, and gives the same top answers as 0.1.0 on the eight requests both ran.
- The installer checks the recorded upstream commit and the original attention signature.
- Pinned to karpo-decision commit c52d3507a254, upstream's newest. Commit 6600011eb5cf, which has the same
  source files, is still accepted without a warning.
- Corrected the documentation on memory sizes, test coverage and measured speed.
- MLX `load()` accepts `max_length`; `decide()` accepts one request or a list, and `decide_many()` remains.
- MLX attention reads reachable keys in chunks of 256 query rows; the last layer runs only on marker rows. On
  the 4-bit 31B a 7,464-token request peaks at 20.9 GB instead of 23.3 GB; its time and top answers are unchanged.
  On the 8-bit copy a 10,196-token request peaks at 35.6 GB instead of 40.4 GB, with the same top answers.
- The MLX converter accepts sharded or single-file sources, only 8 or 4 bits, and appends a modification notice.

### Fixed

- CPU forward now works by lifting the CUDA-only kernel selection in upstream's global layers. On the real 12B,
  0.1.0 stopped on CPU and 0.2.0 matches the Apple GPU's answers.
- MLX loads refuse incomplete, altered or unreadable text weights instead of leaving a partial model.
- MLX conversion checks source names first and completes a validated partial folder before renaming it.

## 0.1.0 (2026-11-21)

- First release.
