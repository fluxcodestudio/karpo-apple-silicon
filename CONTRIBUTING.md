# Contributing

Clone the repository and set up Python 3.12 or later:

```bash
git clone https://github.com/fluxcodestudio/karpo-apple-silicon
cd karpo-apple-silicon
python3.12 -m venv .venv
source .venv/bin/activate
pip install "karpo-decision @ git+https://github.com/machinepulse-ai/karpo-decision@c52d3507a254d3ad8423726c6bc6568c99113e7d"
pip install -e ".[dev,mlx]"
```

Before a pull request, run the unit tests, the upstream suite against a checkout at the pinned commit, and Ruff.
The `upstream` folder is ignored by git and by Ruff:

```bash
git clone https://github.com/machinepulse-ai/karpo-decision upstream
git -C upstream checkout c52d3507a254d3ad8423726c6bc6568c99113e7d
python -m unittest discover -s tests -v
python scripts/check_upstream_suite.py upstream
python -m ruff check .
```

If you change the attention rule, re-run the published-results comparison described in the README and update its
numbers. Report an answer that differs from an NVIDIA GPU in an issue with the request, both responses and
probabilities, checkpoint and conversion and Mac details, and the relevant package versions listed in the README.

For MLX changes, `tests/test_mlx_masks.py` checks masks, chunks and rotary frequencies;
`tests/test_mlx_forward.py` compares small random unquantized models with karpo-decision; and
`tests/test_mlx_convert.py` checks small random conversions, loading, refusal of invalid folders and the
answers of `decide()` against a hand-written calculation. These checks do not run the published quantized
weights. `tests/_support.py` holds what the test modules share. Run the tests through `discover -s tests`, as
above, so that it is found.

## Releasing

1. Run the three checks above on an Apple Silicon Mac. GitHub's hosted macOS runners report no Apple GPU, so the
   tests that need one are skipped there.
2. Answer a request with real weights on each route you touched, and compare with an earlier run. For a change
   to either attention function, repeat the published-results comparison.
3. Set the version in `karpo_apple_silicon/__init__.py`, `CITATION.cff` and `llms.txt`, the release date in
   `CHANGELOG.md` and `CITATION.cff`, and the version and tag in the install sections of `README.md` and
   `llms.txt`.
4. Merge to `main`, tag that commit `vX.Y.Z`, and publish the GitHub release from the changelog entry.
