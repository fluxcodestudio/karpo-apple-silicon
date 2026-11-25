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

Before a pull request, run the unit tests, the upstream suite against a checkout at the pinned commit, and Ruff:

```bash
python -m unittest discover -s tests -v
python scripts/check_upstream_suite.py PATH_TO_KARPO_DECISION_CHECKOUT
python -m ruff check .
```

If you change the attention rule, re-run the published-results comparison described in the README and update its
numbers. Report an answer that differs from an NVIDIA GPU in an issue with the request, both responses and
probabilities, checkpoint and conversion and Mac details, and the relevant package versions listed in the README.

For MLX changes, `tests/test_mlx_masks.py` checks masks, chunks and rotary frequencies;
`tests/test_mlx_forward.py` compares small random unquantized models with karpo-decision; and
`tests/test_mlx_convert.py` checks small random conversions, loading and refusal of invalid folders. These
checks do not run the published quantized weights. The converter also needs `safetensors` to read source files.
