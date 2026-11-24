# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Write a quantized MLX copy of a Karpo Decision 31B checkpoint's text layers (experimental).

    python scripts/convert_mlx.py path/to/karpo-decision-31B path/to/karpo-decision-31B-mlx-8bit --bits 8

Every projection matrix of the decoder layers is quantized (8 or 4 bits, groups of 64): seven in a sliding-window
layer and six in a global layer, where keys and values share one; 410 matrices in the 31B. Embeddings, norms and
layer scalars stay bfloat16; the readout stays float32; the vision tower is left out (the MLX path reads text
only). One source tensor is converted at a time, so no full-precision copy is staged. The copy takes 34.0 GB of
disk at 8 bits and 19.3 GB at 4 bits (the checkpoint itself is 62.6 GB).

The source is a folder of ``model-*.safetensors`` shards or one ``model.safetensors``. The copy is written to
``<target>.partial`` and renamed to ``<target>`` only once it holds exactly the tensors, by name and shape, that
``karpo_apple_silicon.mlx.load`` requires; ``mlx_config.json`` is written last. A source whose tensor names
are not the ones the loader requires is refused before anything is written; the shapes are checked once the
copy is written. A run that stops part-way leaves ``<target>.partial`` in place, and the next run refuses to
start until you remove it.

The output is a derivative of the Karpo Decision weights: if you share it, keep their LICENSE and NOTICE
with it (this script copies them, with the Gemma licence, into the output folder, and appends a notice of
the modification to NOTICE).
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from safetensors import safe_open

from karpo_apple_silicon import mlx as karpo_mlx

if TYPE_CHECKING:
    import torch

#: The bit widths the loader and the published copies were checked with.
BITS = (8, 4)
GROUP_SIZE = 64
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
COPIED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "decision_config.json",
          "readout.safetensors")
LICENCES = ("LICENSE", "NOTICE", "licenses")
#: The text layers' tensors in the checkpoint; everything else (the vision tower, its embedder) is left out.
TEXT_PREFIX = "language_model."
#: Appended to the copied NOTICE: the Apache licence asks that modified files say they were changed.
MODIFICATION_NOTICE = """\
---
Modification notice ({name})

This folder is a converted copy of MachinePulseAI/karpo-decision-31B. The decoder weights
were quantized to {bits} bits (group size {group_size}) and re-saved in MLX format with
scripts/convert_mlx.py from https://github.com/fluxcodestudio/karpo-apple-silicon.
The readout weights, tokenizer, chat template and decision configuration are unchanged.
The vision weights are not included. The conversion is unofficial and is not affiliated
with or endorsed by MachinePulse or Google.
"""


def is_projection(name: str) -> bool:
    """Whether the tensor ``name`` is one of a decoder layer's projection matrices, the ones quantized."""
    return name.endswith(".weight") and any(f".{part}.weight" in name for part in PROJECTIONS)


def to_mlx(tensor: torch.Tensor) -> mx.array:
    """A checkpoint tensor as a bfloat16 MLX array; exact for the bfloat16 the checkpoint is stored in."""
    return mx.array(tensor.float().numpy()).astype(mx.bfloat16)


def text_config(source: Path) -> dict[str, Any]:
    """The checkpoint's ``text_config`` with the global layers' head size and key/value head count under the
    names mlx-lm reads them by."""
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "gemma4":
        raise SystemExit(f"only the 31B (model_type gemma4) has been converted and checked, not "
                         f"{config.get('model_type')!r}")
    text = config["text_config"]
    per_layer = {(values["head_dim"], values["num_key_value_heads"]) for values in text["per_layer_config"].values()}
    if len(per_layer) != 1:
        raise SystemExit("the global layers do not share one head size and key/value head count")
    text["global_head_dim"], text["num_global_key_value_heads"] = per_layer.pop()
    return text


def weight_files(source: Path) -> list[Path]:
    """The checkpoint's weight files: its ``model-*.safetensors`` shards in order, or its one
    ``model.safetensors``."""
    shards = sorted(source.glob("model-*.safetensors"))
    single = source / "model.safetensors"
    if not shards and not single.is_file():
        raise SystemExit(f"{source} has neither model-*.safetensors shards nor model.safetensors")
    return shards or [single]


def written_names(names: list[str]) -> list[str]:
    """The tensors a source file's ``names`` become: a projection as its packed weight, scales and biases; any
    other text tensor under its own name; nothing for what is not a text tensor."""
    out = []
    for name in names:
        if name.startswith(TEXT_PREFIX):
            stem = name.removesuffix("weight")
            out += [name, stem + "scales", stem + "biases"] if is_projection(name) else [name]
    return out


def difference(written: dict[str, tuple[int, ...] | None], expected: dict[str, tuple[int, ...]]) -> str | None:
    """What separates the ``written`` tensors from the ones the loader expects, in words; None when they are the
    same. A written shape of None is not compared (the names alone, before anything is converted)."""
    missing = sorted(expected.keys() - written.keys())
    extra = sorted(written.keys() - expected.keys())
    shapes = sorted(name for name in expected.keys() & written.keys()
                    if written[name] is not None and written[name] != expected[name])
    parts = []
    if missing:
        parts.append(f"{len(missing)} missing (the first: {missing[0]})")
    if extra:
        parts.append(f"{len(extra)} the loader does not know (the first: {extra[0]})")
    if shapes:
        parts.append(f"{len(shapes)} of another shape (the first: {shapes[0]} is {written[shapes[0]]}, "
                     f"expected {expected[shapes[0]]})")
    return "; ".join(parts) or None


def convert(source: Path, target: Path, bits: int) -> None:
    """Write the ``bits``-bit MLX copy of the checkpoint in ``source`` to ``target``. ``SystemExit`` with the
    reason when the source cannot be converted or the result is not what ``karpo_apple_silicon.mlx.load``
    requires; ``target`` is created only by a run that succeeds."""
    partial = target.with_name(target.name + ".partial")
    if bits not in BITS:
        raise SystemExit(f"bits must be {BITS[0]} or {BITS[1]}, not {bits}")
    if target.exists():
        raise SystemExit(f"{target} already exists; nothing was written")
    if partial.exists():
        raise SystemExit(f"{partial} is left from a run that did not finish: remove it and run again")
    absent = [name for name in ("config.json", *COPIED) if not (source / name).is_file()]
    if absent:
        raise SystemExit(f"{source} has no {', '.join(absent)}")
    text = text_config(source)
    try:
        karpo_mlx.decision_config(source)
        expected = karpo_mlx.expected_tensors(text, bits, GROUP_SIZE)
    except ValueError as error:
        raise SystemExit(f"{source} cannot be converted: {error}") from error
    files = weight_files(source)
    # The names alone, before anything is converted: a wrong source fails in seconds, not after the last shard.
    planned: list[str] = []
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            planned += written_names(list(handle.keys()))
    if not planned:
        raise SystemExit(f"{source} holds no tensor named {TEXT_PREFIX}*: nothing to convert")
    repeated = sorted(name for name, count in Counter(planned).items() if count > 1)
    if repeated:
        raise SystemExit(f"{source} holds {len(repeated)} tensors in more than one file (the first: {repeated[0]})")
    problem = difference(dict.fromkeys(planned), expected)
    if problem:
        raise SystemExit(f"{source} does not hold the text layers the loader expects: {problem}")

    partial.mkdir(parents=True)
    written: dict[str, tuple[int, ...]] = {}
    for number, file in enumerate(files, start=1):
        weights: dict[str, mx.array] = {}
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for name in handle.keys():  # noqa: SIM118 - safetensors handle is not iterable
                if not name.startswith(TEXT_PREFIX):
                    continue
                tensor = to_mlx(handle.get_tensor(name))
                if is_projection(name):
                    packed, scales, biases = mx.quantize(tensor, GROUP_SIZE, bits)
                    stem = name.removesuffix("weight")
                    weights[name], weights[stem + "scales"], weights[stem + "biases"] = packed, scales, biases
                    mx.eval(packed, scales, biases)
                else:
                    weights[name] = tensor
                    mx.eval(tensor)
        if weights:
            mx.save_safetensors(str(partial / f"text-{number:05d}.safetensors"), weights)
            written.update({name: tuple(value.shape) for name, value in weights.items()})
            print(f"file {number}/{len(files)}: {len(weights)} tensors", flush=True)
    problem = difference(written, expected)
    if problem:
        raise SystemExit(f"the copy in {partial} is not what the loader expects: {problem}. Remove it before "
                         "running again.")

    for name in COPIED:
        shutil.copy2(source / name, partial / name)
    for name in LICENCES:
        path = source / name
        if path.is_dir():
            shutil.copytree(path, partial / name)
        elif path.exists():
            shutil.copy2(path, partial / name)
    notice = partial / "NOTICE"
    upstream = notice.read_text(encoding="utf-8").rstrip("\n") if notice.exists() else ""
    notice.write_text((upstream + "\n\n" if upstream else "")
                      + MODIFICATION_NOTICE.format(name=target.name, bits=bits, group_size=GROUP_SIZE),
                      encoding="utf-8")
    # Last: a folder with mlx_config.json is a finished one.
    (partial / "mlx_config.json").write_text(json.dumps({"bits": bits, "group_size": GROUP_SIZE, "text_config": text},
                                                        indent=2))
    partial.rename(target)
    print(f"wrote {target}: {len(written)} tensors", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the original checkpoint folder")
    parser.add_argument("target", type=Path, help="the folder to create")
    parser.add_argument("--bits", type=int, choices=BITS, default=8)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.target, arguments.bits)


if __name__ == "__main__":
    main()
