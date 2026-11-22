# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""Write a quantized MLX copy of a Karpo Decision 31B checkpoint's text layers (experimental).

    python scripts/convert_mlx.py path/to/karpo-decision-31B path/to/karpo-decision-31B-mlx-8bit --bits 8

The seven projection matrices of every decoder layer are quantized (8 or 4 bits, groups of 64); embeddings,
norms and layer scalars stay bfloat16; the readout stays float32; the vision tower is left out (the MLX
path reads text only). One source tensor is converted at a time, so no full-precision copy is staged.
Needs about 32 GB of free disk for 8 bits and 18 GB for 4 bits.

The output is a derivative of the Karpo Decision weights: if you share it, keep their LICENSE and NOTICE
with it (this script copies them, with the Gemma licence, into the output folder).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
from safetensors import safe_open

GROUP_SIZE = 64
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
COPIED = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "decision_config.json",
          "readout.safetensors")
LICENCES = ("LICENSE", "NOTICE", "licenses")


def is_projection(name: str) -> bool:
    return name.endswith(".weight") and any(f".{part}.weight" in name for part in PROJECTIONS)


def to_mlx(tensor) -> mx.array:
    return mx.array(tensor.float().numpy()).astype(mx.bfloat16)


def text_config(source: Path) -> dict:
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


def convert(source: Path, target: Path, bits: int) -> None:
    text = text_config(source)
    target.mkdir(parents=True, exist_ok=False)
    (target / "mlx_config.json").write_text(json.dumps({"bits": bits, "group_size": GROUP_SIZE, "text_config": text},
                                                       indent=2))
    for name in COPIED:
        shutil.copy2(source / name, target / name)
    for name in LICENCES:
        path = source / name
        if path.is_dir():
            shutil.copytree(path, target / name)
        elif path.exists():
            shutil.copy2(path, target / name)
    shards = sorted(source.glob("model-*.safetensors"))
    for number, shard in enumerate(shards, start=1):
        weights = {}
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if not name.startswith("language_model."):
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
            mx.save_safetensors(str(target / f"text-{number:05d}.safetensors"), weights)
            print(f"shard {number}/{len(shards)}: {len(weights)} tensors", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="the original checkpoint folder")
    parser.add_argument("target", type=Path, help="the folder to create")
    parser.add_argument("--bits", type=int, choices=(8, 4), default=8)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.target, arguments.bits)
