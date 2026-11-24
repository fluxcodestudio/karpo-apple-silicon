# Copyright 2026 FluxCode Studio and the karpo-apple-silicon contributors
# SPDX-License-Identifier: Apache-2.0
"""``scripts/convert_mlx.py`` and the MLX loader on a tiny random checkpoint written to a temporary folder, with a
character-level tokenizer built on the spot: nothing is downloaded. Needs mlx, mlx-lm, karpo-decision and
transformers' Gemma 4. Skipped where MLX cannot start (no Metal device: sandboxes, some CI runners)."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import string
import tempfile
import unittest
from pathlib import Path


def unavailable() -> str | None:
    """Why the MLX path cannot be tested here, or None."""
    for name in ("mlx", "mlx_lm", "karpo_decision", "transformers.models.gemma4", "tokenizers", "safetensors"):
        try:
            found = importlib.util.find_spec(name)
        except ModuleNotFoundError:
            found = None
        if found is None:
            return f"needs {name}"
    try:
        import mlx.core  # noqa: F401
    except Exception as error:  # MLX raises on import when there is no Metal device
        return f"MLX cannot start: {error}"
    return None


SKIP = unavailable()
SPECIALS = ["<pad>", "<unk>", "<bos>", "<mask>", "<start_of_turn>", "<end_of_turn>"]
#: One token per character: printable ASCII, the chat template's hole and the brackets request text is escaped with.
CHARACTERS = sorted(set(string.printable) | {"\x00", "\uff1c", "\uff1e"})
VOCAB = {token: index for index, token in enumerate(SPECIALS + CHARACTERS)}
CODES = list("ABCDEFGH")
CHAT = ("{{ bos_token }}{% for m in messages %}<start_of_turn>{{ m['role'] }}\n{{ m['content'] }}<end_of_turn>\n"
        "{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}")
TYPES = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
REQUEST = {"state": {"light": "red", "note": "<mask> is escaped"},
           "questions": {"go": {"type": "noul", "instructions": "May the car go?"},
                         "next": {"type": "choice", "criteria": {"stop": "Stay", "drive": None, "turn": "Turn back"}}}}


def converter():
    """``scripts/convert_mlx.py`` as a module (the scripts folder is not a package)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "convert_mlx.py"
    spec = importlib.util.spec_from_file_location("convert_mlx", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_checkpoint(folder: Path, prefix: str = "language_model.", single: bool = False, notice: bool = True) -> dict:
    """A checkpoint folder as karpo-decision saves one: a random Gemma 4 text model in bfloat16 (two shards, or one
    ``model.safetensors``), a vision tensor that must be left out, readout, tokenizer and both configurations.
    Returns the saved tensors."""
    import tokenizers
    import torch
    from karpo_decision.config import DecisionConfig
    from safetensors.torch import save_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    folder.mkdir()
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split(tokenizers.Regex(r"[\s\S]"), behavior="isolated")
    tokenizer.add_special_tokens(SPECIALS)
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<bos>", pad_token="<pad>", unk_token="<unk>",
                            mask_token="<mask>", additional_special_tokens=SPECIALS[4:],
                            chat_template=CHAT).save_pretrained(str(folder))
    (folder / "chat_template.jinja").write_text(CHAT)
    torch.manual_seed(0)
    config = Gemma4TextConfig(vocab_size=len(VOCAB), hidden_size=64, intermediate_size=128,
                              num_hidden_layers=len(TYPES), num_attention_heads=8, num_key_value_heads=4, head_dim=8,
                              global_head_dim=16, num_global_key_value_heads=1, attention_k_eq_v=True,
                              sliding_window=16, layer_types=TYPES, hidden_size_per_layer_input=0,
                              final_logit_softcapping=30.0)
    (folder / "config.json").write_text(json.dumps({"model_type": "gemma4", "text_config": config.to_dict()}))
    DecisionConfig(template="karpo-blocks-v1", codes=tuple(CODES), token_ids=tuple(VOCAB[code] for code in CODES),
                   attention="bidirectional", base_model="tiny-random").save(folder)
    save_file({"weight": torch.randn(len(CODES), 64)}, str(folder / "readout.safetensors"))
    state = {prefix + name: value.to(torch.bfloat16) for name, value in Gemma4TextModel(config).state_dict().items()}
    names = sorted(state)
    vision = {"vision_tower.encoder.layers.0.self_attn.q_proj.linear.weight": torch.zeros(4, 4, dtype=torch.bfloat16)}
    if single:
        save_file({**state, **vision}, str(folder / "model.safetensors"))
    else:
        save_file({name: state[name] for name in names[:30]}, str(folder / "model-00001-of-00002.safetensors"))
        save_file({**{name: state[name] for name in names[30:]}, **vision},
                  str(folder / "model-00002-of-00002.safetensors"))
    if notice:
        (folder / "NOTICE").write_text("Upstream notice.\n")
    return state


@unittest.skipIf(SKIP, SKIP)
class ConvertAndLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.convert = converter()
        cls.work = Path(cls.enterClassContext(tempfile.TemporaryDirectory()))
        cls.state = write_checkpoint(cls.work / "source")
        cls.target = cls.work / "tiny-mlx-8bit"
        with contextlib.redirect_stdout(io.StringIO()):
            cls.convert.convert(cls.work / "source", cls.target, 8)

    def run_convert(self, source: Path, target: Path) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            self.convert.convert(source, target, 8)

    def test_the_folder_is_complete_and_loads(self):
        import mlx.core as mx

        from karpo_apple_silicon import mlx as karpo_mlx

        self.assertFalse((self.work / "tiny-mlx-8bit.partial").exists())
        self.assertEqual(sorted(p.name for p in self.target.glob("text-*")),
                         ["text-00001.safetensors", "text-00002.safetensors"])
        self.assertEqual((self.target / "NOTICE").read_text(), "Upstream notice.\n\n" + self.convert.MODIFICATION_NOTICE
                         .format(name="tiny-mlx-8bit", bits=8, group_size=64))
        self.assertIn("quantized to 8 bits (group size 64)", (self.target / "NOTICE").read_text())
        decider = karpo_mlx.load(self.target)
        layers = decider.model.language_model.layers
        self.assertEqual(layers[0].self_attn.q_proj.weight.dtype, mx.uint32)  # quantized
        self.assertFalse(hasattr(layers[1].self_attn, "v_proj"))  # a global layer: keys and values share one
        # every tensor is the checkpoint's own: an embedding row exactly, a projection to within its 8 bits
        embedding = self.state["language_model.embed_tokens.weight"].float().numpy()
        self.assertTrue((mx.abs(decider.model.language_model.embed_tokens.weight.astype(mx.float32)
                                - mx.array(embedding)).max() == 0).item())
        projection = layers[2].mlp.down_proj
        restored = mx.dequantize(projection.weight, projection.scales, projection.biases, 64, 8).astype(mx.float32)
        original = mx.array(self.state["language_model.layers.2.mlp.down_proj.weight"].float().numpy())
        self.assertLess((mx.abs(restored - original).max() / mx.abs(original).max()).item(), 0.02)

        one = decider.decide(REQUEST)
        self.assertEqual(list(one["answers"]), ["go", "next"])
        self.assertEqual(one["answers"]["next"]["type"], "choice")
        self.assertAlmostEqual(sum(one["answers"]["next"]["probabilities"].values()), 1.0, places=6)
        self.assertEqual(one["usage"]["sequences"], 1)
        both = decider.decide([REQUEST, REQUEST])  # a list for a list
        self.assertEqual(len(both), 2)
        for response in (*both, *decider.decide_many([REQUEST])):  # packed with another: equal to bfloat16 rounding
            self.assertEqual(response["usage"], one["usage"])
            for key in ("stop", "drive", "turn"):
                self.assertAlmostEqual(response["answers"]["next"]["probabilities"][key],
                                       one["answers"]["next"]["probabilities"][key], delta=0.02)
            self.assertAlmostEqual(response["answers"]["go"]["noul"], one["answers"]["go"]["noul"], delta=0.02)
        self.assertEqual(decider.decide([]), [])

    def test_requests_the_folder_cannot_answer_raise_karpo_decisions_errors(self):
        from karpo_decision.encoding import LengthError, TextOnlyError

        from karpo_apple_silicon import mlx as karpo_mlx

        decider = karpo_mlx.load(self.target)
        with self.assertRaises(TextOnlyError):  # raised by karpo-decision's encoder, before any forward pass
            decider.decide({**REQUEST, "images": ["data:image/png;base64,AAAA"]})
        shared = decider.encoder.plan(REQUEST, None)[0].length
        split = karpo_mlx.load(self.target, max_length=shared - 20)  # room for one branch at a time
        self.assertEqual(split.decide(REQUEST)["usage"]["sequences"], 2)
        with self.assertRaises(LengthError):
            karpo_mlx.load(self.target, max_length=50).decide(REQUEST)

    def test_an_incomplete_or_altered_folder_fails_the_load(self):
        import mlx.core as mx

        from karpo_apple_silicon import mlx as karpo_mlx

        def cut_short(folder: Path) -> None:
            shard = folder / "text-00002.safetensors"
            shard.write_bytes(shard.read_bytes()[:shard.stat().st_size // 2])

        for name, damage in (("a shard removed", lambda folder: (folder / "text-00002.safetensors").unlink()),
                             ("a shard cut short", cut_short),
                             ("a tensor nobody expects", lambda folder: mx.save_safetensors(
                                 str(folder / "text-00003.safetensors"), {"language_model.extra": mx.zeros((1,))})),
                             ("no weights at all", lambda folder: [p.unlink() for p in folder.glob("text-*")])):
            folder = self.work / name
            shutil.copytree(self.target, folder)
            damage(folder)
            with self.subTest(name), self.assertRaisesRegex(
                    ValueError, "is incomplete or was not written by scripts/convert_mlx.py") as raised:
                karpo_mlx.load(folder)
            self.assertIn(str(folder), str(raised.exception))
            # MLX's own message is the cause: a ValueError from the strict load, a RuntimeError from reading a file
            self.assertIsInstance(raised.exception.__cause__, RuntimeError if damage is cut_short else ValueError)

    def test_the_configuration_is_checked_before_any_weight_is_read(self):
        """A folder the loader cannot answer with is refused for what is wrong with it, although the folder
        holds no weights either: a causal checkpoint, codes that are not the tokenizer's, more layer types than
        layers, a readout of another size."""
        import mlx.core as mx

        from karpo_apple_silicon import mlx as karpo_mlx

        folder = self.work / "causal"
        shutil.copytree(self.target, folder, ignore=shutil.ignore_patterns("text-*"))
        decision = json.loads((folder / "decision_config.json").read_text())
        (folder / "decision_config.json").write_text(json.dumps({**decision, "attention": "causal"}))
        with self.assertRaisesRegex(ValueError, "only bidirectional checkpoints"):
            karpo_mlx.load(folder)
        (folder / "decision_config.json").write_text(json.dumps({**decision, "token_ids": decision["token_ids"][::-1]}))
        with self.assertRaisesRegex(ValueError, "is not token"):
            karpo_mlx.load(folder)
        (folder / "decision_config.json").write_text(json.dumps(decision))
        metadata = json.loads((folder / "mlx_config.json").read_text())
        text = metadata["text_config"]
        longer = {**text, "layer_types": [*TYPES, "full_attention"]}  # the last type is not the last layer's
        (folder / "mlx_config.json").write_text(json.dumps({**metadata, "text_config": longer}))
        with self.assertRaisesRegex(ValueError, f"{len(TYPES) + 1} layer types for {len(TYPES)} layers"):
            karpo_mlx.load(folder)
        (folder / "mlx_config.json").write_text(json.dumps(metadata))
        mx.save_safetensors(str(folder / "readout.safetensors"), {"weight": mx.zeros((len(CODES), 8))})
        with self.assertRaisesRegex(ValueError, r"readout.safetensors holds a weight of \(8, 8\)"):
            karpo_mlx.load(folder)
        mx.save_safetensors(str(folder / "readout.safetensors"), {"rows": mx.zeros((len(CODES), 64))})
        with self.assertRaisesRegex(ValueError, "readout.safetensors holds no weight"):
            karpo_mlx.load(folder)

    def test_one_file_source_and_no_upstream_notice(self):
        from karpo_apple_silicon import mlx as karpo_mlx

        write_checkpoint(self.work / "single", single=True, notice=False)
        self.run_convert(self.work / "single", self.work / "single-mlx")
        self.assertEqual([p.name for p in (self.work / "single-mlx").glob("text-*")], ["text-00001.safetensors"])
        notice = (self.work / "single-mlx" / "NOTICE").read_text()
        self.assertTrue(notice.startswith("---\nModification notice (single-mlx)"), notice[:60])
        karpo_mlx.load(self.work / "single-mlx")

    def test_a_source_the_loader_could_not_read_is_refused_before_anything_is_written(self):
        write_checkpoint(self.work / "prefixed", prefix="model.language_model.")
        with self.assertRaisesRegex(SystemExit, "holds no tensor named language_model"):
            self.run_convert(self.work / "prefixed", self.work / "prefixed-mlx")
        (self.work / "source-short").mkdir()
        for path in (self.work / "source").iterdir():
            if path.name != "model-00002-of-00002.safetensors":
                shutil.copy2(path, self.work / "source-short" / path.name)
        with self.assertRaisesRegex(SystemExit, "missing"):
            self.run_convert(self.work / "source-short", self.work / "short-mlx")
        for name in ("prefixed-mlx", "prefixed-mlx.partial", "short-mlx", "short-mlx.partial"):
            self.assertFalse((self.work / name).exists(), name)

    def test_a_bit_width_the_loader_was_not_checked_with_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "bits must be 8 or 4, not 6"):
            self.convert.convert(self.work / "source", self.work / "six-bit", 6)
        for name in ("six-bit", "six-bit.partial"):
            self.assertFalse((self.work / name).exists(), name)

    def test_an_existing_target_and_a_leftover_partial_folder_are_refused_and_kept(self):
        with self.assertRaisesRegex(SystemExit, "already exists"):
            self.run_convert(self.work / "source", self.target)
        leftover = self.work / "again.partial"
        leftover.mkdir()
        (leftover / "text-00001.safetensors").write_bytes(b"half a file")
        with self.assertRaisesRegex(SystemExit, "remove it and run again"):
            self.run_convert(self.work / "source", self.work / "again")
        self.assertEqual((leftover / "text-00001.safetensors").read_bytes(), b"half a file")
        self.assertFalse((self.work / "again").exists())


if __name__ == "__main__":
    unittest.main()
