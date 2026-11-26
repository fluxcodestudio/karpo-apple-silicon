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
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from _support import mlx_unavailable

SKIP = mlx_unavailable("transformers.models.gemma4", "tokenizers", "safetensors")
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
#: Another state, and a question of the third type beside the two above.
SCORED = {"state": {"light": "amber", "note": "the road is wet"},
          "questions": {**REQUEST["questions"],
                        "urgency": {"type": "score", "instructions": "How urgent is it?",
                                    "criteria": ["not at all", "somewhat", "very", "extremely"]}}}
#: A soft cap and temperatures that move the answers. The checkpoint's own cap of 30 and temperatures of 1.0 leave
#: them nearly or exactly where they were, so a decider that skipped either step would still pass with those.
CAP = 4.0
TEMPERATURES = {"choice": 0.5, "noul": 2.0, "score": 1.5}


def by_hand(hidden: np.ndarray, readout: np.ndarray, requests: list[dict], cap: float | None,
            temperatures: dict[str, float]) -> list[dict[str, np.ndarray]]:
    """Per request, each question's probabilities over its options, by the rule karpo-decision's ``Readout`` and
    ``Decider.decide`` state, in float64: the logits are the hidden states at the readout rows times the readout
    rows, soft-capped as ``cap * tanh(logits / cap)``; a question reads the first codes of its row, one per
    option, divides them by its type's temperature and takes their softmax. One sequence per request, its
    readout rows in question order."""
    logits = hidden @ readout.T
    if cap is not None:
        logits = cap * np.tanh(logits / cap)
    rows = iter(logits)
    answers = []
    for request in requests:
        answers.append({})
        for key, question in request["questions"].items():
            options = 2 if question["type"] == "noul" else len(question["criteria"])
            scaled = next(rows)[:options] / temperatures[question["type"]]
            weights = np.exp(scaled - scaled.max())
            answers[-1][key] = weights / weights.sum()
    return answers


def numbers(response: dict) -> list[float]:
    """Every probability a response states, in question and option order (a noul answer states one, of true)."""
    return [value for answer in response["answers"].values()
            for value in ([answer["noul"]] if answer["type"] == "noul" else answer["probabilities"].values())]


@contextlib.contextmanager
def forward_passes():
    """Record every forward pass made while the block runs: its sequences' lengths, and the hidden states it
    returned at their readout rows, bfloat16 values that float64 holds exactly."""
    import mlx.core as mx

    from karpo_apple_silicon import mlx as karpo_mlx

    real = karpo_mlx._marker_states
    passes: list[tuple[list[int], np.ndarray]] = []

    def recorded(model, sequences, window, rope):
        states = real(model, sequences, window, rope)
        passes.append(([part.length for part in sequences], np.array(states.astype(mx.float32)).astype(np.float64)))
        return states

    with mock.patch.object(karpo_mlx, "_marker_states", recorded):
        yield passes


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
        self.assertEqual((decider.token_budget, karpo_mlx.DEFAULT_TOKEN_BUDGET), (4096, 4096))
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
        parts = split.encoder.plan(REQUEST, split.max_length)
        self.assertEqual(split.decide(REQUEST)["usage"],  # each part repeats the state: more tokens than unsplit
                         {"input_tokens": sum(part.length for part in parts), "output_tokens": 0, "sequences": 2})
        self.assertGreater(sum(part.length for part in parts), shared)
        with self.assertRaises(LengthError):
            karpo_mlx.load(self.target, max_length=50).decide(REQUEST)

    def test_answers_are_karpo_decisions_readout_rule_worked_by_hand(self):
        """``decide`` against ``by_hand`` on the hidden states of the same forward pass, with a cap and
        temperatures that matter, for all three question types and both readout precisions.

        With a float32 readout nothing is rounded to bfloat16 after the hidden states, so only float32
        arithmetic separates the two: 5.7e-8 when measured, 1e-5 allowed. With a bfloat16 readout the product
        itself is computed in bfloat16, eight bits of mantissa, where the oracle multiplies the same rounded
        rows exactly: 4.3e-4 when measured, 5e-3 allowed."""
        import torch
        from safetensors.numpy import load_file

        from karpo_apple_silicon import mlx as karpo_mlx

        requests = [REQUEST, SCORED]
        for readout_dtype, tolerance in (("float32", 1e-5), ("bfloat16", 5e-3)):
            folder = self.work / f"by hand {readout_dtype}"
            shutil.copytree(self.target, folder)
            metadata = json.loads((folder / "mlx_config.json").read_text())
            metadata["text_config"]["final_logit_softcapping"] = CAP
            (folder / "mlx_config.json").write_text(json.dumps(metadata))
            decision = json.loads((folder / "decision_config.json").read_text())
            (folder / "decision_config.json").write_text(json.dumps({**decision, "temperatures": TEMPERATURES,
                                                                    "readout_dtype": readout_dtype}))
            decider = karpo_mlx.load(folder)
            with forward_passes() as passes:
                responses = decider.decide(requests)
            hidden = np.concatenate([states for _, states in passes])
            readout = torch.from_numpy(load_file(str(folder / "readout.safetensors"))["weight"])
            readout = readout.to(getattr(torch, readout_dtype)).double().numpy()  # the rows as the decider holds them
            want = by_hand(hidden, readout, requests, CAP, TEMPERATURES)
            for request, response, expected in zip(requests, responses, want, strict=True):
                plan = decider.encoder.plan(request, decider.max_length)
                self.assertEqual([part.keys for part in plan], [tuple(request["questions"])])  # as by_hand reads
                self.assertEqual(response["usage"], {"input_tokens": plan[0].length, "output_tokens": 0,
                                                     "sequences": 1})
                for key, question in request["questions"].items():
                    answer, probabilities = response["answers"][key], expected[key]
                    with self.subTest(readout=readout_dtype, question=key):
                        if question["type"] == "noul":
                            self.assertAlmostEqual(answer["noul"], probabilities[1], delta=tolerance)
                            continue
                        levels = np.arange(len(probabilities))
                        names = list(question["criteria"]) if question["type"] == "choice" else [str(i) for i in levels]
                        self.assertEqual(list(answer["probabilities"]), names)
                        np.testing.assert_allclose(list(answer["probabilities"].values()), probabilities, rtol=0,
                                                   atol=tolerance)
                        if question["type"] == "choice":
                            self.assertEqual(answer["choice"], names[int(probabilities.argmax())])
                        else:  # the expected level: each probability's error counts once per level
                            self.assertAlmostEqual(answer["score"], float(probabilities @ levels),
                                                   delta=levels.sum() * tolerance)
            # Left out of the rule, either step moves a probability by far more than is allowed above.
            ones = dict.fromkeys(TEMPERATURES, 1.0)
            for step, without in (("the cap", by_hand(hidden, readout, requests, None, TEMPERATURES)),
                                  ("the temperatures", by_hand(hidden, readout, requests, CAP, ones))):
                moved = max(np.abs(a[key] - b[key]).max() for a, b in zip(want, without, strict=True) for key in a)
                self.assertGreater(moved, 0.1, f"{step} would not show with a {readout_dtype} readout")

    def test_the_token_budget_decides_which_sequences_share_a_forward_pass(self):
        """Three requests of one length, ``L`` tokens each: together under the default budget; two and one when
        the budget is exactly ``2 L``; alone when it is one token less, and when it is 1 (a sequence longer than
        the budget still runs). The answers do not depend on it beyond bfloat16 rounding in the layers, which
        differs with the rows a sequence shares its attention calls with: 2.5e-4 when measured, ten times that
        allowed."""
        from karpo_apple_silicon import mlx as karpo_mlx

        requests = [{**REQUEST, "state": {**REQUEST["state"], "light": light}} for light in ("red", "tan", "sky")]
        decider = karpo_mlx.load(self.target)
        lengths = [decider.encoder.plan(request, decider.max_length)[0].length for request in requests]
        length = lengths[0]
        self.assertEqual(lengths, [length] * 3)
        with forward_passes() as passes:
            together = decider.decide(requests)
        self.assertEqual([sizes for sizes, _ in passes], [[length] * 3])
        self.assertNotEqual(numbers(together[0]), numbers(together[1]))  # the three are different requests
        for budget, expected in ((2 * length, [[length, length], [length]]), (2 * length - 1, [[length]] * 3),
                                 (1, [[length]] * 3)):
            with self.subTest(budget=budget):
                with forward_passes() as passes:
                    responses = karpo_mlx.load(self.target, token_budget=budget).decide(requests)
                self.assertEqual([sizes for sizes, _ in passes], expected)
                for response, packed in zip(responses, together, strict=True):
                    self.assertEqual(response["usage"], packed["usage"])
                    np.testing.assert_allclose(numbers(response), numbers(packed), rtol=0, atol=2.5e-3)

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

    def test_a_tensor_in_two_files_is_reported_as_that(self):
        """Not as an incomplete folder: every tensor is there, one of them twice."""
        import mlx.core as mx

        from karpo_apple_silicon import mlx as karpo_mlx

        folder = self.work / "a tensor twice"
        shutil.copytree(self.target, folder)
        name, value = min(mx.load(str(folder / "text-00001.safetensors")).items())
        mx.save_safetensors(str(folder / "text-00003.safetensors"), {name: value})
        with self.assertRaisesRegex(ValueError, "1 tensors are in more than one file") as raised:
            karpo_mlx.load(folder)
        self.assertIn(str(folder), str(raised.exception))
        self.assertIn(name, str(raised.exception))
        self.assertNotIn("is incomplete", str(raised.exception))

    def test_a_folder_that_is_not_a_converted_copy_says_so(self):
        """The original checkpoint, a folder cut off before its configuration was written, a configuration
        with an entry missing and an unknown readout precision are each a ``ValueError`` naming the problem."""
        from karpo_apple_silicon import mlx as karpo_mlx

        with self.assertRaisesRegex(ValueError, "has no mlx_config.json") as raised:
            karpo_mlx.load(self.work / "source")                    # the checkpoint the copy was made from
        self.assertIn("scripts/convert_mlx.py", str(raised.exception))
        folder = self.work / "small files"
        shutil.copytree(self.target, folder, ignore=shutil.ignore_patterns("text-*"))
        metadata = json.loads((folder / "mlx_config.json").read_text())
        (folder / "mlx_config.json").write_text(json.dumps({k: v for k, v in metadata.items() if k != "group_size"}))
        with self.assertRaisesRegex(ValueError, "mlx_config.json has no group_size"):
            karpo_mlx.load(folder)
        (folder / "mlx_config.json").write_text(json.dumps(metadata))
        decision = json.loads((folder / "decision_config.json").read_text())
        (folder / "decision_config.json").write_text(json.dumps({**decision, "readout_dtype": "float8"}))
        with self.assertRaisesRegex(ValueError, "readout_dtype 'float8'"):
            karpo_mlx.load(folder)
        (folder / "decision_config.json").write_text(json.dumps(decision))
        (folder / "readout.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "has no readout.safetensors"):
            karpo_mlx.load(folder)

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
        with self.assertRaisesRegex(ValueError, "holds no tensor named language_model"):
            self.run_convert(self.work / "prefixed", self.work / "prefixed-mlx")
        (self.work / "source-short").mkdir()
        for path in (self.work / "source").iterdir():
            if path.name != "model-00002-of-00002.safetensors":
                shutil.copy2(path, self.work / "source-short" / path.name)
        with self.assertRaisesRegex(ValueError, "missing"):
            self.run_convert(self.work / "source-short", self.work / "short-mlx")
        for name in ("prefixed-mlx", "prefixed-mlx.partial", "short-mlx", "short-mlx.partial"):
            self.assertFalse((self.work / name).exists(), name)

    def test_a_bit_width_the_loader_was_not_checked_with_is_refused(self):
        with self.assertRaisesRegex(ValueError, "bits must be 8 or 4, not 6"):
            self.convert.convert(self.work / "source", self.work / "six-bit", 6)
        for name in ("six-bit", "six-bit.partial"):
            self.assertFalse((self.work / name).exists(), name)

    def test_an_existing_target_and_a_leftover_partial_folder_are_refused_and_kept(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_convert(self.work / "source", self.target)
        leftover = self.work / "again.partial"
        leftover.mkdir()
        (leftover / "text-00001.safetensors").write_bytes(b"half a file")
        with self.assertRaisesRegex(ValueError, "remove it and run again"):
            self.run_convert(self.work / "source", self.work / "again")
        self.assertEqual((leftover / "text-00001.safetensors").read_bytes(), b"half a file")
        self.assertFalse((self.work / "again").exists())

    def test_missing_files_are_file_not_found_errors(self):
        """A source without its small files, and one without weight files, name what is missing."""
        empty = self.work / "no checkpoint"
        empty.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, "has no config.json, tokenizer.json"):
            self.run_convert(empty, self.work / "from-nothing")
        with self.assertRaisesRegex(FileNotFoundError, "has neither model-.* shards nor model.safetensors"):
            self.convert.weight_files(empty)
        for name in ("from-nothing", "from-nothing.partial"):
            self.assertFalse((self.work / name).exists(), name)

    def test_a_checkpoint_that_cannot_be_converted_is_a_value_error(self):
        """Another model type, global layers of two head sizes, a causal checkpoint and a tensor in two files are
        refused before anything is written. A tensor of another shape is found once the copy is written, and the
        copy stays in ``.partial`` for the next run to refuse."""
        import torch
        from safetensors.torch import load_file, save_file

        def altered(name: str, file: str | None = None, change=None) -> Path:
            folder = self.work / name
            shutil.copytree(self.work / "source", folder)
            if file:
                (folder / file).write_text(json.dumps(change(json.loads((folder / file).read_text()))))
            return folder

        def two_head_sizes(config: dict) -> dict:
            config["text_config"]["per_layer_config"]["1"]["head_dim"] = 8
            return config

        first, third = "model-00001-of-00002.safetensors", "model-00003-of-00002.safetensors"
        embedding = "language_model.embed_tokens.weight"
        twice = altered("a source tensor twice")
        save_file({embedding: load_file(str(twice / first))[embedding]}, str(twice / third))
        for folder, pattern in (
                (altered("another model type", "config.json", lambda config: {**config, "model_type": "gemma3"}),
                 r"only the 31B \(model_type gemma4\) has been converted and checked, not 'gemma3'"),
                (altered("two head sizes", "config.json", two_head_sizes), "do not share one head size"),
                (altered("a causal source", "decision_config.json", lambda config: {**config, "attention": "causal"}),
                 "cannot be converted: only bidirectional checkpoints"),
                (twice, rf"holds 1 tensors in more than one file \(the first: {embedding}\)")):
            target = self.work / (folder.name + " mlx")
            with self.subTest(folder.name), self.assertRaisesRegex(ValueError, pattern):
                self.run_convert(folder, target)
            self.assertFalse(target.exists() or target.with_name(target.name + ".partial").exists(), folder.name)

        reshaped = altered("a tensor of another shape")
        tensors = load_file(str(reshaped / first))
        tensors["language_model.layers.0.layer_scalar"] = torch.zeros(3, dtype=torch.bfloat16)
        save_file(tensors, str(reshaped / first))
        with self.assertRaisesRegex(ValueError, "is not what the loader expects: 1 of another shape"):
            self.run_convert(reshaped, self.work / "reshaped-mlx")
        self.assertTrue((self.work / "reshaped-mlx.partial").is_dir())
        self.assertFalse((self.work / "reshaped-mlx").exists())

    def test_the_command_line_exits_with_the_reason(self):
        """``main()`` ends with what ``convert`` raised as its message, which Python prints to stderr with exit
        status 1: a ``ValueError`` (the target exists) and a ``FileNotFoundError`` (no source files)."""
        nowhere = self.work / "not a folder"
        for source, target, message in (
                (self.work / "source", self.target, f"{self.target} already exists; nothing was written"),
                (nowhere, self.work / "from-nowhere",
                 f"{nowhere} has no {', '.join(('config.json', *self.convert.COPIED))}")):
            with self.subTest(message=message):
                with mock.patch.object(sys, "argv", ["convert_mlx.py", str(source), str(target)]), \
                     self.assertRaises(SystemExit) as raised:
                    self.convert.main()
                self.assertEqual(raised.exception.code, message)


if __name__ == "__main__":
    unittest.main()
