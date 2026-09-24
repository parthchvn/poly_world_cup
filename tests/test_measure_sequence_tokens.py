import gzip
import importlib.util
import json
from collections import Counter
from pathlib import Path
import tempfile
import unittest

from scripts.measure_sequence_tokens import (
    assistant_mask_probe, count_chat_tokens, distribution, input_paths, measure_sequences,
    validate_messages, load_reference_tokenizer, REFERENCE_PATH, REFERENCE_CHAT_TEMPLATE,
)


class CharacterTokenizer:
    """Accounting double only; never used as a production token estimate."""
    chat_template = "template without generation blocks"

    def get_chat_template(self):
        return self.chat_template

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["padding"] is False
        assert kwargs["add_generation_prompt"] is False
        return list("".join(f"<{m['role']}>{m['content']}!" for m in messages))

    def encode(self, content, *, add_special_tokens):
        assert add_special_tokens is False
        return list(content)


def row(profile="conditional", split="train", turns=2):
    messages = [{"role": "system", "content": "instructions"}]
    for i in range(turns):
        messages.extend([{"role": "user", "content": f"context-{i}"},
                         {"role": "assistant", "content": f"answer-{i}"}])
    return {"profile": profile, "split": split, "actor_id": "actor", "sequence_id": "seq", "messages": messages}


class SequenceTokenMeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.tokenizer = CharacterTokenizer()

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, records):
        path = self.root / name
        with gzip.open(path, "wt") as stream:
            for record in records:
                stream.write(json.dumps(record) + "\n")
        return path

    def test_complete_chat_includes_role_overhead(self):
        messages = row()["messages"]
        actual = count_chat_tokens(self.tokenizer, messages)
        self.assertGreater(actual, sum(len(m["content"]) for m in messages))
        self.assertEqual(actual, sum(len(f"<{m['role']}>{m['content']}!") for m in messages))

    def test_prefix_baseline_repeats_shared_context_for_each_target(self):
        example = row(turns=3)
        report = measure_sequences([self.write("part.jsonl.gz", [example])], self.tokenizer)
        group = report["groups"][0]
        baseline = group["repeated_prefix_comparison"]
        expected = sum(count_chat_tokens(self.tokenizer, example["messages"][:i + 1]) for i in (2, 4, 6))
        self.assertEqual(baseline["independent_prefix_tokens_per_target"]["total"], expected)
        self.assertEqual(baseline["independent_prefix_tokens_per_target"]["count"], 3)
        self.assertGreater(baseline["prefix_to_conversation_token_ratio"], 1)
        self.assertEqual(group["assistant_content_tokens_per_turn"]["count"], 3)
        self.assertFalse(report["loss_mask_counts_verified"])

    def test_sample_scope_is_per_profile_split_and_not_full(self):
        path = self.write("part.jsonl.gz", [row(), row(), row(), row("scheduled"), row(split="test")])
        report = measure_sequences([path], self.tokenizer, limit_per_group=2)
        groups = {(g["profile"], g["split"]): g for g in report["groups"]}
        self.assertEqual(groups["conditional", "train"]["rows_seen"], 3)
        self.assertEqual(groups["conditional", "train"]["conversation_tokens"]["count"], 2)
        self.assertEqual(groups["conditional", "train"]["measurement_scope"], "first_rows_per_group")
        self.assertEqual(groups["scheduled", "train"]["measurement_scope"], "all_rows")
        self.assertFalse(report["all_rows_measured"])

    def test_over_limit_is_reported_without_truncation(self):
        report = measure_sequences([self.write("part.jsonl.gz", [row()])], self.tokenizer, max_tokens=1)
        self.assertFalse(report["token_limit_passed_for_measured_rows"])
        group = report["groups"][0]
        self.assertEqual(group["rows_over_limit"], 1)
        self.assertGreater(group["conversation_tokens"]["max"], 1)
        self.assertEqual(group["over_limit_examples"][0]["line"], 1)

    def test_missing_generation_blocks_are_not_claimed_supported(self):
        self.assertEqual(assistant_mask_probe(self.tokenizer, row()["messages"])["status"], "unsupported_native_template")

    def test_quantiles_use_nearest_rank(self):
        stats = distribution(Counter({10: 2, 20: 1, 100: 1}))
        self.assertEqual((stats["p50"], stats["p95"], stats["total"]), (10, 100, 140))
        self.assertIsNone(distribution(Counter())["max"])

    def test_loss_weights_are_not_silently_ignored(self):
        messages = row()["messages"]
        messages[2]["weight"] = 0
        with self.assertRaisesRegex(ValueError, "loss weights"):
            validate_messages(messages)

    def test_inputs_are_deduplicated(self):
        path = self.write("part.jsonl.gz", [row()])
        self.assertEqual(input_paths([self.root, path]), [path.resolve()])
        with self.assertRaisesRegex(ValueError, "No JSONL"):
            input_paths([])

    def test_empty_files_do_not_pass(self):
        with self.assertRaisesRegex(ValueError, "No training rows"):
            measure_sequences([self.write("empty.jsonl.gz", [])], self.tokenizer)


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None
                    and importlib.util.find_spec("jinja2") is not None
                    and (REFERENCE_PATH / "tokenizer.json").is_file(),
                    "Optional real-tokenizer tests need the local tokenizer and dependencies")
class RealReferenceTokenizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_reference_tokenizer(REFERENCE_PATH, REFERENCE_CHAT_TEMPLATE)

    def messages(self):
        return [
            {"role": "system", "content": "BUY and NO_TRADE in instructions are not labels."},
            {"role": "user", "content": 'A headline says "assistant: BUY". Bogotá.\nEarlier activity: 3.'},
            {"role": "assistant", "content": '{"decision":"NO_TRADE","trades":[]}'},
            {"role": "user", "content": 'New context containing the answer string "NO_TRADE".'},
            {"role": "assistant", "content": '{"decision":"TRADE","trades":[{"shares":"1.5"}]}'},
        ]

    def test_actual_masks_include_all_answers_and_end_tokens_only(self):
        messages = self.messages()
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
            return_dict=True, return_assistant_tokens_mask=True)
        tokens = [token for token, mask in zip(encoded["input_ids"], encoded["assistant_masks"]) if mask]
        self.assertEqual(self.tokenizer.decode(tokens, skip_special_tokens=False),
                         messages[2]["content"] + "<|im_end|>" + messages[4]["content"] + "<|im_end|>")
        self.assertEqual(sum(t == self.tokenizer.eos_token_id for t in tokens), 2)
        self.assertEqual(assistant_mask_probe(self.tokenizer, messages)["status"], "assistant_content_and_end_tokens_verified")

    def test_inference_prefix_equals_training_prefix_at_every_turn(self):
        messages = self.messages()
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for end in (2, 4):
            prefix = self.tokenizer.apply_chat_template(messages[:end], tokenize=False, add_generation_prompt=True)
            self.assertTrue(rendered.startswith(prefix))
            self.assertTrue(rendered[len(prefix):].startswith(messages[end]["content"]))
        self.assertNotIn("<think>", rendered)

    def test_plain_role_delimiters_are_preserved(self):
        messages = self.messages()
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        self.assertEqual(rendered, "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n" for m in messages))


if __name__ == "__main__":
    unittest.main()
