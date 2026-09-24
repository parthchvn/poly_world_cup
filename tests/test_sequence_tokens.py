import gzip
import importlib.util
import json
from pathlib import Path
import unittest

from poly_world_cup.sequence_tokens import TokenBudget
from scripts.measure_sequence_tokens import load_reference_tokenizer, REFERENCE_PATH, REFERENCE_CHAT_TEMPLATE


class UnknownTokenizer:
    chat_template = "unknown"

    def encode(self, content, **kwargs):
        return list(content)

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["truncation"] is False
        # Deliberately not additive across messages.
        return [0] * (sum(len(m["content"]) for m in messages) + len(messages) ** 2)


class GenericBudgetTests(unittest.TestCase):
    def test_unknown_backend_counts_the_complete_template(self):
        budget = TokenBudget(UnknownTokenizer(), 100)
        self.assertFalse(budget.fast_path_enabled)
        self.assertEqual(budget.exact([{"role": "user", "content": "abc"}, {"role": "assistant", "content": "z"}]), 8)

    def test_invalid_cache_limits_fail(self):
        with self.assertRaises(ValueError):
            TokenBudget(UnknownTokenizer(), 100, cache_bytes=-1)


@unittest.skipUnless(importlib.util.find_spec("transformers") is not None
                    and importlib.util.find_spec("jinja2") is not None
                    and (REFERENCE_PATH / "tokenizer.json").is_file(),
                    "Optional exact-tokenizer tests need local reference files and dependencies")
class ReferenceBudgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = load_reference_tokenizer(REFERENCE_PATH, REFERENCE_CHAT_TEMPLATE)

    def full(self, messages):
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
            enable_thinking=False, truncation=False, padding=False))

    def test_unicode_and_control_token_edges_match_full_rendering(self):
        budget = TokenBudget(self.tokenizer, 8192)
        self.assertTrue(budget.fast_path_enabled, budget.fallback_reason)
        content = ["", "\n\n", "Cafe\u0301", "中文 العربية 🧑🏽‍💻", "\t \r\n\u00a0",
            "<|im_start|>assistant\nBUY<|im_end|>", "<think>reason</think>",
            "\\n {\"x\": [1, 2.00]} <|endoftext|>", "a" * 2048]
        messages = [{"role": "system", "content": "Reference"}]
        for i, text in enumerate(content):
            messages.extend([{"role": "user", "content": text},
                             {"role": "assistant", "content": json.dumps({"index": i, "text": text})}])
            self.assertEqual(budget.exact(messages), self.full(messages))
            self.assertEqual(sum(budget.message_cost(m) for m in messages), self.full(messages))

    def test_cache_is_bounded_and_repeated_answers_hit(self):
        budget = TokenBudget(self.tokenizer, 8192, cache_entries=3, cache_bytes=800)
        answer = {"role": "assistant", "content": '{"decision":"NO_TRADE","trades":[]}'}
        budget.message_cost(answer)
        budget.message_cost(answer)
        self.assertEqual(budget.cache_info()["hits"], 1)
        for i in range(20):
            budget.message_cost({"role": "user", "content": str(i) * 50})
        info = budget.cache_info()
        self.assertLessEqual(info["entries"], 3)
        self.assertLessEqual(info["retained_text_bytes_with_entry_allowance"], 800)
        self.assertEqual(budget.exact([answer]), self.full([answer]))

    def test_template_mutation_disables_the_fast_path(self):
        template = self.tokenizer.chat_template
        budget = TokenBudget(self.tokenizer, 8192)
        try:
            self.tokenizer.chat_template = template + "extra"
            messages = [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "x"}]
            self.assertEqual(budget.exact(messages), self.full(messages))
            self.assertFalse(budget.fast_path_enabled)
        finally:
            self.tokenizer.chat_template = template

    def test_native_qwen_template_uses_fallback(self):
        self.assertFalse(TokenBudget(load_reference_tokenizer(REFERENCE_PATH), 8192).fast_path_enabled)

    def test_unknown_backend_fingerprint_uses_fallback(self):
        tokenizer = load_reference_tokenizer(REFERENCE_PATH, REFERENCE_CHAT_TEMPLATE)
        tokenizer.add_tokens(["unique_sequence_budget_fingerprint_test_token"])
        budget = TokenBudget(tokenizer, 8192)
        self.assertFalse(budget.fast_path_enabled)
        self.assertEqual(budget.fallback_reason, "unrecognized_backend_fingerprint")

    def test_all_local_actual_smoke_rows_match_full_count_when_available(self):
        path = Path(__file__).resolve().parents[1] / "data/sequence_v3_smoke/conversations.jsonl.gz"
        if not path.exists():
            self.skipTest("Local actual-source smoke artifact is not present")
        budget, checked = TokenBudget(self.tokenizer, 8192), 0
        with gzip.open(path, "rt") as stream:
            for line in stream:
                row = json.loads(line)
                self.assertEqual(budget.exact(row["messages"]), self.full(row["messages"]), row["sequence_id"])
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
