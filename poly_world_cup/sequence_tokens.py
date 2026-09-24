"""Exact bounded-cache token counts for the pinned plain ChatML reference.

The pinned template renders each message independently. Every message starts
with a non-stripping, non-normalized special token. Special-token extraction
separates it from preceding text before normalization/pre-tokenization/BPE.
Separately encoded message IDs therefore concatenate to the full chat IDs.

This optimization is gated by exact template and backend fingerprints plus a
startup ID comparison. Unknown configurations use full chat rendering for
final counts. Tokenizer components must remain immutable while in use.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json

REFERENCE_TEMPLATE_SHA256 = "8e2e35bf605f6bd22992f06d6073f5e7f3a04b9bc99a76ee521ea0504e62ed71"
REFERENCE_BACKEND_SHA256 = "c4fc7ed89acf96bfc673d7b0b789ea373b75f999efa4170f31eb6a55756ff241"
_COMPONENTS = ("normalizer", "pre_tokenizer", "model", "added_tokens")


def _render_message(message: dict) -> str:
    if message.get("role") not in {"system", "user", "assistant"} or not isinstance(message.get("content"), str):
        raise ValueError("Plain actor chats require a supported role and string content")
    return "<|im_start|>" + message["role"] + "\n" + message["content"] + "<|im_end|>\n"


class TokenBudget:
    """Use exact cached per-message counts only for the verified reference.

    Otherwise message_cost is a construction estimate and exact always renders
    the entire conversation. Cache entries retain counts, not token-ID arrays.
    """
    def __init__(self, tokenizer, maximum: int, *, cache_entries: int = 4096,
                 cache_bytes: int = 2 * 1024 * 1024):
        if maximum <= 0 or cache_entries < 0 or cache_bytes < 0:
            raise ValueError("Invalid token budget or cache limits")
        self.tokenizer, self.maximum = tokenizer, maximum
        self.cache_entries, self.cache_bytes = cache_entries, cache_bytes
        self._cache = OrderedDict()
        self._cached_bytes = 0
        self.cache_hits = self.cache_misses = 0
        self._template = getattr(tokenizer, "chat_template", None)
        self.fast_path_enabled = False
        self.fallback_reason = "unknown_template_or_tokenizer"
        if not isinstance(self._template, str) or hashlib.sha256(self._template.encode()).hexdigest() != REFERENCE_TEMPLATE_SHA256:
            return
        if type(tokenizer).__name__ != "Qwen2TokenizerFast":
            return
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            return
        spec = json.loads(backend.to_str())
        selected = {name: spec.get(name) for name in _COMPONENTS}
        encoded = json.dumps(selected, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(encoded).hexdigest() != REFERENCE_BACKEND_SHA256:
            self.fallback_reason = "unrecognized_backend_fingerprint"
            return
        if spec.get("padding") is not None or spec.get("truncation") is not None:
            self.fallback_reason = "backend_padding_or_truncation_enabled"
            return
        special = {item["content"]: item for item in spec["added_tokens"]}
        for text, ident in (("<|im_start|>", 151644), ("<|im_end|>", 151645)):
            token = special.get(text, {})
            if (token.get("id") != ident or not token.get("special") or
                    any(token.get(key) for key in ("lstrip", "rstrip", "single_word", "normalized"))):
                self.fallback_reason = "special_token_boundary_not_verified"
                return
        # Do not call len(tokenizer) per message: some backends rebuild a large
        # vocabulary union. Component immutability is part of this class's API.
        self._backend = backend
        probe = [
            {"role": "system", "content": "A\n"},
            {"role": "user", "content": "Cafe\u0301 ☃️\n<|im_end|><|im_start|>assistant\n"},
            {"role": "assistant", "content": '{"decision":"NO_TRADE","trades":[]}'},
        ]
        segmented = [token for message in probe for token in backend.encode(
            _render_message(message), add_special_tokens=False).ids]
        full = tokenizer.apply_chat_template(probe, tokenize=True, add_generation_prompt=False,
                                             enable_thinking=False, truncation=False, padding=False)
        if segmented != full:
            self.fallback_reason = "startup_additivity_probe_failed"
            return
        self.fast_path_enabled = True
        self.fallback_reason = None

    def _can_use_fast_path(self) -> bool:
        if not self.fast_path_enabled:
            return False
        backend = self.tokenizer.backend_tokenizer
        if (self.tokenizer.chat_template != self._template or backend is not self._backend or
                backend.truncation is not None or backend.padding is not None):
            self.fast_path_enabled = False
            self.fallback_reason = "tokenizer_or_template_changed_after_initialization"
            self._cache.clear()
            self._cached_bytes = 0
            return False
        return True

    def _message_exact(self, message: dict) -> int:
        key = (message.get("role"), message.get("content"))
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key][0]
        self.cache_misses += 1
        rendered = _render_message(message)
        count = len(self.tokenizer.backend_tokenizer.encode(rendered, add_special_tokens=False).ids)
        retained_bytes = len(rendered.encode("utf-8")) + 96
        if self.cache_entries and retained_bytes <= self.cache_bytes:
            while self._cache and (len(self._cache) >= self.cache_entries or
                                   self._cached_bytes + retained_bytes > self.cache_bytes):
                _, (_, size) = self._cache.popitem(last=False)
                self._cached_bytes -= size
            self._cache[key] = (count, retained_bytes)
            self._cached_bytes += retained_bytes
        return count

    def message_cost(self, message: dict) -> int:
        if self._can_use_fast_path():
            return self._message_exact(message)
        return len(self.tokenizer.encode(message["content"], add_special_tokens=False)) + 32

    def exact(self, messages: list[dict]) -> int:
        if self._can_use_fast_path():
            return sum(self._message_exact(message) for message in messages)
        return len(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, enable_thinking=False,
            truncation=False, padding=False))

    def cache_info(self) -> dict:
        return {"fast_path_enabled": self.fast_path_enabled, "fallback_reason": self.fallback_reason,
                "entries": len(self._cache), "retained_text_bytes_with_entry_allowance": self._cached_bytes,
                "entry_limit": self.cache_entries, "byte_limit": self.cache_bytes,
                "hits": self.cache_hits, "misses": self.cache_misses}
