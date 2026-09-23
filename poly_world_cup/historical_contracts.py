"""Reconstruct initial contract semantics from historical Polygon event evidence.

The provider is an explicit trust boundary: this validates decoded RPC evidence,
not a consensus proof. No current Gamma question text is promoted to a feature.
Initial metadata is distinct from later rule clarifications and outcomes.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any

CHAIN_ID = 137
ADAPTER = "0xd91e80cf2e7be2e162c6513ced06f1dd0da35296"
QUESTION_TOPIC = "0xaac410f87d423a922a7b226ac68f0c2eaf5bf6d15e644ac0758c7f96e2c253f7"
MARKET_TOPIC = "0xf059ab16d1ca60e123eab60e3c02b68faf060347c701a5d14885a8e1def7b3a8"
CONDITION_SELECTOR = "04329c03"
POSITION_SELECTOR = "752b5ba5"
SOURCE_COMMIT = "f78b35b0863b4308a431ca307d06f49b2ea65e78"
SOURCE_URL = f"https://github.com/Polymarket/neg-risk-ctf-adapter/blob/{SOURCE_COMMIT}/src/NegRiskAdapter.sol"
_HEX32 = re.compile(r"0x[0-9a-f]{64}\Z")


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def evidence_id(evidence: dict) -> str:
    return "polygon_initial_contract:" + hashlib.sha256(_json(evidence)).hexdigest()


def _hex(value: str, size: int | None = None) -> bytes:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError("Malformed hex evidence")
    try:
        result = bytes.fromhex(value[2:])
    except ValueError as exc:
        raise ValueError("Malformed hex evidence") from exc
    if size is not None and len(result) != size:
        raise ValueError("Unexpected hex evidence length")
    return result


def _uint(value: str) -> int:
    if not isinstance(value, str) or not re.fullmatch(r"0x[0-9a-fA-F]+", value):
        raise ValueError("Malformed RPC quantity")
    return int(value, 16)


def _utc(seconds: int) -> str:
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def decode_metadata_event(log: dict, block: dict, *, market: bool = False) -> dict:
    """Decode exactly the pinned adapter's (uint256,bytes) event payload."""
    if log.get("removed") is not False or log.get("address", "").lower() != ADAPTER:
        raise ValueError("Event is removed or belongs to another contract")
    topics = log.get("topics", [])
    if len(topics) != 3 or topics[0] != (MARKET_TOPIC if market else QUESTION_TOPIC):
        raise ValueError("Unexpected event signature")
    if any(not _HEX32.fullmatch(str(topic)) for topic in topics):
        raise ValueError("Malformed event topic")
    for key in ("blockHash", "transactionHash"):
        if not _HEX32.fullmatch(str(log.get(key, ""))):
            raise ValueError("Malformed event block/transaction hash")
    if log["blockHash"] != block.get("hash") or _uint(log["blockNumber"]) != _uint(block["number"]):
        raise ValueError("Event does not match its block header")
    timestamp = _uint(block["timestamp"])
    if "blockTimestamp" in log and _uint(log["blockTimestamp"]) != timestamp:
        raise ValueError("Event/block timestamp mismatch")
    raw = _hex(log["data"])
    if len(raw) < 96 or len(raw) % 32 or int.from_bytes(raw[32:64], "big") != 64:
        raise ValueError("Noncanonical event ABI")
    length = int.from_bytes(raw[64:96], "big")
    if len(raw) != 96 + ((length + 31) // 32) * 32 or any(raw[96 + length:]):
        raise ValueError("Truncated or noncanonical event bytes")
    metadata = raw[96:96 + length].decode("utf-8", errors="strict")
    first = int.from_bytes(raw[:32], "big")
    if not market:
        if first >= 256 or int(topics[2], 16) & 255 != first:
            raise ValueError("Question index mismatch")
        if topics[1] != topics[2][:-2] + "00":
            raise ValueError("Question market ID mismatch")
    return {"metadata": metadata, "timestamp": timestamp, "value": first,
            "market_id": topics[1], "question_id": None if market else topics[2]}


def _question_fields(text: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"question: (.+?), description: ([\s\S]+), id: ([0-9]+)", text)
    if not match:
        raise ValueError("Unsupported initial question metadata format")
    return match.groups()


def _market_fields(text: str) -> tuple[str, str, str]:
    match = re.fullmatch(r"title: (.+?), description: ([\s\S]+), id: ([0-9]+)", text)
    if not match:
        raise ValueError("Unsupported initial market metadata format")
    return match.groups()


def mapping_call_specs(question_id: str, block_number: str) -> list[dict]:
    if not _HEX32.fullmatch(question_id):
        raise ValueError("Invalid question ID")
    tail = question_id[2:]
    return [{"method": "eth_call", "params": [{"to": ADAPTER, "data": "0x" + data}, block_number]}
            for data in (CONDITION_SELECTOR + tail,
                         POSITION_SELECTOR + tail + (1).to_bytes(32, "big").hex(),
                         POSITION_SELECTOR + tail + bytes(32).hex())]


def _decode_evidence(evidence: dict) -> dict:
    if evidence.get("chain_id") != CHAIN_ID or evidence.get("adapter") != ADAPTER:
        raise ValueError("Unsupported chain or adapter")
    if evidence.get("source_code_url") != SOURCE_URL:
        raise ValueError("Unpinned contract source")
    if not str(evidence.get("rpc_url", "")).startswith("https://"):
        raise ValueError("Missing RPC provenance")
    question = decode_metadata_event(evidence["question_log"], evidence["question_block"])
    market = decode_metadata_event(evidence["market_log"], evidence["market_block"], market=True)
    if question["market_id"] != market["market_id"] or market["timestamp"] > question["timestamp"]:
        raise ValueError("Question is not in the earlier prepared market")
    calls = evidence["mapping_calls"]
    specs = mapping_call_specs(question["question_id"], evidence["question_log"]["blockNumber"])
    if len(calls) != len(specs) or any({k: call.get(k) for k in ("method", "params")} != spec
                                    for call, spec in zip(calls, specs)):
        raise ValueError("Mapping evidence must call the historical preparation block")
    results = [_hex(call["result"], 32) for call in calls]
    condition = "0x" + results[0].hex()
    yes, no = (str(int.from_bytes(value, "big")) for value in results[1:])
    if yes == no or yes == "0" or no == "0":
        raise ValueError("Invalid token mapping")
    question_text, initial_rules, gamma_market_id = _question_fields(question["metadata"])
    title, initial_description, gamma_event_id = _market_fields(market["metadata"])
    return {"condition_id": condition, "question_id": question["question_id"],
            "neg_risk_market_id": question["market_id"], "question": question_text,
            "fixture_title": title, "yes_token_id": yes, "no_token_id": no,
            "initialized_at_utc": _utc(max(question["timestamp"], market["timestamp"])),
            "fixture_initialized_at_utc": _utc(market["timestamp"]),
            "initial_rules": initial_rules, "initial_fixture_description": initial_description,
            "gamma_market_id": gamma_market_id, "gamma_event_id": gamma_event_id,
            "evidence_id": evidence_id(evidence),
            "semantics_scope": "initial_question_and_token_mapping"}


def build_contract_record(evidence: dict, *, fixture_id: str) -> dict:
    return {**_decode_evidence(evidence), "fixture_id": fixture_id, "evidence": evidence,
            "schema_version": 1, "historical_semantics_verified": True,
            "provider_trust_assumption": "RPC supplies canonical Polygon events, headers and historical call results; not a consensus proof"}


def validate_contract_record(record: dict) -> dict:
    """Re-decode raw event/header/call evidence; do not trust feature flags."""
    expected = _decode_evidence(record["evidence"])
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Contract evidence disagrees with derived field: {key}")
    if record.get("schema_version") != 1 or record.get("historical_semantics_verified") is not True:
        raise ValueError("Unsupported contract evidence record")
    if not re.fullmatch(r"espn:[0-9]+", str(record.get("fixture_id", ""))):
        raise ValueError("Missing audited fixture linkage")
    return record


def extract_initial_resolution_clause(initial_rules: str) -> str | None:
    """Return an exact, unambiguous captured regulation-duration sentence.

    Only the audited template is supported. Additional duration/extra-time/
    penalty statements make the source ambiguous for this small extractor,
    even when a human might establish consistency between them. Unsupported
    prose produces no feature; no fallback sentence is invented.
    """
    if not isinstance(initial_rules, str):
        return None
    scope_terms = re.compile(
        r"\b(?:90\s+minutes|regulation|regular\s+play|stoppage\s+time|"
        r"extra[\s-]+time|overtime|penalt(?:y|ies))\b", re.I)
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", initial_rules)
                 if sentence.strip()]
    candidates = [sentence for sentence in sentences if scope_terms.search(sentence)]
    if len(candidates) != 1 or len(candidates[0]) > 400:
        return None
    candidate = candidates[0]
    audited = r"This market refers only to the outcome within the first 90 minutes of regular play plus stoppage time\."
    return candidate if re.fullmatch(audited, candidate, re.I) else None


def verified_contract_context(record: dict, query_us: int | None) -> dict | None:
    """Return minimal initial semantics strictly before the query timestamp.

    Call validate_contract_record once on catalog loading. This cheap function
    is safe to call for each observation after that independent validation.
    """
    if query_us is None or record.get("historical_semantics_verified") is not True:
        return None
    initialized = datetime.fromisoformat(record["initialized_at_utc"].replace("Z", "+00:00"))
    if int(initialized.timestamp()) * 1_000_000 >= query_us:
        return None
    context = {key: record[key] for key in ("question", "fixture_title", "yes_token_id", "no_token_id",
               "initialized_at_utc", "semantics_scope", "evidence_id")}
    clause = extract_initial_resolution_clause(record["initial_rules"])
    if clause is not None:
        context["initial_resolution_clause"] = clause
    return context
