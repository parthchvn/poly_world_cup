"""Bounded receipt checks; these never certify full trade-history coverage."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from .io import write_json
from .trades import ADDRESS, HEX_32

RPC_URL = "https://polygon.drpc.org"
CONTRACTS = {
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e": "v1",
    "0xc5d563a36ae78145c45a50134d48a1215220f80a": "v1",
    "0xe111180000d2663c0091e4f400237545b87b996b": "v2",
    "0xe2222d279d744050d28e00520010520000310f59": "v2",
}
TOPICS = {
    "v1": "0xd0a08e8c493f9c94f29311604c9de1b4e8c8d4c06bd0c789af57f2d65bfec0f6",
    "v2": "0xd543adfd945773f1a62f74f0ee55a5e3b9b1a28262980ba90b1a89f2ea84d8ee",
}
SOURCE_REVISIONS = {
    "v1": "https://github.com/Polymarket/ctf-exchange/tree/ed5c7708b7be3aa98bf5f0c6602b57cc498e2ef4",
    "v2": "https://github.com/Polymarket/ctf-exchange-v2/tree/ccc0596074f4dfd62c944fbca4de252893b82b4b",
}
SCALE = Decimal(1000000)


def _number(value) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("Boolean is not an amount")
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError("Invalid decimal amount") from error
    if not number.is_finite() or number < 0:
        raise ValueError("Amount must be finite and nonnegative")
    return number


def _hex_integer(value: str) -> int:
    if not isinstance(value, str) or not value.startswith("0x") or len(value) < 3:
        raise ValueError("Expected hexadecimal RPC quantity")
    return int(value, 16)


def _address_topic(value: str) -> str:
    if not HEX_32.fullmatch(value) or value[2:26] != "0" * 24:
        raise ValueError("Invalid ABI address topic")
    return "0x" + value[-40:].lower()


def decode_order_filled(log: dict) -> dict | None:
    """Decode known V1/V2 OrderFilled logs, preserving gross raw amounts.

    Non-OrderFilled events return None. Known event signatures at unknown
    contracts, removed logs, and malformed ABI layouts are rejected.
    """
    topics = log.get("topics", [])
    if not topics or topics[0].lower() not in TOPICS.values():
        return None
    address = str(log.get("address", "")).lower()
    version = CONTRACTS.get(address)
    if version is None:
        raise ValueError("OrderFilled signature at unsupported contract")
    if log.get("removed") is True:
        raise ValueError("Removed blockchain log")
    if len(topics) != 4 or topics[0].lower() != TOPICS[version]:
        raise ValueError("Contract and OrderFilled ABI do not agree")
    if not HEX_32.fullmatch(topics[1]):
        raise ValueError("Invalid order hash")
    data = log.get("data", "")
    words_expected = 5 if version == "v1" else 7
    if not isinstance(data, str) or not data.startswith("0x") or len(data) != 2 + 64 * words_expected:
        raise ValueError("Unexpected OrderFilled data length")
    words = [int(data[i:i + 64], 16) for i in range(2, len(data), 64)]
    if version == "v1":
        maker_asset, taker_asset, making, taking, fee = words
        if (maker_asset == 0) == (taker_asset == 0):
            raise ValueError("V1 fill must exchange collateral and one outcome token")
        side = "BUY" if maker_asset == 0 else "SELL"
        token_id = taker_asset if maker_asset == 0 else maker_asset
    else:
        side_number, token_id, making, taking, fee, _, _ = words
        if side_number not in (0, 1) or token_id == 0:
            raise ValueError("Invalid V2 side or token")
        side = "BUY" if side_number == 0 else "SELL"
    cash, tokens = (making, taking) if side == "BUY" else (taking, making)
    if tokens <= 0:
        raise ValueError("Cannot reconcile zero-token fill")
    transaction = str(log.get("transactionHash", "")).lower()
    if not HEX_32.fullmatch(transaction):
        raise ValueError("Invalid log transaction hash")
    index = _hex_integer(log.get("logIndex"))
    maker, taker = _address_topic(topics[2]), _address_topic(topics[3])
    with localcontext() as context:
        context.prec = 90
        price = format(Decimal(cash) / Decimal(tokens), "f")
        size = format(Decimal(tokens) / SCALE, "f")
    return {
        "canonical_log_id": f"137:{address}:{transaction}:{index}",
        "transaction_hash": transaction, "log_index": index,
        "contract_address": address, "exchange_version": version,
        "order_hash": topics[1].lower(), "maker_field": maker,
        "taker_field": taker, "maker_order_side": side,
        "token_id": str(token_id), "maker_amount_raw": str(making),
        "taker_amount_raw": str(taking), "fee_raw": str(fee),
        "gross_token_amount_raw": str(tokens), "gross_cash_amount_raw": str(cash),
        "gross_size": size, "gross_price": price,
        "emission_path": "exchange_aggregate" if taker == address else "counterparty_fill",
    }


def reconcile_observation(observation: dict, receipt: dict | None, *,
                          size_tolerance: str = "0.000001",
                          price_tolerance: str = "0.000001") -> dict:
    """Report compatible log candidates, without resolving ambiguous execution paths.

    Side inversion is considered only for the same outcome token. Complementary
    token mint/merge paths require their own aggregate order event; BUY NO is
    never silently changed into SELL YES. API amounts are compared with gross
    shares and, for a V1 BUY order owner, shares after the logged token fee.
    """
    transaction = str(observation.get("transaction_hash", "")).lower()
    wallet = str(observation.get("proxy_wallet", "")).lower()
    if not HEX_32.fullmatch(transaction) or not ADDRESS.fullmatch(wallet):
        raise ValueError("Invalid observation transaction or wallet")
    if observation.get("side") not in ("BUY", "SELL"):
        raise ValueError("Invalid observation side")
    size, price = _number(observation["size"]), _number(observation["price"])
    size_eps, price_eps = _number(size_tolerance), _number(price_tolerance)
    result = {"observation_id": observation.get("observation_id"),
              "transaction_hash": transaction, "proxy_wallet": wallet,
              "token_id": observation["token_id"], "side": observation["side"],
              "size": str(size), "price": str(price), "candidates": [],
              "decode_errors": [], "training_coverage_certified": False,
              "observation_identity_upgraded": False,
              "size_tolerance": str(size_eps), "price_tolerance": str(price_eps)}
    if receipt is None:
        return {**result, "status": "unknown_receipt"}
    if str(receipt.get("transactionHash", "")).lower() != transaction:
        return {**result, "status": "receipt_transaction_mismatch"}
    if receipt.get("status") != "0x1":
        return {**result, "status": "unsuccessful_or_unknown_transaction_status"}
    result["block_number"] = _hex_integer(receipt["blockNumber"])
    decoded, seen_ids = [], set()
    for index, log in enumerate(receipt.get("logs", [])):
        try:
            event = decode_order_filled(log)
            if event is None:
                continue
            if event["transaction_hash"] != transaction:
                raise ValueError("Log transaction does not match receipt")
            if event["canonical_log_id"] in seen_ids:
                raise ValueError("Duplicate canonical log in receipt")
            seen_ids.add(event["canonical_log_id"])
            decoded.append(event)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            result["decode_errors"].append({"receipt_log_position": index, "error": str(error)})
    result["decoded_orderfilled_log_count"] = len(decoded)
    for event in decoded:
        if event["token_id"] != str(observation["token_id"]):
            continue
        roles = []
        if wallet == event["maker_field"] and observation["side"] == event["maker_order_side"]:
            roles.append("active_order_owner_in_aggregate" if event["emission_path"] == "exchange_aggregate"
                         else "maker_field_order_owner")
        opposite = "SELL" if event["maker_order_side"] == "BUY" else "BUY"
        if (wallet == event["taker_field"] and observation["side"] == opposite
                and event["emission_path"] != "exchange_aggregate"):
            roles.append("same_token_opposite_side_counterparty")
        for role in roles:
            with localcontext() as context:
                context.prec = 90
                variants = {"gross": Decimal(event["gross_size"])}
                if (event["exchange_version"] == "v1" and event["maker_order_side"] == "BUY"
                        and wallet == event["maker_field"]):
                    net = (Decimal(event["gross_token_amount_raw"]) - Decimal(event["fee_raw"])) / SCALE
                    if net >= 0:
                        variants["v1_owner_after_logged_token_fee"] = net
                matched_amounts = [name for name, amount in variants.items() if abs(amount - size) <= size_eps]
                price_matches = abs(Decimal(event["gross_price"]) - price) <= price_eps
            result["candidates"].append({**event, "wallet_role_evidence": role,
                                          "amount_variants": {k: str(v) for k, v in variants.items()},
                                          "matching_amount_variants": matched_amounts,
                                          "price_matches_within_tolerance": price_matches,
                                          "amount_and_price_compatible": bool(matched_amounts) and price_matches})
    matches = [c for c in result["candidates"] if c["amount_and_price_compatible"]]
    result["compatible_log_candidate_count"] = len(matches)
    result["status"] = (
        "ambiguous_multiple_log_candidates" if len(matches) > 1 else
        "compatible_with_decode_gaps" if matches and result["decode_errors"] else
        "matched_single_log_candidate" if matches else
        "amount_or_price_mismatch" if result["candidates"] else
        "unsupported_or_invalid_logs" if result["decode_errors"] else
        "no_compatible_wallet_token_direction")
    return result


class ReceiptClient:
    """Read-only Polygon RPC client with checked local response provenance."""
    def __init__(self, cache_dir: Path, rpc_url: str = RPC_URL, timeout: float = 30):
        self.cache_dir, self.rpc_url, self.timeout = Path(cache_dir), rpc_url, timeout
        self.chain_checked = False

    def _rpc(self, method: str, params: list):
        payload = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
        request = Request(self.rpc_url, data=payload, headers={"Content-Type": "application/json",
                          "User-Agent": "Mozilla/5.0 (PolyWorldCupResearch/0.1)"})
        with urlopen(request, timeout=self.timeout) as response:
            raw = response.read()
        data = json.loads(raw)
        if data.get("error") is not None or "result" not in data:
            raise ValueError(f"RPC error: {data.get('error', 'missing result')}")
        return data["result"]

    def get_receipt(self, transaction_hash: str) -> dict | None:
        if not HEX_32.fullmatch(transaction_hash):
            raise ValueError("Invalid transaction hash")
        path = self.cache_dir / f"{transaction_hash.lower()}.json"
        if path.exists():
            captured = json.loads(path.read_text())
            body = json.dumps(captured["receipt"], sort_keys=True, separators=(",", ":")).encode()
            if (captured.get("chain_id") != 137 or captured.get("rpc_url") != self.rpc_url
                    or hashlib.sha256(body).hexdigest() != captured.get("receipt_sha256")):
                raise ValueError("Receipt cache integrity or provider mismatch")
            return captured["receipt"]
        if not self.chain_checked:
            if self._rpc("eth_chainId", []) != "0x89":
                raise ValueError("RPC endpoint is not Polygon chain 137")
            self.chain_checked = True
        receipt = self._rpc("eth_getTransactionReceipt", [transaction_hash])
        if receipt is not None:
            body = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
            write_json(path, {"rpc_url": self.rpc_url, "chain_id": 137,
                             "retrieved_at": datetime.now(timezone.utc).isoformat(),
                             "receipt_sha256": hashlib.sha256(body).hexdigest(), "receipt": receipt})
        return receipt


def select_observations(registry: dict, trades_root: Path, *, max_transactions: int = 104) -> list[dict]:
    """Choose one newest-page observation per fixture from committed pages.

    This is deterministic coverage sampling, not a random error-rate estimate.
    It does not validate the complete collection. Uncommitted pages are ignored.
    """
    if type(max_transactions) is not int or max_transactions < 1:
        raise ValueError("max_transactions must be positive")
    contracts = {}
    for contract in registry["contracts"]:
        contracts.setdefault(contract["fixture_id"], []).append(contract)
    selected, seen = [], set()
    for fixture in sorted(registry["fixtures"], key=lambda f: (f.get("kickoff_utc", ""), f["fixture_id"])):
        found = None
        for contract in sorted(contracts.get(fixture["fixture_id"], []), key=lambda c: c["condition_id"]):
            path = Path(trades_root) / contract["condition_id"] / "manifest.json"
            if not path.exists():
                continue
            manifest = json.loads(path.read_text())
            for page in manifest.get("pages", []):
                source = path.parent / page["file"]
                opener = gzip.open if source.suffix == ".gz" else open
                with opener(source, "rt", encoding="utf-8") as stream:
                    for line in stream:
                        row = json.loads(line)
                        if row["transaction_hash"] not in seen:
                            found = {"fixture_id": fixture["fixture_id"], "observation": row,
                                     "collection_page": str(source)}
                            break
                if found:
                    break
            if found:
                break
        if found:
            selected.append(found)
            seen.add(found["observation"]["transaction_hash"])
        if len(selected) >= max_transactions:
            break
    return selected


def run_sample(registry: dict, trades_root: Path, output_dir: Path, *,
               max_transactions: int = 104, rpc_url: str = RPC_URL,
               requests_per_second: float = 2.0, workers: int = 4, client=None) -> dict:
    """Persist bounded receipt-check results; RPC failure is an unknown result."""
    from .batch import RequestPacer

    pacer = RequestPacer(requests_per_second)
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer between 1 and 8")
    output_dir = Path(output_dir)
    client = client or ReceiptClient(output_dir / "receipts", rpc_url)
    selected = select_observations(registry, trades_root, max_transactions=max_transactions)
    previous_report = output_dir / "sample_report.json"
    archived_previous = None
    if previous_report.exists():
        previous = json.loads(previous_report.read_text())
        digest = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
        archived_previous = f"runs/{digest}.json"
        write_json(output_dir / archived_previous, previous)
    report = {"schema_version": 1, "scope": "bounded_receipt_sample_only",
              "selection": "one newest committed-page observation per fixture, deterministic contract order",
              "fixture_count": len(registry["fixtures"]), "selected_fixture_count": len(selected),
              "max_transactions": max_transactions, "rpc_url": rpc_url,
              "sample_collection_status": "running", "workers": workers,
              "previous_report_archive": archived_previous,
              "missing_fixture_ids": sorted({f["fixture_id"] for f in registry["fixtures"]}
                                             - {item["fixture_id"] for item in selected}),
              "source_revisions": SOURCE_REVISIONS, "training_coverage_certified": False,
              "full_history_reconciled": False, "results": []}
    def check(item):
        pacer.acquire()
        try:
            result = reconcile_observation(item["observation"], client.get_receipt(item["observation"]["transaction_hash"]))
        except Exception as error:
            result = {"transaction_hash": item["observation"]["transaction_hash"],
                      "observation_id": item["observation"].get("observation_id"),
                      "status": "unknown_rpc_or_input_error", "error": str(error)[:500]}
        return {"fixture_id": item["fixture_id"], **result}

    completed = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="receipt-sample") as pool:
        pending = {pool.submit(check, item): index for index, item in enumerate(selected)}
        for future in as_completed(pending):
            completed[pending[future]] = future.result()
            report["results"] = [completed[index] for index in sorted(completed)]
            report["status_counts"] = dict(Counter(row["status"] for row in report["results"]))
            report["updated_at"] = datetime.now(timezone.utc).isoformat()
            write_json(output_dir / "sample_report.json", report)
    report["status_counts"] = dict(Counter(row["status"] for row in report["results"]))
    report["sample_collection_status"] = "completed"
    report["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output_dir / "sample_report.json", report)
    return report
