import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from poly_world_cup.reconcile import (CONTRACTS, TOPICS, ReceiptClient,
                                     decode_order_filled, reconcile_observation, run_sample)

TX = "0x" + "a" * 64
WALLET = "0x" + "b" * 40
OTHER = "0x" + "c" * 40
V1, V2 = list(CONTRACTS)[0], list(CONTRACTS)[2]


def log(*, version="v2", maker=WALLET, taker=OTHER, side=0,
        making=4000000, taking=10000000, fee=0, index=1, token=123):
    contract = V1 if version == "v1" else V2
    words = ([0 if side == 0 else token, token if side == 0 else 0, making, taking, fee]
             if version == "v1" else [side, token, making, taking, fee, 0, 0])
    return {"address": contract, "transactionHash": TX, "logIndex": hex(index),
            "topics": [TOPICS[version], "0x" + "d" * 64,
                       "0x" + "0" * 24 + maker[2:], "0x" + "0" * 24 + taker[2:]],
            "data": "0x" + "".join(f"{word:064x}" for word in words), "removed": False}


def observation(**extra):
    return {"transaction_hash": TX, "proxy_wallet": WALLET, "token_id": "123",
            "side": "BUY", "size": "10", "price": "0.4", **extra}


def receipt(*logs):
    return {"transactionHash": TX, "blockNumber": "0x123", "status": "0x1", "logs": list(logs)}


class ReconcileTest(unittest.TestCase):
    def test_v2_gross_decimal_amounts_and_side_are_decoded_exactly(self):
        decoded = decode_order_filled(log(making=3333333))
        self.assertEqual(decoded["gross_price"], "0.3333333")
        self.assertEqual(decoded["maker_amount_raw"], "3333333")
        result = reconcile_observation(observation(price="0.3333333"), receipt(log(making=3333333)))
        self.assertEqual(result["status"], "matched_single_log_candidate")
        self.assertFalse(result["observation_identity_upgraded"])

    def test_passive_fill_and_active_aggregate_remain_ambiguous(self):
        passive = log(maker=OTHER, taker=WALLET, side=1, making=10000000, taking=4000000)
        aggregate = log(maker=WALLET, taker=V2, index=2)
        result = reconcile_observation(observation(), receipt(passive, aggregate))
        self.assertEqual(result["status"], "ambiguous_multiple_log_candidates")
        self.assertEqual(result["compatible_log_candidate_count"], 2)
        self.assertEqual({r["emission_path"] for r in result["candidates"]},
                         {"counterparty_fill", "exchange_aggregate"})
        self.assertFalse(result["training_coverage_certified"])

    def test_buy_no_is_not_silently_relabelled_sell_yes(self):
        result = reconcile_observation(observation(token_id="999", side="SELL"), receipt(log()))
        self.assertEqual(result["status"], "no_compatible_wallet_token_direction")

    def test_v1_owner_token_fee_is_explicit_variant(self):
        result = reconcile_observation(observation(size="9.9"), receipt(log(version="v1", fee=100000)))
        self.assertEqual(result["status"], "matched_single_log_candidate")
        self.assertEqual(result["candidates"][0]["matching_amount_variants"],
                         ["v1_owner_after_logged_token_fee"])

    def test_amount_discrepancy_is_not_hidden_by_wallet_match(self):
        result = reconcile_observation(observation(size="9.5"), receipt(log()))
        self.assertEqual(result["status"], "amount_or_price_mismatch")
        self.assertEqual(len(result["candidates"]), 1)

    def test_unknown_contract_removed_log_and_duplicate_fail_closed(self):
        for bad in ({**log(), "address": OTHER}, {**log(), "removed": True},
                    {**log(), "data": "0x01"}):
            with self.subTest(bad=bad):
                result = reconcile_observation(observation(), receipt(bad))
                self.assertEqual(result["status"], "unsupported_or_invalid_logs")
        result = reconcile_observation(observation(), receipt(log(), log()))
        self.assertEqual(result["status"], "compatible_with_decode_gaps")
        self.assertEqual(result["compatible_log_candidate_count"], 1)

    def test_missing_or_failed_receipt_never_proves_absence(self):
        self.assertEqual(reconcile_observation(observation(), None)["status"], "unknown_receipt")
        failed = {**receipt(log()), "status": "0x0"}
        self.assertEqual(reconcile_observation(observation(), failed)["status"],
                         "unsuccessful_or_unknown_transaction_status")

    def test_receipt_wrong_transaction_is_not_accepted(self):
        wrong = {**receipt(log()), "transactionHash": "0x" + "f" * 64}
        self.assertEqual(reconcile_observation(observation(), wrong)["status"], "receipt_transaction_mismatch")

    def test_receipt_cache_checks_provider_chain_and_checksum(self):
        with TemporaryDirectory() as root:
            client = ReceiptClient(Path(root))
            with patch.object(client, "_rpc", side_effect=["0x89", receipt(log())]) as rpc:
                first = client.get_receipt(TX)
                self.assertEqual(client.get_receipt(TX), first)
                self.assertEqual(rpc.call_count, 2)
            path = Path(root) / f"{TX}.json"
            captured = json.loads(path.read_text())
            captured["receipt"]["status"] = "0x0"
            path.write_text(json.dumps(captured))
            with self.assertRaisesRegex(ValueError, "integrity"):
                client.get_receipt(TX)

    def test_parallel_sample_preserves_failures_and_reports_coverage_scope(self):
        other_tx = "0x" + "e" * 64
        selected = [{"fixture_id": "f1", "observation": observation()},
                    {"fixture_id": "f2", "observation": observation(transaction_hash=other_tx)}]

        class Client:
            def get_receipt(self, transaction):
                if transaction == other_tx:
                    raise OSError("RPC unavailable")
                return receipt(log())

        with TemporaryDirectory() as root, patch("poly_world_cup.reconcile.select_observations", return_value=selected):
            result = run_sample({"fixtures": [{"fixture_id": "f1"}, {"fixture_id": "f2"}]}, Path(root), Path(root),
                                client=Client(), requests_per_second=10000)
            self.assertEqual([r["fixture_id"] for r in result["results"]], ["f1", "f2"])
            self.assertEqual(result["status_counts"], {"matched_single_log_candidate": 1,
                                                       "unknown_rpc_or_input_error": 1})
            self.assertEqual(result["sample_collection_status"], "completed")
            self.assertFalse(result["full_history_reconciled"])
            self.assertEqual(json.loads((Path(root) / "sample_report.json").read_text()), result)


if __name__ == "__main__":
    unittest.main()
