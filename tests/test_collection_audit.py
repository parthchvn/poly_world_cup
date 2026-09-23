"""An audit must verify saved observations before reporting collection progress."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from poly_world_cup.audit import audit_registry
from poly_world_cup.cli import main
from poly_world_cup.http import request_url
from poly_world_cup.trades import ingest_condition, validate_collection

CONDITION = "0x" + "a" * 64


def registry():
    return {
        "fixtures": [{"fixture_id": str(i), "mapping_status": "matched"} for i in range(104)],
        "contracts": [{"fixture_id": str(i), "condition_id": CONDITION if i == 0 else f"0x{i:064x}"} for i in range(104)],
    }


class Client:
    def get_json(self, url, params=None):
        row = {
            "condition_id": CONDITION, "proxy_wallet": "0x" + "b" * 40,
            "token_id": "123", "side": "BUY", "size": "2.1", "price": "0.4",
            "timestamp": 1780000000, "transaction_hash": "0x" + "c" * 64,
        }
        data = {"data": [row, row], "pagination": {"has_more": False, "next_cursor": None}}
        return SimpleNamespace(data=data, url=request_url(url, params),
                               body_sha256=hashlib.sha256(json.dumps(data).encode()).hexdigest(),
                               retrieved_at="2026-09-23T00:00:00Z")


class CollectionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = ingest_condition(Client(), condition_id=CONDITION, output_dir=self.root)
        self.manifest = self.root / CONDITION / "manifest.json"
        self.page = self.root / CONDITION / self.state["pages"][0]["file"]

    def audit(self):
        return audit_registry(registry(), self.root)

    def assert_invalid(self, report):
        self.assertFalse(report["structural_audit_passed"])
        self.assertFalse(report["sft_ready"])
        self.assertEqual(report["conditions_with_manifests"], 1)
        self.assertEqual(report["conditions_with_verified_manifests"], 0)
        self.assertEqual(report["conditions_with_invalid_manifests"], 1)
        self.assertEqual(report["observation_count"], 0)
        self.assertEqual(report["ingestion_manifests"], [])
        self.assertIn("COLLECTION_INTEGRITY", {b["code"] for b in report["blockers"]})

    def test_valid_audit_is_read_only_and_preserves_duplicate_observations(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = self.audit()
        self.assertTrue(report["structural_audit_passed"])
        self.assertEqual(report["conditions_with_verified_manifests"], 1)
        self.assertEqual(report["conditions_without_manifests"], 103)
        self.assertEqual(report["observation_count"], 2)
        self.assertFalse(report["sft_ready"])
        self.assertEqual(validate_collection(self.root, condition_id=CONDITION), self.state)
        after = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_missing_page_is_invalid_not_zero_activity(self):
        self.page.unlink()
        self.assert_invalid(self.audit())

    def test_corrupt_page_is_excluded_from_counts(self):
        self.page.write_text('{}\n')
        self.assert_invalid(self.audit())

    def test_false_manifest_totals_are_rejected(self):
        self.state["row_count"] = 999999
        self.manifest.write_text(json.dumps(self.state))
        self.assert_invalid(self.audit())

    def test_malformed_manifest_is_reported_instead_of_aborting_audit(self):
        self.manifest.write_text('{broken')
        self.assert_invalid(self.audit())

    def test_changed_filters_cannot_be_self_certified(self):
        # Modify both copies: checking them only against each other is insufficient.
        self.state["parameters"]["taker_only"] = "true"
        self.state["pages"][0]["parameters"]["taker_only"] = "true"
        self.manifest.write_text(json.dumps(self.state))
        self.assert_invalid(self.audit())

    def test_terminal_state_does_not_override_broken_cursor_chain(self):
        self.state["next_cursor"] = "unexpected"
        self.manifest.write_text(json.dumps(self.state))
        self.assert_invalid(self.audit())

    def test_duplicate_condition_case_cannot_double_count_observations(self):
        data = registry()
        data["contracts"].append({"fixture_id": "0", "condition_id": "0x" + "A" * 64})
        report = audit_registry(data, self.root)
        self.assertFalse(report["structural_audit_passed"])
        self.assertEqual(report["observation_count"], 2)
        self.assertEqual(report["conditions_with_verified_manifests"], 1)

    def test_invalid_condition_is_rejected_before_path_lookup(self):
        data = registry()
        data["contracts"][0]["condition_id"] = "../outside"
        report = audit_registry(data, self.root)
        self.assertFalse(report["structural_audit_passed"])
        self.assertEqual(report["observation_count"], 0)

    def test_cli_returns_failure_and_saves_diagnostic_for_corrupt_page(self):
        self.page.unlink()
        source = self.root / "registry.json"
        source.write_text(json.dumps(registry()))
        output = self.root / "audit.json"
        with contextlib.redirect_stdout(io.StringIO()):
            code = main(["audit", "--registry", str(source), "--trades-root", str(self.root), "--output", str(output)])
        self.assertEqual(code, 2)
        self.assert_invalid(json.loads(output.read_text()))


if __name__ == "__main__":
    unittest.main()
