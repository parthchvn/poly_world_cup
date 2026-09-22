import unittest

from poly_world_cup.audit import audit_registry


class AuditTests(unittest.TestCase):
    def test_all_fixtures_does_not_mean_ready_for_training(self):
        registry = {
            "fixtures": [{"fixture_id": str(i), "mapping_status": "matched"} for i in range(104)],
            "contracts": [{"fixture_id": str(i), "condition_id": f"0x{i:064x}"} for i in range(104)],
        }
        result = audit_registry(registry)
        self.assertTrue(result["structural_audit_passed"])
        self.assertFalse(result["sft_ready"])
        self.assertIn("SOURCE_COVERAGE", {row["code"] for row in result["blockers"]})

    def test_missing_and_duplicate_fixtures_fail_audit(self):
        result = audit_registry({"fixtures": [{"fixture_id": "same"}, {"fixture_id": "same"}], "contracts": []})
        self.assertFalse(result["structural_audit_passed"])
        self.assertGreaterEqual(len(result["structural_errors"]), 2)

    def test_matched_fixture_without_contracts_is_inconsistent(self):
        result = audit_registry({"fixtures": [{"fixture_id": str(i), "mapping_status": "matched"} for i in range(104)], "contracts": []})
        self.assertFalse(result["structural_audit_passed"])
        self.assertTrue(any("no contracts" in error for error in result["structural_errors"]))

    def test_discovery_failure_cannot_be_hidden_by_summary_counts(self):
        registry = {
            "fixtures": [{"fixture_id": str(i), "mapping_status": "matched"} for i in range(104)],
            "contracts": [{"fixture_id": str(i), "condition_id": f"0x{i:064x}"} for i in range(104)],
            "report": {"coverage_complete": False, "failures": [{"code": "duplicate_token_id"}]},
        }
        self.assertFalse(audit_registry(registry)["structural_audit_passed"])

    def test_orphan_contract_fails_audit(self):
        result = audit_registry({"fixtures": [], "contracts": [{"condition_id": "c", "fixture_id": "unknown"}]})
        self.assertTrue(any("Orphan" in error for error in result["structural_errors"]))


if __name__ == "__main__":
    unittest.main()
