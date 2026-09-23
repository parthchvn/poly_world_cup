"""Structural and readiness audit; API traversal is not completeness evidence."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .trades import HEX_32, TradeIngestionError, validate_collection


def audit_registry(registry: dict, trades_root: Path | None = None) -> dict:
    fixtures = registry.get("fixtures", [])
    contracts = registry.get("contracts", [])
    errors: list[str] = []
    fixture_ids = [row.get("fixture_id") for row in fixtures]
    if len(fixtures) != 104:
        errors.append(f"Expected 104 fixtures for the 2026 tournament; found {len(fixtures)}")
    if any(not value for value in fixture_ids) or len(set(fixture_ids)) != len(fixture_ids):
        errors.append("Fixture IDs must be present and unique")
    condition_ids = [row.get("condition_id") for row in contracts]
    valid_condition_ids = [value.lower() for value in condition_ids
                           if isinstance(value, str) and HEX_32.fullmatch(value)]
    if len(valid_condition_ids) != len(condition_ids):
        errors.append("Condition IDs must be 32-byte 0x-prefixed hex strings")
    if len(set(valid_condition_ids)) != len(valid_condition_ids):
        errors.append("Condition IDs must be unique after case normalization")
    for row in contracts:
        if row.get("fixture_id") not in fixture_ids:
            errors.append(f"Orphan contract: {row.get('condition_id')}")
    mapped_fixtures = {row.get("fixture_id") for row in contracts}
    for row in fixtures:
        if row.get("mapping_status") != "matched":
            errors.append(f"Unresolved fixture mapping: {row.get('fixture_id')}")
        elif row.get("fixture_id") not in mapped_fixtures:
            errors.append(f"Matched fixture has no contracts: {row.get('fixture_id')}")
    source_report = registry.get("report", {})
    if source_report.get("failures") or source_report.get("coverage_complete") is False:
        errors.append("Discovery report contains unresolved coverage failures")

    manifests = []
    missing = []
    invalid = []
    for condition_id in sorted(set(valid_condition_ids)):
        path = Path(trades_root) / condition_id / "manifest.json" if trades_root else None
        if path is None or not path.exists():
            missing.append(condition_id)
            continue
        try:
            manifest = validate_collection(Path(trades_root), condition_id=condition_id)
        except TradeIngestionError as exc:
            invalid.append({"condition_id": condition_id, "reason": str(exc)})
            errors.append(f"Invalid collection for {condition_id}: {exc}")
            continue
        manifests.append(manifest)

    # These require independent evidence/implementation. Changing a manifest's
    # boolean cannot turn this initial collector into a certified training set.
    blockers = [
        {"code": "EXECUTION_IDENTITY", "detail": "Reconcile wallet-side observations to canonical executions and verified maker/taker roles; API rows lack log/order IDs."},
        {"code": "HISTORICAL_AVAILABILITY", "detail": "Establish historical availability and versions of news, market state, fixture metadata, and prior activity."},
        {"code": "SOURCE_COVERAGE", "detail": "Independently establish label coverage; API exhaustion and filtered observations do not prove full trade history."},
        {"code": "HISTORICAL_UNIVERSE", "detail": "Construct checkpoint cohorts from global past activity or explicitly restrict to already observed tournament participants."},
        {"code": "INVENTORY", "detail": "Reconcile initial balances, transfers, splits, merges, redemptions, and fills; retain unknowns until established."},
        {"code": "SPLIT_AND_BASELINES", "detail": "Freeze fixture-disjoint temporal splits, build baselines, and run context ablations before SFT."},
    ]
    if missing:
        blockers.insert(0, {"code": "UNINGESTED_CONDITIONS", "detail": f"No local ingestion manifest for {len(missing)} of {len(condition_ids)} conditions."})
    if invalid:
        blockers.insert(0, {"code": "COLLECTION_INTEGRITY", "detail": f"{len(invalid)} collections failed integrity checks and are excluded from observation totals."})
    mapping_counts = dict(sorted(Counter(row.get("mapping_status", "unknown") for row in fixtures).items()))
    if any(status != "matched" for status in mapping_counts):
        blockers.insert(0, {"code": "FIXTURE_MAPPING", "detail": f"Review fixture mapping statuses: {mapping_counts}"})
    return {
        "scope": "2026 FIFA World Cup, all 104 fixtures, match-result contracts",
        "fixture_count": len(fixtures), "contract_count": len(contracts),
        "fixture_mapping_status_counts": mapping_counts,
        "discovery_warnings": source_report.get("warnings", []),
        "discovery_failures": source_report.get("failures", []),
        "conditions_with_manifests": len(manifests) + len(invalid),
        "conditions_with_verified_manifests": len(manifests),
        "conditions_with_invalid_manifests": len(invalid),
        "invalid_collections": invalid,
        "conditions_with_observations": sum(manifest.get("row_count", 0) > 0 for manifest in manifests),
        "observation_count": sum(manifest.get("row_count", 0) for manifest in manifests),
        "manifest_status_counts": dict(sorted(Counter(manifest.get("api_traversal_status", "unknown") for manifest in manifests).items())),
        "zero_page_manifest_count": sum(manifest.get("page_count", 0) == 0 for manifest in manifests),
        "conditions_without_manifests": len(missing),
        "missing_condition_ids": missing,
        "ingestion_manifests": manifests,
        "structural_errors": errors,
        "structural_audit_passed": not errors,
        "sft_ready": False,
        "blockers": blockers,
        "note": "This is a collection-foundation audit, not a scientific validity certificate. No training examples are exported.",
        "integrity_scope": "Observation totals include only committed normalized pages whose hashes, counts, timestamps, and cursor chains verify. Raw HTTP bodies and source completeness are not certified.",
    }
