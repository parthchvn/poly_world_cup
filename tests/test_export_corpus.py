import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index
from scripts.export_corpus import export_corpus

CONDITION = "0x" + "1" * 64
REGISTRY = {"fixtures": [{"fixture_id": "fixture:1"}], "contracts": [{
    "condition_id": CONDITION, "fixture_id": "fixture:1", "selection": "mexico",
    "tokens": [{"token_id": "123", "outcome": "Yes"}, {"token_id": "456", "outcome": "No"}]}]}


class ExportCorpusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.trades = self.root / "trades"
        self.page = self.trades / CONDITION / "pages" / "page.jsonl.gz"
        self.page.parent.mkdir(parents=True)
        self.raw = {"data": []}
        observations = []
        for number in range(3):
            observations.append({"observation_id": f"observation:{number}", "condition_id": CONDITION,
                "token_id": "123", "side": "BUY", "size": "1.000001", "price": "0.12345678",
                "proxy_wallet": "wallet:a", "block_timestamp": f"2026-06-11T12:00:0{number}Z",
                "transaction_hash": f"tx:{number}"})
        data = "".join(json.dumps(row) + "\n" for row in observations).encode()
        self.page.write_bytes(gzip.compress(data))
        raw_data = json.dumps(self.raw).encode()
        raw_hash = hashlib.sha256(raw_data).hexdigest()
        body = self.root / "cache" / CONDITION / "bodies" / (raw_hash + ".json.gz")
        body.parent.mkdir(parents=True)
        body.write_bytes(gzip.compress(raw_data))
        # An unrelated cached article must never appear in a raw trade release.
        (body.parent / ("b" * 64 + ".json")).write_text('{"story":"copyrighted article"}')
        self.manifest_path = self.trades / CONDITION / "manifest.json"
        self.manifest = {"condition_id": CONDITION, "api_traversal_status": "exhausted", "row_count": 3,
            "page_count": 1, "pages": [{"file": "pages/page.jsonl.gz", "row_count": 3,
                "normalized_sha256": hashlib.sha256(data).hexdigest(), "body_sha256": raw_hash,
                "request_url": "https://data-api.polymarket.com/v2/trades?condition=" + CONDITION}]}
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.registry = self.root / "registry.json"
        self.registry.write_text(json.dumps(REGISTRY))
        self.news = self.root / "news.jsonl"
        self.record = {"news_id": "n:1", "title": "Mexico squad announced", "fixture_ids": ["fixture:1"],
                       "published_at_utc": "2026-06-10T00:00:00Z", "historical_availability_verified": False,
                       "body": None, "description": None}
        self.news.write_text(json.dumps(self.record) + "\n")
        self.database = self.root / "corpus.sqlite"
        build_attribution_index(registry=REGISTRY, trade_pages=[self.page], news_records=[self.record], output_path=self.database)

    def tearDown(self):
        self.tmp.cleanup()

    def export(self, **kwargs):
        args = dict(database=self.database, registry=self.registry, news=self.news, archived_news=[],
                    trades_root=self.trades, output_dir=self.root / "release", sample_fixtures=1)
        args.update(kwargs)
        return export_corpus(**args)

    def test_portable_corpus_retains_every_trade_and_manifest_hashes(self):
        result = self.export()
        artifact = self.root / "release" / result["artifacts"][0]["file"]
        self.assertEqual(result["counts"]["trade_observations"], 3)
        unpacked = self.root / "unpacked"
        with tarfile.open(artifact) as archive:
            archive.extractall(unpacked, filter="data")
        with sqlite3.connect(unpacked / "attribution.sqlite") as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM trades").fetchone()[0], 3)
            self.assertEqual(db.execute("SELECT shares,price FROM trades LIMIT 1").fetchone(), ("1.000001", "0.12345678"))
        manifest = json.loads((unpacked / "MANIFEST.json").read_text())
        for item in manifest["members"]:
            self.assertEqual(hashlib.sha256((unpacked / item["path"]).read_bytes()).hexdigest(), item["sha256"])
        self.assertEqual(len((unpacked / "context_samples.jsonl").read_text().splitlines()), 3)
        self.assertFalse(result["training_ready"])

    def test_raw_export_is_separate_and_only_includes_referenced_trade_bodies(self):
        result = self.export(include_raw_trades=True, trade_cache=self.root / "cache")
        self.assertEqual(len(result["artifacts"]), 2)
        with tarfile.open(self.root / "release" / "trade_provenance.tar.gz") as archive:
            names = archive.getnames()
        self.assertTrue(any(name.startswith("cache/") and "/bodies/" in name for name in names))
        self.assertFalse(any("b" * 64 in name or "html" in name or "news" in name for name in names))
        with tarfile.open(self.root / "release" / "world_cup_corpus.tar.gz") as archive:
            self.assertFalse(any(name.startswith("cache/") for name in archive.getnames()))

    def test_incomplete_collection_is_rejected(self):
        self.manifest["api_traversal_status"] = "partial_page_limit"
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "exhausted traversal"):
            self.export()

    def test_database_missing_trade_rows_is_rejected(self):
        with sqlite3.connect(self.database) as db:
            db.execute("DELETE FROM trades WHERE trade_row_id=1")
        with self.assertRaisesRegex(ValueError, "every manifested trade"):
            self.export()

    def test_news_bodies_are_rejected_even_inside_sqlite(self):
        with sqlite3.connect(self.database) as db:
            row = {**self.record, "body": "A full article body"}
            db.execute("UPDATE news SET record_json=?", (json.dumps(row),))
        with self.assertRaisesRegex(ValueError, "body field"):
            self.export()

    def test_mismatched_news_catalog_is_rejected(self):
        self.news.write_text(json.dumps({**self.record, "title": "A later changed headline"}) + "\n")
        with self.assertRaisesRegex(ValueError, "news content differs"):
            self.export()

    def test_duplicate_manifest_page_is_rejected(self):
        self.manifest["pages"] *= 2
        self.manifest["page_count"] = 2
        self.manifest["row_count"] = 6
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "repeats a normalized page"):
            self.export()

    def test_unsafe_page_path_is_rejected(self):
        self.manifest["pages"][0]["file"] = "../../outside.jsonl"
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "Unsafe manifest"):
            self.export()

    def test_immutable_export_avoids_backup_and_keeps_original_database(self):
        before = self.database.read_bytes()
        result = self.export(immutable_database=True)
        self.assertEqual(self.database.read_bytes(), before)
        self.assertTrue(result["disk_estimate"]["immutable_database_requested"])
        self.assertTrue((self.root / "release" / "world_cup_corpus.tar.gz").exists())

    def test_concurrent_immutable_database_change_discards_archive(self):
        import os
        from unittest.mock import patch
        from scripts import export_corpus as exporter
        original = exporter._archive
        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            stat = self.database.stat()
            os.utime(self.database, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000000))
            return result
        with patch.object(exporter, "_archive", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "changed during export"):
                self.export(immutable_database=True)
        self.assertFalse((self.root / "release" / "world_cup_corpus.tar.gz").exists())

    def test_invalid_raw_provenance_report_is_rejected(self):
        path = self.root / "provenance.json"
        path.write_text(json.dumps({"raw_provenance_verified": False}))
        with self.assertRaisesRegex(ValueError, "did not verify"):
            self.export(provenance_report=path)

    def test_verified_raw_provenance_counts_must_match(self):
        path = self.root / "provenance.json"
        path.write_text(json.dumps({"raw_provenance_verified": True, "verified_observation_count": 4,
                                   "verified_condition_count": 1, "verified_raw_page_count": 1}))
        with self.assertRaisesRegex(ValueError, "counts do not match"):
            self.export(provenance_report=path)

    def checkpoint_inputs(self):
        fixtures = [{"fixture_id": f"fixture:{i}"} for i in range(1, 105)]
        contracts = []
        for index in range(312):
            condition = CONDITION if index == 0 else "0x" + f"{index:064x}"
            contracts.append({"condition_id": condition, "fixture_id": f"fixture:{index // 3 + 1}",
                              "selection": "test", "tokens": [{"token_id": str(10000 + index * 2), "outcome": "Yes"},
                                                                   {"token_id": str(10001 + index * 2), "outcome": "No"}]})
            if index == 0:
                manifest = {**self.manifest, "api_traversal_status": "paused"}
            else:
                manifest = {"condition_id": condition, "api_traversal_status": "paused" if index < 19 else "exhausted",
                            "row_count": 0, "page_count": 0, "pages": []}
            path = self.trades / condition / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest))
        self.registry.write_text(json.dumps({"fixtures": fixtures, "contracts": contracts}))
        provenance = self.root / "checkpoint_provenance.json"
        provenance.write_text(json.dumps({"raw_provenance_verified": True, "condition_selection": "explicit", "verified_observation_count": 3,
            "verified_condition_count": 312, "verified_raw_page_count": 1, "api_exhausted_conditions": 293,
            "requested_condition_count": 312, "normalized_integrity_verified_conditions": 312, "errors": []}))
        return provenance

    def test_paused_checkpoints_still_fail_without_explicit_partial_flag(self):
        provenance = self.checkpoint_inputs()
        with self.assertRaisesRegex(ValueError, "exhausted traversal"):
            self.export(provenance_report=provenance)

    def test_explicit_partial_export_labels_archives_and_both_manifests(self):
        provenance = self.checkpoint_inputs()
        result = self.export(allow_partial=True, provenance_report=provenance,
                             include_raw_trades=True, trade_cache=self.root / "cache")
        self.assertTrue(result["partial"])
        self.assertFalse(result["training_ready"])
        self.assertEqual(result["manifest_status_counts"], {"exhausted": 293, "paused": 19})
        self.assertEqual([row["file"] for row in result["artifacts"]],
                         ["world_cup_partial_corpus.tar.gz", "trade_partial_provenance.tar.gz"])
        for artifact in result["artifacts"]:
            with tarfile.open(self.root / "release" / artifact["file"]) as archive:
                manifest = json.load(archive.extractfile("MANIFEST.json"))
                self.assertTrue(manifest["partial"])
                self.assertEqual(manifest["manifest_status_counts"], {"exhausted": 293, "paused": 19})
                readme = archive.extractfile("README.md").read().decode()
                self.assertIn("INCOMPLETE CHECKPOINT", readme)
                self.assertIn("293 exhausted", readme)
                self.assertIn("19 paused", readme)
        self.assertFalse((self.root / "release" / "world_cup_corpus.tar.gz").exists())

    def test_partial_export_requires_every_registry_manifest(self):
        provenance = self.checkpoint_inputs()
        (self.trades / ("0x" + f"{311:064x}") / "manifest.json").unlink()
        with self.assertRaisesRegex(ValueError, "regular input file"):
            self.export(allow_partial=True, provenance_report=provenance)

    def test_partial_export_does_not_accept_failed_or_arbitrary_status(self):
        provenance = self.checkpoint_inputs()
        self.manifest["api_traversal_status"] = "failed"
        self.manifest_path.write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, "only paused or exhausted"):
            self.export(allow_partial=True, provenance_report=provenance)

    def test_partial_export_requires_full_312_contract_universe(self):
        with self.assertRaisesRegex(ValueError, "104 fixtures and 312"):
            self.export(allow_partial=True)

    def test_partial_export_requires_saved_page_provenance_verification(self):
        self.checkpoint_inputs()
        with self.assertRaisesRegex(ValueError, "passing provenance report"):
            self.export(allow_partial=True)

    def test_partial_export_checks_reported_exhaustion_against_saved_states(self):
        provenance = self.checkpoint_inputs()
        report = json.loads(provenance.read_text())
        report["api_exhausted_conditions"] = 312
        provenance.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "traversal counts disagree"):
            self.export(allow_partial=True, provenance_report=provenance)

    def test_partial_export_keeps_exact_database_row_count_gate(self):
        provenance = self.checkpoint_inputs()
        with sqlite3.connect(self.database) as db:
            db.execute("DELETE FROM trades WHERE trade_row_id=1")
        with self.assertRaisesRegex(ValueError, "every manifested trade"):
            self.export(allow_partial=True, provenance_report=provenance)

    def test_dry_run_does_not_copy_or_create_release_directory(self):
        result = self.export(dry_run=True)
        self.assertTrue(result["dry_run"])
        self.assertGreater(result["estimated_extra_disk_required_bytes"], self.database.stat().st_size)
        self.assertFalse((self.root / "release").exists())


if __name__ == "__main__":
    unittest.main()
