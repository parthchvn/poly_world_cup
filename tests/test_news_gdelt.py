import hashlib
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from poly_world_cup.attribution import _verified_news_time
from poly_world_cup.news_gdelt import (
    _load_cached, _write_gzip, batch_schedule, batch_time, candidate_title,
    collect_gdelt, historical_versions, package_gdelt_evidence, parse_batch, parse_record, retrieval_policy,
)

STAMP = "20260610120000"
RETRIEVED = "2026-09-23T12:00:00Z"


def row(title="Mexico vs South Africa: World Cup preview", url="https://example.org/soccer/preview", **changes):
    fields = [STAMP + "-0", STAMP, "1", "example.org", url] + [""] * 22
    fields[26] = "<PAGE_TITLE>" + title + "</PAGE_TITLE>"
    for key, value in changes.items():
        fields[int(key)] = value
    return "\t".join(fields).encode()


def archive(lines):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(STAMP + ".gkg.csv", b"\n".join(lines) + b"\n")
    return output.getvalue()


class GdeltNewsTests(unittest.TestCase):
    def parse(self, raw):
        return parse_record(raw, stamp=STAMP, row_number=1,
                            archive_sha256="a" * 64, retrieved_at=RETRIEVED)

    def test_capture_not_publisher_claim(self):
        raw = row(**{"26": "<PAGE_TITLE>Mexico vs South Africa</PAGE_TITLE><PAGE_PRECISEPUBTIMESTAMP>20260101000000</PAGE_PRECISEPUBTIMESTAMP>"})
        record = self.parse(raw)
        self.assertEqual(record["availability_upper_utc"], "2026-06-10T12:15:00Z")
        self.assertEqual(record["captured_at_utc"], "2026-06-10T12:00:00Z")
        self.assertIsNone(record["published_at_utc"])
        self.assertIsNotNone(_verified_news_time(record))
        self.assertEqual(record["historical_content_sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(record["fixture_links"], [])

    def test_html_entities_and_exact_record_hash(self):
        raw = row("Cura&#xE7;ao &amp; Germany")
        record = self.parse(raw + b"\r\n")
        self.assertEqual(record["title"], "Curaçao & Germany")
        self.assertEqual(record["historical_content_sha256"], hashlib.sha256(raw).hexdigest())

    def test_batch_identity_and_date_must_agree(self):
        for changes in ({"0": "20260710120000-0"}, {"1": "20260610121500"}):
            with self.assertRaises(ValueError):
                self.parse(row(**changes))

    def test_conflicting_titles_fail(self):
        with self.assertRaises(ValueError):
            self.parse(row(**{"26": "<PAGE_TITLE>first</PAGE_TITLE><PAGE_TITLE>second</PAGE_TITLE>"}))

    def test_nonweb_and_bad_urls_are_not_archived_news(self):
        self.assertIsNone(self.parse(row(**{"2": "2"})))
        self.assertIsNone(self.parse(row(url="file:///private")))
        self.assertIsNone(self.parse(row(url="https://name:secret@example.org/")))

    def test_malformed_encoding_and_fields_fail(self):
        with self.assertRaises(UnicodeDecodeError):
            self.parse(row() + b"\xff")
        with self.assertRaises(ValueError):
            self.parse(b"\t".join(row().split(b"\t")[:-1]))

    def test_candidates_are_broad_and_not_linked(self):
        self.assertTrue(candidate_title("France vs Senegal preview", "https://example.org/a", ["France", "Senegal"]))
        self.assertFalse(candidate_title("France has a new railway", "https://example.org/a", ["France", "Senegal"]))
        raw = archive([row(), row("Weather this weekend", url="https://example.org/weather")])
        records, evidence, meta = parse_batch(raw, stamp=STAMP, retrieved_at=RETRIEVED, teams=["Mexico", "South Africa"])
        self.assertEqual(meta["record_count"], 2)
        self.assertEqual(len(records), 1)
        self.assertEqual(evidence[0]["row_number"], 1)
        self.assertEqual(meta["archive_zip_sha256"], hashlib.sha256(raw).hexdigest())

    def test_zip_layout_and_size_are_bounded(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("../wrong.csv", row())
        with self.assertRaises(ValueError):
            parse_batch(buf.getvalue(), stamp=STAMP, retrieved_at=RETRIEVED, teams=[])
        with patch("poly_world_cup.news_gdelt.MAX_COMPRESSED", 1):
            with self.assertRaises(ValueError):
                parse_batch(archive([row()]), stamp=STAMP, retrieved_at=RETRIEVED, teams=[])

    def test_invalid_utf8_record_is_rejected_without_corrupting_valid_rows(self):
        records, _, meta = parse_batch(archive([row() + b"\xff", row()]), stamp=STAMP,
                                       retrieved_at=RETRIEVED, teams=[])
        self.assertEqual(len(records), 1)
        self.assertEqual(len(meta["invalid_encoding_rows"]), 1)
        self.assertEqual(meta["invalid_encoding_rows"][0]["row_number"], 1)
        self.assertEqual(records[0]["availability_evidence"][0]["archive_row_number"], 2)

    def test_cached_evidence_revalidated_and_corruption_fails(self):
        records, evidence, meta = parse_batch(archive([row()]), stamp=STAMP, retrieved_at=RETRIEVED, teams=[])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            path = folder / (STAMP + ".rows.jsonl.gz")
            meta["retained_evidence_sha256"] = _write_gzip(path, evidence)
            (folder / (STAMP + ".json")).write_text(json.dumps(meta))
            replayed, _ = _load_cached(folder, STAMP)
            self.assertEqual(replayed, records)
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaises(ValueError):
                _load_cached(folder, STAMP)

    def test_network_failure_remains_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch("poly_world_cup.news_gdelt.urlopen", side_effect=OSError("unavailable")):
                report = collect_gdelt(output=Path(temp), stamps=[STAMP], teams=[], workers=1)
            self.assertEqual(report["failed_batches"], 1)
            self.assertFalse(report["historical_coverage_complete"])
            self.assertEqual(report["candidate_historical_title_versions"], 0)

    def test_changed_retrieval_policy_does_not_reuse_narrow_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            collect_gdelt(output=Path(temp), stamps=[], teams=["France"], workers=1)
            with self.assertRaisesRegex(ValueError, "Retrieval policy changed"):
                collect_gdelt(output=Path(temp), stamps=[], teams=["France", "Senegal"], workers=1)

    def test_schedule_validation(self):
        self.assertEqual(batch_schedule(STAMP, "20260610123000", 15),
                         [STAMP, "20260610121500", "20260610123000"])
        for value in ["20260610120100", "20260610120001", "20260230120000"]:
            with self.assertRaises(ValueError):
                batch_time(value)
        with self.assertRaises(ValueError):
            batch_schedule(STAMP, STAMP, 7)

    def test_title_reversions_are_not_erased(self):
        rows = [{"news_id": str(i), "news_item_id": "same", "title": title,
                 "availability_upper_utc": f"2026-06-10T12:{i:02}:00Z"}
                for i, title in enumerate(["A", "A", "B", "A"])]
        versions, conflicts = historical_versions(rows)
        self.assertEqual([r["title"] for r in versions], ["A", "B", "A"])
        self.assertEqual([r["version_rank"] for r in versions], [0, 1, 2])
        self.assertEqual(conflicts, 0)

    def test_simultaneous_conflicting_titles_are_quarantined(self):
        rows = [{"news_id": title, "news_item_id": "same", "title": title,
                 "availability_upper_utc": "2026-06-10T12:00:00Z"}
                for title in ["A", "B"]]
        versions, conflicts = historical_versions(rows)
        self.assertEqual(versions, [])
        self.assertEqual(conflicts, 2)

    def test_evidence_package_checksums_and_exact_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            batches = source / "batches"
            batches.mkdir(parents=True)
            records, evidence, meta = parse_batch(archive([row()]), stamp=STAMP, retrieved_at=RETRIEVED, teams=[])
            meta["retained_evidence_sha256"] = _write_gzip(batches / (STAMP + ".rows.jsonl.gz"), evidence)
            (batches / (STAMP + ".json")).write_text(json.dumps(meta))
            (source / "retrieval_policy.json").write_text(json.dumps(retrieval_policy([])))
            (source / "report.json").write_text(json.dumps({"batches": [meta], "candidate_historical_title_versions": 1}))
            (source / "news.jsonl").write_text(json.dumps(records[0]) + "\n")
            output = root / "package"
            manifest = package_gdelt_evidence(source, output)
            self.assertEqual(manifest["retained_archive_records"], 1)
            for item in manifest["files"]:
                self.assertEqual(hashlib.sha256((output / item["path"]).read_bytes()).hexdigest(), item["sha256"])
            exact = json.loads(gzip.decompress((output / "records/part-00001.jsonl.gz").read_bytes()))
            self.assertEqual(exact["raw_record"].encode(), row())
            with self.assertRaises(FileExistsError):
                package_gdelt_evidence(source, output)


if __name__ == "__main__":
    unittest.main()
