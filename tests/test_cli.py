import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from poly_world_cup.cli import main


class CliTests(unittest.TestCase):
    def test_discovery_reports_partial_failure_even_with_104_fixtures(self):
        result = {"fixtures": [{"fixture_id": str(i)} for i in range(104)], "contracts": [],
                  "report": {"coverage_complete": False, "failures": [{"code": "unmatched_fixture"}]}}
        with tempfile.TemporaryDirectory() as folder, patch("poly_world_cup.registry.discover_registry", return_value=result), contextlib.redirect_stdout(io.StringIO()):
            exit_code = main(["discover", "--output", folder, "--cache", str(Path(folder) / "cache")])
            self.assertEqual(exit_code, 2)
            self.assertEqual(json.loads((Path(folder) / "registry.json").read_text()), result)

    def test_news_source_errors_return_failure_but_absent_recaps_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            registry = Path(folder) / "registry.json"
            registry.write_text("{}")
            for errors, expected in [([{"error": "source unavailable"}], 2), ([], 0)]:
                with self.subTest(errors=errors), patch("poly_world_cup.news.collect_news", return_value={
                    "errors": errors, "summary_articles_missing": ["espn:123"]
                }), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["collect-news", "--registry", str(registry),
                                           "--cache", folder, "--output", folder]), expected)

    def test_partial_or_corrupt_trades_do_not_reach_attribution_builder(self):
        with tempfile.TemporaryDirectory() as folder:
            registry = Path(folder) / "registry.json"
            registry.write_text("{}")
            for corrupt, allow_partial in [(False, False), (True, False), (True, True)]:
                audit = {"structural_audit_passed": not corrupt,
                         "structural_errors": ["corrupt page"] if corrupt else [],
                         "conditions_without_manifests": 1, "manifest_status_counts": {"exhausted": 311},
                         "contract_count": 312}
                with self.subTest(corrupt=corrupt, allow_partial=allow_partial), patch(
                    "poly_world_cup.cli.audit_registry", return_value=audit
                ), patch("poly_world_cup.attribution.build_attribution_index") as builder, contextlib.redirect_stderr(io.StringIO()):
                    args = ["attribute", "--registry", str(registry)]
                    if allow_partial:
                        args.append("--allow-partial")
                    self.assertEqual(main(args), 2)
                    builder.assert_not_called()

    def test_archive_uses_capture_window_not_current_publication_claim(self):
        with tempfile.TemporaryDirectory() as folder:
            news = Path(folder) / "news.jsonl"
            records = [{"news_item_id": "republished", "within_study_window": False},
                       {"news_item_id": "dated_in_window", "within_study_window": True}]
            news.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            with patch("poly_world_cup.news_archive.ArchiveCollector") as collector, contextlib.redirect_stdout(io.StringIO()):
                collector.return_value.collect.return_value = {}
                self.assertEqual(main(["archive-news", "--news", str(news), "--output", folder]), 0)
                self.assertEqual(collector.return_value.collect.call_args.args[0], records)


if __name__ == "__main__":
    unittest.main()
