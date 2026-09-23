import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from poly_world_cup.sft import _JSONLines, _verify_gzip_rows
from tests import test_sft as export_fixtures


class SFTGzipIntegrityTests(unittest.TestCase):
    def test_valid_checksum_does_not_make_truncated_gzip_valid(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.jsonl.gz"
            path.write_bytes(gzip.compress(b'{"row":1}\n')[:-5])
            with self.assertRaisesRegex(ValueError, "Incomplete or invalid finalized gzip"):
                _verify_gzip_rows(path, 1)

    def test_crc_valid_wrong_count_or_missing_line_ending_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "part.jsonl.gz"
            path.write_bytes(gzip.compress(b'{"row":1}\n'))
            with self.assertRaisesRegex(ValueError, "wrong row count"):
                _verify_gzip_rows(path, 2)
            path.write_bytes(gzip.compress(b'{"row":1}'))
            with self.assertRaisesRegex(ValueError, "incomplete JSONL"):
                _verify_gzip_rows(path, 0)

    def test_corruption_after_close_prevents_atomic_publication(self):
        fixture = export_fixtures.SFTExportTests()
        fixture.setUp()
        try:
            fixture.build()
            original = _JSONLines.close
            corrupted = []

            def close_then_damage(writer):
                path = Path(writer.raw.name)
                original(writer)
                if "verified_news_only" in path.parts and not corrupted:
                    path.write_bytes(path.read_bytes()[:-10])
                    corrupted.append(path)

            with patch.object(_JSONLines, "close", close_then_damage):
                with self.assertRaisesRegex(ValueError, "Incomplete or invalid finalized gzip"):
                    fixture.export()
            self.assertTrue(corrupted)
            self.assertFalse(fixture.output.exists())
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
