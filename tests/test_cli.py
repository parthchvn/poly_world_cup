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


if __name__ == "__main__":
    unittest.main()
