#!/usr/bin/env python3
"""Retrieve independently timestamped GKG headlines, with resumable evidence."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.news_gdelt import batch_schedule, collect_gdelt, package_gdelt_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="20260608000000")
    parser.add_argument("--end", default="20260719234500")
    parser.add_argument("--step-minutes", type=int, default=180)
    parser.add_argument("--stamps", type=Path, help="JSON list of exact batches; unioned with already successful batches")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--package-output", type=Path, help="Create a new compact archive-evidence package after collection")
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text())
    fixtures = registry["fixtures"]
    teams = set()
    for fixture in fixtures:
        for key in ("home_team", "away_team"):
            team = fixture[key]
            teams.add(team["name"] if isinstance(team, dict) else team)
    if args.stamps:
        stamps = json.loads(args.stamps.read_text())
    else:
        stamps = batch_schedule(args.start, args.end, args.step_minutes)
    for path in (args.output / "batches").glob("*.json"):
        if not path.name.endswith(".error.json"):
            stamps.append(path.stem)
    def progress(value):
        if value["completed_batches"] % 20 == 0 or value["completed_batches"] == value["requested_batches"]:
            print(json.dumps(value), flush=True)
    report = collect_gdelt(output=args.output, stamps=stamps, teams=sorted(teams), workers=args.workers, progress=progress)
    print(json.dumps({k: v for k, v in report.items() if k != "batches"}, indent=2))
    if args.package_output:
        manifest = package_gdelt_evidence(args.output, args.package_output)
        print(json.dumps({"evidence_files": len(manifest["files"]),
                          "retained_archive_records": manifest["retained_archive_records"]}))


if __name__ == "__main__":
    main()
