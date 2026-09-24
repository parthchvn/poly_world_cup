#!/usr/bin/env python3
"""Copy an actor-market export, omitting two repeated news metadata fields."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import tempfile


def compact_export(source: Path, destination: Path):
    source, destination = source.resolve(), destination.resolve()
    if not (source / "actors").is_dir():
        raise ValueError(f"No actors directory in {source}")
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError("Choose a new --out directory outside the source export")
    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="compact-actor-news-", dir=destination.parent))
    counts = {"actor_files": 0, "rows": 0, "fields_removed": 0}
    news_fields = set()
    try:
        shutil.copytree(source, work, dirs_exist_ok=True)
        for path in sorted((work / "actors").iterdir()):
            if not (path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz")):
                continue
            opener = gzip.open if path.name.endswith(".gz") else open
            temporary = path.with_name(path.name + ".tmp")
            with opener(path, "rt", encoding="utf-8") as incoming, opener(temporary, "wt", encoding="utf-8") as outgoing:
                for line in incoming:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for item in row.get("news", []):
                        for field in ("time_basis", "match_clock"):
                            if field in item:
                                del item[field]
                                counts["fields_removed"] += 1
                        news_fields.update(item)
                    outgoing.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
                    counts["rows"] += 1
            temporary.replace(path)
            counts["actor_files"] += 1
        manifest_path = work / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["news_feature_fields"] = sorted(news_fields) if news_fields else ["time", "type", "text"]
            manifest["news_source_metadata"] = "espn_events.jsonl"
            manifest["timestamp_semantics"] = "ESPN provider wallclock is an occurrence proxy; per-event timing bases and explicit estimates are recorded in espn_events.jsonl"
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if destination.exists():
            raise ValueError(f"Output appeared during conversion: {destination}")
        work.rename(destination)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    print(json.dumps({"output": str(destination), **counts}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Existing actor-market export directory")
    parser.add_argument("--out", type=Path, required=True, help="New export directory")
    args = parser.parse_args()
    try:
        compact_export(args.source, args.out)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
