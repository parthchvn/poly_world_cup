#!/usr/bin/env python3
"""Print an exact, complete actor conversation through the release index."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dataset", type=Path)
    p.add_argument("--actor")
    p.add_argument("--profile", choices=["conditional_trades", "scheduled_windows"])
    p.add_argument("--split", choices=["train", "validation", "test"])
    p.add_argument("--chunk", type=int)
    p.add_argument("--messages-only", action="store_true")
    args = p.parse_args()
    root = args.dataset.resolve()
    for path in sorted((root / "actor_index").glob("*.jsonl.gz")):
        with gzip.open(path, "rt") as f:
            for line in f:
                entry = json.loads(line)
                if any(value is not None and entry[key] != value for key, value in
                    (("actor_id", args.actor), ("profile", args.profile), ("split", args.split), ("chunk_index", args.chunk))):
                    continue
                shard = (root / entry["path"]).resolve()
                if not shard.is_relative_to(root):
                    raise ValueError("Unsafe index path")
                with gzip.open(shard, "rt") as stream:
                    for number, data in enumerate(stream, 1):
                        if number == entry["line"]:
                            row = json.loads(data)
                            if row["sequence_id"] != entry["sequence_id"]:
                                raise ValueError("Index/record identity mismatch")
                            print(json.dumps({"messages": row["messages"]} if args.messages_only else row,
                                             ensure_ascii=False, indent=2))
                            return
                raise ValueError("Index points beyond shard")
    p.error("No conversation matches the requested filters")


if __name__ == "__main__":
    main()
