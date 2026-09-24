#!/usr/bin/env python3
"""Recount sealed export shards, then validate the frozen release in full."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poly_world_cup.sequence_token_validation import recount_closed_shards
from poly_world_cup.sequence_validation import require
from scripts.validate_actor_sequences import validate_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","source","evidence","report","tokenizer","token-receipts"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--chat-template",type=Path,default=Path(__file__).resolve().parents[1]/"configs/actor_sequence_chat_template.jinja")
    parser.add_argument("--token-workers",type=int,choices=range(1,33),default=7)
    parser.add_argument("--closed-shard-batch",type=int,default=2)
    parser.add_argument("--poll-seconds",type=int,choices=range(1,51),default=45)
    args = parser.parse_args()
    log = lambda value:print(value,flush=True)
    log("Watching export; independently recounting only sealed shards with one worker")
    while not (args.dataset/"manifest.json").is_file():
        result = recount_closed_shards(args.dataset,args.tokenizer,args.chat_template,args.token_receipts,
            limit=args.closed_shard_batch,progress=log)
        if not result["shards_recomputed"]:
            time.sleep(args.poll_seconds)
    # Existence alone is insufficient: this must be the unsampled full-cohort release.
    # Export writes the manifest last; a short read race must fail rather than certify it.
    manifest = json.loads((args.dataset/"manifest.json").read_text())
    require(manifest.get("sample") is False and manifest.get("status") == "exported","Refusing sampled or unfinished export")
    with sqlite3.connect(f"file:{args.source.resolve()}?mode=ro&immutable=1",uri=True) as source:
        observations,actors = source.execute("SELECT COUNT(*),COUNT(DISTINCT wallet) FROM trades").fetchone()
    require(manifest["selected_observations"] == observations and manifest["counts"]["actors"] == actors,
        "Final export is not the complete recovered cohort")
    log(f"Final manifest found: {actors:,} actors / {observations:,} observations; starting full source validation and final token reconciliation")
    result = validate_release(args,progress=log)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix+".tmp")
    temporary.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    temporary.replace(args.report)
    log(json.dumps({k:v for k,v in result.items() if k!="token_recount"},sort_keys=True))


if __name__ == "__main__":
    main()
