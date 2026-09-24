#!/usr/bin/env python3
"""Validate all selected observations, prediction windows and optional exact tokens."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poly_world_cup.sequence_validation import validate_sequences
from poly_world_cup.sequence_token_validation import recount_tokens,attach_token_recount


def validate_release(args,progress=print):
    if not args.tokenizer:
        return validate_sequences(args.dataset,args.source,args.evidence,progress=progress)
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix="token-coordinator") as coordinator:
        future = coordinator.submit(recount_tokens,args.dataset,args.tokenizer,args.chat_template,
            workers=args.token_workers,progress=progress,receipt_cache=getattr(args,"token_receipts",None))
        structural = validate_sequences(args.dataset,args.source,args.evidence,progress=progress)
        tokens = future.result()
    return attach_token_recount(structural,tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","source","evidence","report"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--tokenizer",type=Path)
    parser.add_argument("--chat-template",type=Path,default=Path(__file__).resolve().parents[1]/"configs/actor_sequence_chat_template.jinja")
    parser.add_argument("--token-workers",type=int,choices=range(1,33),default=7,metavar="1-32")
    parser.add_argument("--token-receipts",type=Path,help="Reuse independently recounted, hash-bound shard receipts stored outside the dataset")
    args = parser.parse_args()
    result = validate_release(args,progress=lambda text:print(text,flush=True))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="token_recount"},sort_keys=True),flush=True)


if __name__ == "__main__": main()
