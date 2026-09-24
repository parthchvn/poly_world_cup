#!/usr/bin/env python3
"""Reconstruct auxiliary shards privately; never replace active export files."""
import argparse
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poly_world_cup.actor_sequences import SequencePolicy,normalize_observations
from poly_world_cup.sequence_context import ContextCatalog
from poly_world_cup.sequence_validation import canonical,normalize_source,require,sha,source_actors,PROFILES,SPLITS
from poly_world_cup.sft import _JSONLines,_Shards


def compare_existing(original, rebuilt):
    """Every intact blob must match exactly; a truncated blob must be a prefix."""
    old,new = original.read_bytes(),rebuilt.read_bytes()
    require(new.startswith(old),f"Reconstruction differs from existing compressed prefix: {original}")
    count=0
    try:
        with gzip.open(original,"rb") as stream:
            for line in stream:
                count+=1
        require(old==new,f"Previously intact shard changed: {original}")
        status="intact_exact_match"
    except EOFError:
        status="truncated_prefix_preserved"
    return {"original_sha256":hashlib.sha256(old).hexdigest(),"original_bytes":len(old),
        "sha256":hashlib.sha256(new).hexdigest(),"bytes":len(new),"old_status":status,
        "original_readable_rows":count}


def rebuild_observation_shard(dataset,source,evidence,output,number):
    require(not output.resolve().is_relative_to(dataset.resolve()),"Repair output must be outside active dataset")
    policy=SequencePolicy(**json.loads((dataset/"policy.json").read_text()))
    splits=json.loads((dataset/"split_policy.json").read_text())
    catalog=ContextCatalog(evidence,policy.initial_news_per_scope)
    start,end=(number-1)*100000+1,number*100000
    target=output/f"observations/part-{number:05d}.jsonl.gz"
    original=dataset/f"observations/part-{number:05d}.jsonl.gz"
    before=source.stat()
    writer=_JSONLines(target)
    boundaries={}
    seen=0
    try:
        with sqlite3.connect(f"file:{source.resolve()}?mode=ro&immutable=1",uri=True) as db:
            db.row_factory=sqlite3.Row
            for actor,raw in source_actors(db,policy):
                if seen+len(raw)<start-1:
                    seen+=len(raw)
                    continue
                normalized=normalize_observations(raw,catalog,splits)
                require(normalized==normalize_source(raw,catalog,splits),"Independent observation normalization disagrees")
                for row in normalized:
                    seen+=1
                    if seen in (start-1,end+1):boundaries[seen]=row
                    if start<=seen<=end:writer.write(row)
                if seen>end:break
    finally:
        writer.close()
    require(writer.count==100000,"Rebuilt observation shard has wrong row count")
    left=None
    with gzip.open(dataset/f"observations/part-{number-1:05d}.jsonl.gz","rt") as stream:
        for line in stream:left=json.loads(line)
    with gzip.open(dataset/f"observations/part-{number+1:05d}.jsonl.gz","rt") as stream:
        right=json.loads(next(stream))
    require(left==boundaries[start-1] and right==boundaries[end+1],"Adjacent observation shard boundaries disagree with source ordinals")
    count=0
    with gzip.open(target,"rt") as rebuilt,gzip.open(original,"rt") as old:
        try:
            for line in old:
                require(json.loads(line)==json.loads(next(rebuilt)),"Preserved observation row changed")
                count+=1
        except EOFError:
            pass
    check=compare_existing(original,target)
    with gzip.open(target,"rb") as stream:
        decoded_rows=sum(1 for _ in stream)
    require(decoded_rows==100000,"Rebuilt observation gzip fails complete row reconciliation")
    after=source.stat()
    require((before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),"Source changed during reconstruction")
    result={"kind":"observations","path":target.relative_to(output).as_posix(),"rows":writer.count,"decoded_rows":decoded_rows,
        "source_ordinal_first":start,"source_ordinal_last":end,"preserved_rows_compared":count,
        "source_sha256":sha(source),"adjacent_boundaries_match":True,**check}
    return result


def replacement(path,dataset,overlay):
    relative=path.relative_to(dataset).as_posix()
    if isinstance(overlay,dict) and relative in overlay:
        selected=Path(overlay[relative]["path"])
        require(sha(selected)==overlay[relative]["sha256"],"Prepared conversation replacement hash mismatch")
        return selected
    if overlay is not None and not isinstance(overlay,dict) and (overlay/relative).is_file():
        return overlay/relative
    return path


def row_key(row):
    return row["actor_id"],SPLITS.index(row["split"]),PROFILES.index(row["profile"]),row["chunk_index"]


def indexed_stream(dataset,profile,split,overlay=None,paths=None,state=None):
    previous=None
    paths=sorted((dataset/profile/split).glob("part-*.jsonl.gz")) if paths is None else paths
    for shard,path in enumerate(paths,1):
        require(path.name==f"part-{shard:05d}.jsonl.gz","Noncontiguous conversation shards")
        relative=path.relative_to(dataset).as_posix()
        path=replacement(path,dataset,overlay)
        with gzip.open(path,"rt",encoding="utf-8") as stream:
            for line,value in enumerate(stream,1):
                row=json.loads(value)
                require((row["profile"],row["split"])==(profile,split),"Conversation partition mismatch")
                key=row_key(row)
                require(previous is None or key>previous,"Conversation stream is not strictly ordered")
                previous=key
                record={k:row[k] for k in ("sequence_id","actor_id","profile","split","chunk_index")}
                record.update(path=relative,line=line,sha256=hashlib.sha256(canonical(row).encode()).hexdigest())
                if state is not None:state["head_key"]=key
                yield key,record
    if state is not None:state["finished"]=True


def rebuild_index(dataset,output,overlay=None,last_shard=None,*,shard_size=100000):
    require(last_shard is not None or (dataset/"manifest.json").is_file(),"Wait for the export to finish before rebuilding the complete index")
    require(last_shard is None or type(last_shard) is int and last_shard>0,"Invalid last index shard")
    require(not output.resolve().is_relative_to(dataset.resolve()),"Repair output must be outside active dataset")
    bound=None if last_shard is None else last_shard*shard_size
    paths={(p,s):sorted((dataset/p/s).glob("part-*.jsonl.gz")) for p in PROFILES for s in SPLITS}
    if bound is not None:
        # Snapshot only sealed shards. Each stream must extend strictly beyond
        # the requested prefix, so truncating the snapshots cannot omit rows.
        paths={key:value[:-1] for key,value in paths.items()}
        require(all(paths.values()),"Not every partition has a sealed shard yet")
    writer=_Shards(output/"actor_index",shard_size)
    previous=None
    next_record=None
    boundaries={}
    frontiers={(p,s):{"finished":False} for p in PROFILES for s in SPLITS}
    try:
        streams=[indexed_stream(dataset,p,s,overlay,paths[p,s],frontiers[p,s]) for p in PROFILES for s in SPLITS]
        for position,(key,record) in enumerate(heapq.merge(*streams,key=lambda item:item[0]),1):
            require(previous is None or key>previous,"Merged conversation keys are not unique and ordered")
            if bound is not None and position>bound:
                next_record=record
                break
            previous=key
            writer.write(record)
            number=(position-1)//shard_size+1
            if (position-1)%shard_size==0:boundaries[number]={"first_key":key,"first_sequence_id":record["sequence_id"]}
            boundaries[number].update(last_key=key,last_sequence_id=record["sequence_id"])
    finally:
        writer.close()
    if bound is None:
        manifest=json.loads((dataset/"manifest.json").read_text())
        expected=sum(manifest["profiles"][p][s].get("conversations",0) for p in PROFILES for s in SPLITS)
    else:
        expected=bound
        require(next_record is not None,"Closed conversation snapshots do not cover the entire requested index prefix")
        require(all(not state["finished"] and state.get("head_key",())>previous for state in frontiers.values()),
            "A closed conversation partition ends before the required global prefix")
        with gzip.open(dataset/f"actor_index/part-{last_shard+1:05d}.jsonl.gz","rt") as stream:
            require(json.loads(next(stream))==next_record,"Right index boundary differs from the original next shard")
    require(writer.count==expected,"Reconstructed index omits or adds conversations")
    rebuilt_paths=sorted((output/"actor_index").glob("part-*.jsonl.gz"))
    if bound is None:
        require({p.name for p in rebuilt_paths}=={p.name for p in (dataset/"actor_index").glob("part-*.jsonl.gz")},"Reconstructed index shard set differs")
    else:
        require(len(rebuilt_paths)==last_shard,"Reconstructed bounded index shard set differs")
    files=[]
    for number,path in enumerate(rebuilt_paths,1):
        with gzip.open(path,"rb") as stream:decoded_rows=sum(1 for _ in stream)
        require(decoded_rows==min(shard_size,expected-(number-1)*shard_size),"Rebuilt index gzip row count mismatch")
        files.append({"path":path.relative_to(output).as_posix(),"decoded_rows":decoded_rows,
            "global_first":(number-1)*shard_size+1,"global_last":min(number*shard_size,expected),
            **boundaries[number],**compare_existing(dataset/"actor_index"/path.name,path)})
    return {"kind":"actor_index","status":"verified_private_index_reconstruction_not_installed",
        "rows":writer.count,"files":files,"last_shard":last_shard,
        "conversation_overlays":overlay if isinstance(overlay,dict) else None,
        "closed_stream_frontiers":{p+"/"+s:state for (p,s),state in frontiers.items()} if bound is not None else None,
        "right_boundary_matches":next_record is not None,"right_boundary_sequence_id":None if next_record is None else next_record["sequence_id"]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=("observations","actor_index"))
    for name in ("dataset","output","report"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source",type=Path)
    parser.add_argument("--evidence",type=Path)
    parser.add_argument("--shard",type=int)
    parser.add_argument("--conversation-overlay",type=Path)
    parser.add_argument("--last-index-shard",type=int,help="Rebuild a sealed prefix while the remaining export runs")
    args=parser.parse_args()
    if args.kind=="observations":
        require(args.source and args.evidence and args.shard and args.shard>1,"Observation repair requires source, evidence and interior shard number")
        result=rebuild_observation_shard(args.dataset,args.source,args.evidence,args.output,args.shard)
    else:
        overlay=args.conversation_overlay
        if overlay is not None and overlay.is_file():overlay=json.loads(overlay.read_text())
        result=rebuild_index(args.dataset,args.output,overlay,args.last_index_shard)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":main()
