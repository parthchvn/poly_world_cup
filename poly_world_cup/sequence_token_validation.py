"""Bounded parallel recount with the actual full chat template for every row."""
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
from importlib.metadata import version
import json
from multiprocessing import get_context
import os
from pathlib import Path, PurePosixPath

from .sequence_validation import PROFILES, SPLITS, require, sha

_TOKENIZER = None
_METHOD = "full_apply_chat_template_no_truncation"


def _fingerprint(reference, template):
    return {"tokenizer_files":{p.name:sha(p) for p in Path(reference).iterdir() if p.is_file()},
        "chat_template_sha256":sha(template),"implementation_sha256":sha(__file__),
        "runtime_versions":{name:version(name) for name in ("transformers","tokenizers","jinja2")},
        "method":_METHOD,"schema_version":1}


def _receipt_path(cache, context, name, metadata):
    key = json.dumps({"context":context,"path":name,**metadata},sort_keys=True,separators=(",",":"))
    return Path(cache)/(hashlib.sha256(key.encode()).hexdigest()+".json")


def _read_receipt(cache, context, name, metadata):
    if cache is None:
        return None
    path = _receipt_path(cache,context,name,metadata)
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    require(value.get("context") == context,"Token receipt fingerprint mismatch")
    result = value["result"]
    payload = json.dumps(result,sort_keys=True,separators=(",",":"))
    require(value.get("result_sha256") == hashlib.sha256(payload.encode()).hexdigest(),"Corrupt token receipt")
    profile,split,_ = PurePosixPath(name).parts
    require(set(result) == {"path","sha256","bytes","profile","split","conversations","tokens"}
        and result["path"] == name and result["sha256"] == metadata["sha256"]
        and result["bytes"] == metadata["bytes"] and (result["profile"],result["split"]) == (profile,split)
        and all(type(result[k]) is int and result[k]>0 for k in ("conversations","tokens")),"Invalid token receipt")
    return result


def _write_receipt(cache, context, result):
    if cache is None:
        return
    path = _receipt_path(cache,context,result["path"],{k:result[k] for k in ("sha256","bytes")})
    path.parent.mkdir(parents=True,exist_ok=True)
    payload = json.dumps(result,sort_keys=True,separators=(",",":"))
    value = {"context":context,"result":result,"result_sha256":hashlib.sha256(payload.encode()).hexdigest()}
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,sort_keys=True)+"\n")
    temporary.replace(path)


def _cache_outside_dataset(cache, root):
    if cache is not None:
        require(not Path(cache).resolve().is_relative_to(Path(root).resolve()),"Token receipts must be outside the dataset")


def recount_closed_shards(dataset_root,tokenizer_path,template_path,receipt_cache,*,limit=8,progress=print):
    """Recount only shards sealed by opening a subsequent same-partition shard.

    Receipts are provisional evidence until a final manifest binds the exact
    compressed bytes and the complete per-partition row/token totals.
    """
    root,reference,template = Path(dataset_root).resolve(),Path(tokenizer_path).resolve(),Path(template_path).resolve()
    _cache_outside_dataset(receipt_cache,root)
    require(type(limit) is int and limit>0,"Closed-shard batch limit must be positive")
    context = _fingerprint(reference,template)
    tasks = []
    for profile in PROFILES:
        for split in SPLITS:
            paths = sorted((root/profile/split).glob("part-*.jsonl.gz"))
            for path in paths[:-1]:
                name = path.relative_to(root).as_posix()
                metadata = {"bytes":path.stat().st_size,"sha256":sha(path)}
                if _read_receipt(receipt_cache,context,name,metadata) is None:
                    tasks.append((str(root),name,metadata))
                    if len(tasks)>=limit:
                        break
            if len(tasks)>=limit:
                break
        if len(tasks)>=limit:
            break
    if not tasks:
        return {"shards_recomputed":0,"conversations":0,"tokens":0}
    totals = Counter(shards_recomputed=0,conversations=0,tokens=0)
    for result in _results(tasks,1,(str(reference),str(template),context["tokenizer_files"],context["chat_template_sha256"])):
        _write_receipt(receipt_cache,context,result)
        totals.update(shards_recomputed=1,conversations=result["conversations"],tokens=result["tokens"])
        progress(f"Sealed-shard full-template receipt: {result['path']} / {result['conversations']:,} rows / {result['tokens']:,} tokens")
    require(_fingerprint(reference,template) == context,"Tokenizer/template/validator changed during provisional recount")
    return dict(totals)


def _initialize(reference, template, expected_files, expected_template):
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    root = Path(reference)
    require({p.name:sha(p) for p in root.iterdir() if p.is_file()} == expected_files, "Reference tokenizer files differ from export")
    require(sha(template) == expected_template,"Chat template differs from export")
    _TOKENIZER = AutoTokenizer.from_pretrained(root,local_files_only=True)
    _TOKENIZER.chat_template = Path(template).read_text()


def _recount_file(task):
    root,name,metadata = task
    path = Path(root)/name
    profile,split,_ = PurePosixPath(name).parts
    require(path.stat().st_size == metadata["bytes"] and sha(path) == metadata["sha256"],"Recount shard checksum mismatch")
    rows,total = 0,0
    with gzip.open(path,"rt",encoding="utf-8") as stream:
        for line in stream:
            require(line.strip(),"Blank conversation row")
            row = json.loads(line)
            require((row["profile"],row["split"]) == (profile,split),"Token row partition mismatch")
            require(type(row["token_count"]) is int and row["token_count"]>0,"Invalid declared token count")
            actual = len(_TOKENIZER.apply_chat_template(row["messages"],tokenize=True,
                add_generation_prompt=False,enable_thinking=False,truncation=False,padding=False))
            require(actual == row["token_count"],f"Actual chat token count mismatch: {name}:{rows+1}")
            rows += 1; total += actual
    require(rows>0,"Empty conversation shard")
    require(path.stat().st_size == metadata["bytes"] and sha(path) == metadata["sha256"],"Recount shard changed while reading")
    return {"path":name,**metadata,"profile":profile,"split":split,"conversations":rows,"tokens":total}


def _results(tasks,workers,initargs):
    if workers == 1:
        _initialize(*initargs)
        for task in tasks: yield _recount_file(task)
        return
    with ProcessPoolExecutor(max_workers=workers,mp_context=get_context("spawn"),initializer=_initialize,initargs=initargs) as pool:
        iterator,pending = iter(tasks),deque()
        for _ in range(workers*2):
            task = next(iterator,None)
            if task is not None: pending.append(pool.submit(_recount_file,task))
        while pending:
            result = pending.popleft().result()
            task = next(iterator,None)
            if task is not None: pending.append(pool.submit(_recount_file,task))
            yield result


def recount_tokens(dataset_root,tokenizer_path,template_path,*,workers=7,progress=print,receipt_cache=None):
    require(type(workers) is int and 1<=workers<=32,"Token workers must be between 1 and 32")
    root,reference,template = Path(dataset_root).resolve(),Path(tokenizer_path).resolve(),Path(template_path).resolve()
    _cache_outside_dataset(receipt_cache,root)
    context = _fingerprint(reference,template)
    manifest_sha = sha(root/"manifest.json")
    manifest = json.loads((root/"manifest.json").read_text())
    expected_files = manifest["tokenizer"]["files"]
    expected_template = manifest["tokenizer"]["chat_template_sha256"]
    require(sha(template) == expected_template,"Reference template hash mismatch")
    require({p.name:sha(p) for p in reference.iterdir() if p.is_file()} == expected_files,"Reference tokenizer hash mismatch")
    entries = []
    for name,metadata in manifest["artifacts"].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts,"Unsafe artifact path")
        if path.parts[0] in PROFILES:
            require(len(path.parts)==3 and path.parts[1] in SPLITS and path.name.startswith("part-") and path.name.endswith(".jsonl.gz"),"Unexpected conversation artifact")
            entries.append((name,metadata))
    require(entries,"No conversations to recount")
    actual_files = {p.relative_to(root).as_posix() for profile in PROFILES for p in (root/profile).rglob("*") if p.is_file()}
    require(actual_files == {name for name,_ in entries},"Missing or unlisted conversation shard")
    profiles = {p:{s:Counter() for s in SPLITS} for p in PROFILES}
    files,pending_tasks = [],[]
    for name,metadata in sorted(entries):
        path = root/name
        require(path.stat().st_size == metadata["bytes"] and sha(path) == metadata["sha256"],"Recount shard checksum mismatch")
        cached = _read_receipt(receipt_cache,context,name,metadata)
        if cached is None:
            pending_tasks.append((str(root),name,metadata))
        else:
            files.append(cached)
            profiles[cached["profile"]][cached["split"]].update(conversations=cached["conversations"],tokens=cached["tokens"])
    reused = len(files)
    if reused:
        progress(f"Reusing {reused:,}/{len(entries):,} exact closed-shard full-template receipts")
    tasks = iter(pending_tasks)
    for result in _results(tasks,workers,(str(reference),str(template),expected_files,expected_template)):
        _write_receipt(receipt_cache,context,result)
        profiles[result["profile"]][result["split"]].update(conversations=result["conversations"],tokens=result["tokens"])
        files.append(result)
        if len(files)==1 or len(files)%16==0 or len(files)==len(entries):
            progress(f"Actual full-template recount: {len(files):,}/{len(entries):,} shards")
    for profile in PROFILES:
        for split in SPLITS:
            for key in ("conversations","tokens"):
                require(profiles[profile][split][key] == manifest["profiles"][profile][split].get(key,0),"Full token recount omitted rows or differs from manifest totals")
    require(sha(root/"manifest.json") == manifest_sha,"Manifest changed during token recount")
    require(sha(template) == expected_template and {p.name:sha(p) for p in reference.iterdir() if p.is_file()} == expected_files,"Tokenizer/template changed during recount")
    require(_fingerprint(reference,template) == context,"Token validator changed during recount")
    # Recheck cached files after the entire pass, just like newly recounted files.
    for name,metadata in entries:
        require((root/name).stat().st_size == metadata["bytes"] and sha(root/name) == metadata["sha256"],"Recount shard changed before completion")
    files.sort(key=lambda value:value["path"])
    return {"schema_version":1,"status":"passed","all_rows_recomputed":True,
        "method":_METHOD,"manifest_sha256":manifest_sha,"receipt_shards_reused":reused,
        "implementation_sha256":context["implementation_sha256"],
        "tokenizer_files":expected_files,"chat_template_sha256":expected_template,"workers":workers,
        "max_queued_shards":workers*2,"files":files,"profiles":{p:{s:dict(c) for s,c in ps.items()} for p,ps in profiles.items()},
        "conversations":sum(f["conversations"] for f in files),"tokens":sum(f["tokens"] for f in files)}


def attach_token_recount(structural,recount):
    require(structural.get("status") == recount.get("status") == "passed" and recount.get("all_rows_recomputed") is True,"Incomplete validation cannot be attached")
    require(structural["manifest_sha256"] == recount["manifest_sha256"],"Token recount and source validation used different manifests")
    for profile in PROFILES:
        for split in SPLITS:
            for key in ("conversations","tokens"):
                require(structural["profiles"][profile][split].get(key,0) == recount["profiles"][profile][split].get(key,0),"Token/source validation row counts differ")
    return {**structural,"actual_token_counts_recomputed":True,"token_recount":recount}
