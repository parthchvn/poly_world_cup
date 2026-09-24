#!/usr/bin/env python3
"""Stream one validated actor-conversation partition to messages-only JSONL."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile

PROFILES = ("conditional_trades", "scheduled_windows")
SPLITS = ("train", "validation", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def loads(value):
    return json.loads(value, object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Invalid JSON constant: " + value)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_partition(dataset, validation, profile, split):
    """Bind a complete release and its full independent validation report."""
    require(profile in PROFILES and split in SPLITS, "Choose one known profile and split")
    root, validation = Path(dataset).resolve(), Path(validation).resolve()
    manifest_path = root / "manifest.json"
    manifest_bytes, report_bytes = manifest_path.read_bytes(), validation.read_bytes()
    manifest, report = loads(manifest_bytes), loads(report_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    require(manifest.get("schema_version") == 3 and manifest.get("status") == "exported"
            and manifest.get("sample") is False, "A completed, non-sample v3 manifest is required")
    require(report.get("schema_version") == 1 and report.get("status") == "passed"
            and report.get("manifest_sha256") == manifest_sha, "Passed validation must match this manifest")
    require(report.get("actual_token_counts_recomputed") is True,
            "Full actual-token validation is required, not structural validation alone")
    require(report.get("source_sha256") == manifest["source"]["sha256"]
            and report.get("fixtures") == 104 and report.get("contracts") == 312,
            "Validation does not cover the complete tournament source")
    artifacts = manifest["artifacts"]
    require(report.get("artifact_count") == len(artifacts), "Validation artifact coverage mismatch")
    recount = report.get("token_recount", {})
    require(recount.get("status") == "passed" and recount.get("all_rows_recomputed") is True
            and recount.get("manifest_sha256") == manifest_sha
            and recount.get("method") == "full_apply_chat_template_no_truncation",
            "A matching full, untruncated token recount is required")
    require(recount.get("tokenizer_files") == manifest["tokenizer"]["files"]
            and recount.get("chat_template_sha256") == manifest["tokenizer"]["chat_template_sha256"],
            "Token recount used a different tokenizer or template")
    total_rows = total_tokens = 0
    for task in PROFILES:
        for partition in SPLITS:
            expected = manifest["profiles"][task][partition]
            for key in ("conversations", "tokens"):
                value = expected.get(key, 0)
                require(type(value) is int and value >= 0, "Invalid manifest partition counts")
                require(report["profiles"][task][partition].get(key, 0) == value
                        and recount["profiles"][task][partition].get(key, 0) == value,
                        "Validation omitted conversations or token counts")
            total_rows += expected.get("conversations", 0)
            total_tokens += expected.get("tokens", 0)
    require(recount.get("conversations") == total_rows and recount.get("tokens") == total_tokens,
            "Full token recount totals mismatch")
    conversation_artifacts, selected = {}, []
    for name, metadata in artifacts.items():
        relative = PurePosixPath(name)
        require(relative.parts and not relative.is_absolute() and ".." not in relative.parts
                and str(relative) == name, "Unsafe manifest artifact path")
        if relative.parts[0] in PROFILES:
            require(len(relative.parts) == 3 and relative.parts[1] in SPLITS
                    and relative.name.startswith("part-") and relative.name.endswith(".jsonl.gz"),
                    "Unexpected conversation artifact path")
            conversation_artifacts[name] = metadata
            if tuple(relative.parts[:2]) == (profile, split):
                selected.append(name)
    recounted = {}
    for item in recount.get("files", []):
        name = item["path"]
        require(name not in recounted, "Repeated token-recount shard")
        recounted[name] = {key: item[key] for key in ("bytes", "sha256")}
    require(recounted == conversation_artifacts, "Full token recount shard coverage mismatch")
    actual = {path.relative_to(root).as_posix()
              for path in (root / profile / split).rglob("*") if path.is_file()}
    require(actual == set(selected), "Missing or unlisted selected conversation shard")
    require(selected or manifest["profiles"][profile][split].get("conversations", 0) == 0,
            "Selected partition has no conversation shards")
    return {"root": root, "validation": validation, "manifest": manifest,
            "manifest_sha256": manifest_sha, "validation_sha256": hashlib.sha256(report_bytes).hexdigest(),
            "profile": profile, "split": split, "shards": sorted(selected)}


def iter_partition_messages(partition, statistics):
    """Yield whole conversations only; verify each shard before yielding its rows.

    Consume to exhaustion to run final aggregate and file-stability checks.
    The CLI additionally keeps output private until every check succeeds.
    """
    root, manifest = partition["root"], partition["manifest"]
    for name in partition["shards"]:
        path = root / name
        require(path.resolve().is_relative_to(root) and not path.is_symlink(), "Unsafe selected shard")
        metadata = manifest["artifacts"][name]
        require(path.stat().st_size == metadata["bytes"] and sha(path) == metadata["sha256"],
                "Selected shard checksum mismatch: " + name)
        before = path.stat()
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                require(line.strip(), "Blank conversation row")
                row = loads(line)
                require((row.get("profile"), row.get("split")) == (partition["profile"], partition["split"]),
                        "Conversation partition mismatch")
                messages = row.get("messages")
                require(isinstance(messages, list) and len(messages) >= 3 and len(messages) % 2 == 1,
                        "Expected a complete system/user/assistant conversation")
                for index, message in enumerate(messages):
                    role = "system" if index == 0 else "user" if index % 2 else "assistant"
                    require(isinstance(message, dict) and set(message) == {"role", "content"}
                            and message["role"] == role and isinstance(message["content"], str),
                            f"Invalid message schema or turn order: {name}:{number}")
                require(type(row.get("token_count")) is int and row["token_count"] > 0
                        and type(row.get("requires_long_context")) is bool, "Invalid length metadata")
                statistics["conversations"] += 1
                statistics["target_turns"] += len(messages) // 2
                statistics["tokens"] += row["token_count"]
                statistics["long_context_sequences"] += row["requires_long_context"]
                statistics["max_reference_tokens"] = max(statistics["max_reference_tokens"], row["token_count"])
                yield {"messages": messages}
        after = path.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Selected shard changed while reading")
    expected = manifest["profiles"][partition["profile"]][partition["split"]]
    for key in ("conversations", "target_turns", "tokens", "long_context_sequences"):
        require(statistics[key] == expected.get(key, 0), "Export totals differ from validated manifest: " + key)
    require(sha(root / "manifest.json") == partition["manifest_sha256"]
            and sha(partition["validation"]) == partition["validation_sha256"],
            "Manifest or validation changed during export")


def iter_messages(dataset, validation, profile, split):
    """Load messages directly without creating a second corpus copy.

    Consume to exhaustion for final aggregate/stability checks. Each shard is
    checked against the fully validated manifest before its first row is read.
    """
    partition = checked_partition(dataset, validation, profile, split)
    statistics = dict.fromkeys(("conversations", "target_turns", "tokens", "long_context_sequences", "max_reference_tokens"), 0)
    yield from iter_partition_messages(partition, statistics)


def export_messages(dataset, validation, profile, split, output):
    partition = checked_partition(dataset, validation, profile, split)
    output = Path(output).absolute()
    require(not output.resolve().is_relative_to(partition["root"]), "Output must be outside the frozen dataset")
    require(output.name.endswith((".jsonl", ".jsonl.gz")), "Output must end in .jsonl or .jsonl.gz")
    require(not output.exists() and not output.is_symlink(), "Output already exists; choose a new path")
    require(output.resolve() != partition["validation"], "Output cannot replace the validation report")
    output.parent.mkdir(parents=True, exist_ok=True)
    statistics = dict.fromkeys(("conversations", "target_turns", "tokens", "long_context_sequences", "max_reference_tokens"), 0)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".partial", dir=output.parent)
    try:
        with os.fdopen(fd, "wb") as raw:
            destination = gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0, compresslevel=5) if output.name.endswith(".gz") else raw
            try:
                for row in iter_partition_messages(partition, statistics):
                    destination.write((json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8"))
            finally:
                if destination is not raw:
                    destination.close()
        # Hard-link publication is atomic and fails if another writer created output.
        os.link(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"status": "exported", "profile": profile, "split": split,
            "manifest_sha256": partition["manifest_sha256"], "validation_sha256": partition["validation_sha256"],
            "output": str(output), "output_bytes": output.stat().st_size, "output_sha256": sha(output),
            "fields": ["messages"], "truncated": False, **statistics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export_messages(args.dataset, args.validation, args.profile, args.split, args.output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
