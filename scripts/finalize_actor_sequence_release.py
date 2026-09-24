#!/usr/bin/env python3
"""Create exact-row previews and mask probes after complete release validation.

This helper never writes to the dataset or edits final documentation. It reads
only manifest-listed, closed shards, with a bounded representative-row search.
The Markdown draft is written under ignored data/ for final editorial review.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.measure_sequence_tokens import (
    REFERENCE_CHAT_TEMPLATE, REFERENCE_PATH, assistant_mask_probe,
    file_sha256, load_reference_tokenizer, validate_messages,
)

PROFILES = ("conditional_trades", "scheduled_windows")
SPLITS = ("train", "validation", "test")
REPO = Path(__file__).resolve().parents[1]
VERIFIED_MASK = "assistant_content_and_end_tokens_verified"


def require(value, message):
    if not value:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return str(path)


def read_validated_release(dataset, validation):
    manifest_path = dataset / "manifest.json"
    manifest_hash = file_sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(validation.read_text())
    require(manifest.get("schema_version") == 3 and manifest.get("sample") is False,
            "An exported full v3 release, not a smoke sample, is required")
    require(report.get("status") == "passed" and report.get("manifest_sha256") == manifest_hash,
            "A passed independent validation of this exact manifest is required")
    recount = report.get("token_recount", {})
    require(report.get("actual_token_counts_recomputed") is True and
            recount.get("status") == "passed" and recount.get("all_rows_recomputed") is True and
            recount.get("manifest_sha256") == manifest_hash,
            "The same-manifest full token recount must have passed")
    require(report.get("source_sha256") == manifest["source"]["sha256"],
            "Validation refers to another recovered source")
    require((report.get("fixtures"), report.get("contracts")) == (104, 312),
            "The final preview requires all 104 fixtures and 312 contracts")
    coverage = manifest["fixture_target_coverage"]
    require(report.get("fixture_target_coverage") == coverage and all(
        coverage.get(key, {}).get("present") == 104 and coverage[key].get("missing") == []
        for key in ("selected_observations", "conditional_targets", "scheduled_target_observations")),
        "Every fixture must have observed and positive-target coverage in both task views")
    return manifest, report, manifest_hash


def row_features(row):
    messages = validate_messages(row["messages"])
    actions = Counter()
    news, history = 0, 0
    for message in messages:
        if message["role"] == "user":
            content = json.loads(message["content"])
            news += len(content.get("news", []))
            history += len(content.get("history", []))
        elif message["role"] == "assistant":
            actions[json.loads(message["content"])["action"]] += 1
    return {"news_entries": news, "explicit_history_observations": history,
            "assistant_turns": sum(actions.values()), "actions": dict(actions),
            "requires_long_context": row["requires_long_context"]}


def score(features, profile):
    both = bool(features["actions"].get("TRADE") and features["actions"].get("NO_TRADE"))
    return (int(both) if profile == "scheduled_windows" else 1,
            int(features["news_entries"] > 0), int(features["explicit_history_observations"] > 0),
            int(features["assistant_turns"] > 1), int(not features["requires_long_context"]))


def bounded_candidates(dataset, manifest, profile, split, *, max_shards, max_rows):
    prefix = f"{profile}/{split}/"
    shard_names = sorted(name for name in manifest["artifacts"]
                         if name.startswith(prefix) and PurePosixPath(name).name.startswith("part-")
                         and name.endswith(".jsonl.gz"))[:max_shards]
    require(shard_names, f"No closed manifest-listed shard for {profile}/{split}")
    best, first, examined = None, None, 0
    scanned = []
    for name in shard_names:
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe manifest path")
        path = dataset / name
        metadata = manifest["artifacts"][name]
        before = path.stat()
        require(before.st_size == metadata["bytes"] and file_sha256(path) == metadata["sha256"],
                f"Closed shard no longer matches the validated manifest: {name}")
        scanned.append({"path": reference(path), "sha256": metadata["sha256"]})
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, 1):
                require(line.strip(), f"Blank row at {name}:{line_number}")
                row = json.loads(line)
                require((row.get("profile"), row.get("split")) == (profile, split), "Shard partition mismatch")
                features = row_features(row)
                candidate = {"source": {"path": reference(path), "line": line_number,
                    "shard_sha256": metadata["sha256"],
                    "jsonl_line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest()},
                    "features": features, "row": row}
                if first is None:
                    first = candidate
                if best is None or score(features, profile) > score(best["features"], profile):
                    best = candidate
                examined += 1
                if examined >= max_rows or score(best["features"], profile) == (1, 1, 1, 1, 1):
                    break
        after = path.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), "Shard changed during preview selection")
        if examined >= max_rows or score(best["features"], profile) == (1, 1, 1, 1, 1):
            break
    return best, first, {"profile": profile, "split": split, "rows_examined": examined,
        "maximum_rows": max_rows, "maximum_shards": max_shards, "shards_read": scanned,
        "selection_method": "best feature coverage in bounded leading shard prefixes; not a random or representative statistical sample"}


def documentation_draft(manifest, validation, policy, preview_path, validation_path):
    counts = manifest["counts"]
    conversations = sum(v.get("conversations", 0) for splits in manifest["profiles"].values() for v in splits.values())
    oversized = sum(v.get("long_context_sequences", 0) for splits in manifest["profiles"].values() for v in splits.values())
    lines = ["# Proposed release-status update", "", "Draft only; final documentation has not been edited.", "",
        f"The v3 captured-observation release has passed full source/context validation and an independent token recount for all 104 fixtures and 312 binary contracts. It contains {manifest['selected_observations']:,} selected observations from {counts['actors']:,} actors, serialized as {conversations:,} complete conversations across two task views.", "",
        "| Task | Split | Conversations | Target turns | Tokens |",
        "| --- | --- | ---: | ---: | ---: |"]
    for profile in PROFILES:
        for split in SPLITS:
            group = manifest["profiles"][profile][split]
            lines.append(f"| `{profile}` | {split} | {group.get('conversations', 0):,} | {group.get('target_turns', 0):,} | {group.get('tokens', 0):,} |")
    lines += ["", f"All {manifest['tokens']['total']:,} reference-template tokens were recounted. The longest conversation is {manifest['tokens']['max']:,} tokens; {oversized:,} conversations exceed the configured {policy['max_tokens']:,}-token target and must not be silently truncated.", "",
        f"The activity filter is `{policy['filter_scope']}` with a maximum of {policy['max_trades_per_market']} captured observations per binary contract. Scheduled queries forecast {policy['horizon_seconds']} seconds; training negatives are retained with probability {policy['train_negative_probability']}, and evaluation negatives with probability {policy['evaluation_negative_probability']}.", "",
        f"Exact unshortened rows and one assistant-mask probe per task/split: `{reference(preview_path)}`. Full validation: `{reference(validation_path)}`.", "",
        f"Manifest SHA-256: `{validation['manifest_sha256']}`. Dataset status: `{manifest.get('status')}`; `format_ready_for_sft={manifest.get('format_ready_for_sft')}`, `prospective_training_ready={manifest.get('prospective_training_ready')}`, `model_training_performed={manifest.get('model_training_performed')}`. These flags are copied from the release and should not be strengthened into an accuracy claim.", "",
        "Use only each row's `messages` for training. Rows are complete conversations; attention does not cross row boundaries. Preview rows are deliberately chosen for readable feature coverage, not sampled for performance estimation. Six mask probes do not certify an actual trainer's labels. Earlier assistant turns are true observed labels during teacher-forced training/evaluation.", "",
        "Retrospective cohort filtering, captured-API absence, selected-cohort monitoring bounds and historical-news availability remain the documented scope. This release does not demonstrate canonical on-chain completeness, actors' private beliefs, deliberate inactivity, predictive accuracy or a completed model-training experiment.", ""]
    return "\n".join(lines)


def finalize_release(dataset, validation_path, tokenizer_path, template_path, preview_path, draft_path,
                     *, max_shards=3, max_rows=2000):
    dataset, validation_path = Path(dataset).resolve(), Path(validation_path).resolve()
    preview_path, draft_path = Path(preview_path).resolve(), Path(draft_path).resolve()
    require(max_shards > 0 and max_rows > 0, "Search bounds must be positive")
    require(not preview_path.is_relative_to(dataset) and not draft_path.is_relative_to(dataset),
            "Reports and drafts must be outside immutable dataset artifacts")
    require(len({preview_path, draft_path, validation_path}) == 3,
            "Preview, documentation draft and existing validation report must have distinct paths")
    require(draft_path.is_relative_to(REPO / "data"), "Documentation draft must stay under ignored data/")
    manifest, validation, manifest_hash = read_validated_release(dataset, validation_path)
    validation_hash = file_sha256(validation_path)
    policy_path = dataset / "policy.json"
    require(file_sha256(policy_path) == manifest["artifacts"]["policy.json"]["sha256"],
            "Policy no longer matches the validated manifest")
    policy = json.loads(policy_path.read_text())
    require(file_sha256(Path(template_path)) == manifest["tokenizer"]["chat_template_sha256"], "Template differs from validated export")
    for name, digest in manifest["tokenizer"]["files"].items():
        require(PurePosixPath(name).name == name and file_sha256(Path(tokenizer_path) / name) == digest,
                "Tokenizer file differs from validated export: " + name)
    tokenizer = load_reference_tokenizer(tokenizer_path, chat_template=template_path)
    examples, probes, searches = {}, [], []
    for profile in PROFILES:
        for split in SPLITS:
            best, first, search = bounded_candidates(dataset, manifest, profile, split,
                                                     max_shards=max_shards, max_rows=max_rows)
            searches.append(search)
            if profile not in examples or score(best["features"], profile) > score(examples[profile]["features"], profile):
                examples[profile] = best
            probe = assistant_mask_probe(tokenizer, first["row"]["messages"])
            require(probe["status"] == VERIFIED_MASK, f"Assistant-mask probe failed: {profile}/{split}: {probe}")
            require(probe["total_tokens"] == first["row"]["token_count"], "Probe token count differs from complete recount")
            probes.append({"profile": profile, "split": split, "sequence_id": first["row"]["sequence_id"],
                           "source": first["source"], **probe})
    require(score(examples["scheduled_windows"]["features"], "scheduled_windows")[0] == 1,
            "Bounded search found no scheduled conversation containing both labels; explicitly increase search bounds")
    require(file_sha256(dataset / "manifest.json") == manifest_hash and file_sha256(validation_path) == validation_hash,
            "Manifest or validation report changed while preparing preview")
    preview = {"schema_version": 1, "status": "preview_and_six_mask_probes_verified",
        "dataset": reference(dataset), "manifest_sha256": manifest_hash,
        "validation": {"path": reference(validation_path), "sha256": validation_hash,
            "status": validation["status"], "manifest_sha256": validation["manifest_sha256"],
            "actual_token_counts_recomputed": validation["actual_token_counts_recomputed"]},
        "release_counts": {key: manifest[key] for key in
            ("selected_observations", "quarantined_observations", "counts", "profiles", "tokens", "fixture_target_coverage")},
        "release_flags": {key: manifest[key] for key in
            ("status", "format_ready_for_sft", "prospective_training_ready", "model_training_performed")},
        "exact_unshortened_rows": [examples[p] for p in PROFILES], "assistant_mask_probes": probes,
        "bounded_search": searches, "rows_shortened": False, "dataset_modified": False,
        "limitations": ["Preview rows are selected within bounded leading shard prefixes, not a statistical sample",
            "Each row object is copied in full; JSON pretty-printing changes formatting, not any field value or message content",
            "Mask verification covers six rows with this tokenizer/template, not a trainer implementation or every loss mask",
            "The independent whole-release report, not the preview, certifies full-row validation",
            *validation.get("limitations", [])]}
    draft = documentation_draft(manifest, validation, policy, preview_path, validation_path)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    draft_path.write_text(draft)
    return {"preview": reference(preview_path), "draft": reference(draft_path),
            "manifest_sha256": manifest_hash, "exact_rows": 2, "mask_probes": len(probes),
            "rows_examined": sum(s["rows_examined"] for s in searches)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "datasets/world_cup_2026_actor_sequences_v3")
    parser.add_argument("--validation", type=Path, default=REPO / "reports/actor_sequences_validation.json")
    parser.add_argument("--tokenizer", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--chat-template", type=Path, default=REFERENCE_CHAT_TEMPLATE)
    parser.add_argument("--preview", type=Path, default=REPO / "reports/actor_sequences_preview.json")
    parser.add_argument("--docs-draft", type=Path, default=REPO / "data/sequence_v3_publication/docs_update_draft.md")
    parser.add_argument("--max-shards-per-group", type=int, default=3)
    parser.add_argument("--max-rows-per-group", type=int, default=2000)
    args = parser.parse_args(argv)
    print(json.dumps(finalize_release(args.dataset, args.validation, args.tokenizer, args.chat_template,
        args.preview, args.docs_draft, max_shards=args.max_shards_per_group,
        max_rows=args.max_rows_per_group), sort_keys=True))


if __name__ == "__main__":
    main()
