#!/usr/bin/env python3
"""Publish an explicit file list through an active, authorized GitHub MCP connection.

The default is a local dry run. --publish uploads content-addressed blobs and
advances a branch without force. Supply --expected-head after reviewing the
branch and --hosted-apps-helper pointing to the session's existing
library_hosted_apps.py. This script neither supplies nor manages credentials.
With --publish --stage-only, upload and checkpoint the blobs without creating
a tree or commit or moving a branch. A later --publish resumes those uploads.

The JSON file list is an array of {"local_path": "...", "path": "repo/path",
"sha256": "optional expected digest"}. Relative local paths resolve beside it.
The checkpoint is local operational state and should not be published.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import sys
from threading import Lock
from typing import Any
from urllib.parse import quote


MAX_BLOB_BYTES = 99 * 1024 * 1024
TREE_BATCH_SIZE = 25
REQUIRED_TOOLS = {"fetch", "create_blob", "create_tree", "create_commit", "update_ref"}


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def load_files(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("File list must be a nonempty JSON array")
    result, seen = [], set()
    for row in rows:
        target = row["path"]
        parts = PurePosixPath(target).parts
        if (not target or target.startswith("/") or "\\" in target or "\0" in target
                or any(part in ("..", ".git") for part in parts)
                or str(PurePosixPath(target)) != target or target in seen):
            raise ValueError(f"Invalid or duplicate repository path: {target!r}")
        local = Path(row["local_path"])
        if not local.is_absolute():
            local = path.parent / local
        if local.is_symlink() or not local.is_file():
            raise ValueError(f"Expected a regular local file: {local}")
        if local.stat().st_size > MAX_BLOB_BYTES:
            raise ValueError(f"Blob exceeds the 99 MiB limit: {target}")
        data = local.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if row.get("sha256") is not None and row["sha256"] != digest:
            raise ValueError(f"Expected SHA-256 does not match: {target}")
        result.append({"path": target, "local_path": str(local.resolve()),
                       "sha256": digest, "git_sha": git_blob_sha(data), "bytes": len(data)})
        seen.add(target)
    return sorted(result, key=lambda row: row["path"])


def unwrap(result: dict[str, Any]) -> dict[str, Any]:
    """Remove standard MCP and GitHub fetch text envelopes, never arbitrary fields."""
    current: Any = result
    for _ in range(10):
        if isinstance(current, str):
            current = json.loads(current)
        elif isinstance(current, dict) and isinstance(current.get("structuredContent"), dict):
            current = current["structuredContent"]
        elif isinstance(current, dict) and isinstance(current.get("content"), str):
            current = current["content"]
        elif isinstance(current, dict) and isinstance(current.get("content"), list):
            blocks = current["content"]
            texts = [b["text"] for b in blocks if b.get("type") == "text"]
            if len(texts) != 1:
                raise ValueError("Expected one JSON text block from GitHub")
            current = texts[0]
        elif isinstance(current, dict):
            return current
        else:
            raise ValueError("GitHub response is not a JSON object")
    raise ValueError("GitHub response envelope is too deeply nested")


class GitHubTools:
    def __init__(self, helper: Path):
        spec = importlib.util.spec_from_file_location("publication_hosted_apps", helper)
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load the existing hosted-app client")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.client = module.HostedAppsClient()
        discovered, cursor, seen = [], None, set()
        for _ in range(100):
            response = self.client._rpc("tools/list", {} if cursor is None else {"cursor": cursor})
            discovered.extend(response.get("tools", []))
            cursor = response.get("nextCursor", response.get("next_cursor"))
            if not cursor:
                break
            if cursor in seen:
                raise ValueError("Hosted-app tool discovery repeated a cursor")
            seen.add(cursor)
        else:
            raise ValueError("Hosted-app tool discovery exceeded the page limit")
        self.tools = {}
        for short in REQUIRED_TOOLS:
            matches = [tool for tool in discovered
                       if tool.get("name") in (f"github_{short}", f"github.{short}",
                                               f"mcp__codex_apps__github_{short}")]
            if len(matches) != 1:
                raise ValueError(f"Need exactly one advertised GitHub {short} tool")
            tool = matches[0]
            meta = tool.get("_meta", {})
            connector = next((meta[k] for k in ("connector_id", "connectorId", "app_id", "appId")
                              if meta.get(k)), None)
            if connector is None:
                raise ValueError(f"GitHub {short} is missing connector metadata")
            self.tools[short] = (connector, tool["name"])
        if len({value[0] for value in self.tools.values()}) != 1:
            raise ValueError("Publication tools must belong to one GitHub connection")

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        connector, full_name = self.tools[name]
        return unwrap(self.client.call_tool(connector, full_name, arguments))

    def head(self, repository: str, branch: str) -> str:
        return self.call("fetch", {
            "url": f"https://api.github.com/repos/{repository}/git/ref/heads/{quote(branch, safe='/')}"
        })["object"]["sha"]


def write_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def publish(files: list[dict[str, Any]], args: argparse.Namespace, github: GitHubTools) -> str | None:
    checkpoint = args.checkpoint
    if checkpoint.resolve() in {Path(row["local_path"]) for row in files}:
        raise ValueError("Do not include the publication checkpoint in the file list")
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
        "repository": args.repository, "branch": args.branch, "uploaded": {}, "status": "planned"
    }
    if state["repository"] != args.repository or state["branch"] != args.branch:
        raise ValueError("Checkpoint belongs to another repository or branch")
    plan = [{k: row[k] for k in ("path", "git_sha", "sha256", "bytes")} for row in files]
    fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    current = github.head(args.repository, args.branch)
    if state.get("commit") == current and state.get("plan_sha256") == fingerprint:
        state["status"] = "published"
        write_checkpoint(checkpoint, state)
        return current
    if current != args.expected_head:
        raise ValueError("Branch changed from --expected-head; review its new contents before publishing")
    state.update(plan_sha256=fingerprint, base_commit=current, status="uploading")
    write_checkpoint(checkpoint, state)
    lock = Lock()

    def upload(row: dict[str, Any]) -> None:
        previous = state["uploaded"].get(row["path"])
        if previous and previous.get("git_sha") == row["git_sha"] and previous.get("sha256") == row["sha256"]:
            return
        data = Path(row["local_path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"File changed after planning: {row['path']}")
        try:
            content, encoding = data.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            content, encoding = base64.b64encode(data).decode("ascii"), "base64"
        response = github.call("create_blob", {"repository_full_name": args.repository,
                                               "content": content, "encoding": encoding})
        if response.get("sha") != row["git_sha"]:
            raise ValueError(f"Uploaded Git blob hash differs: {row['path']}")
        with lock:
            state["uploaded"][row["path"]] = {k: row[k] for k in ("git_sha", "sha256", "bytes")}
            write_checkpoint(checkpoint, state)
            print(json.dumps({"uploaded": row["path"], "bytes": row["bytes"]}), flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(upload, row) for row in files]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    if github.head(args.repository, args.branch) != current:
        raise ValueError("Branch advanced while uploading; blobs are checkpointed, review and rerun")
    if args.stage_only:
        state["status"] = "blobs_staged"
        write_checkpoint(checkpoint, state)
        return None
    parent = github.call("fetch", {
        "url": f"https://api.github.com/repos/{args.repository}/git/commits/{current}"
    })
    tree_sha = parent["tree"]["sha"]
    for start in range(0, len(files), TREE_BATCH_SIZE):
        tree = github.call("create_tree", {"repository_full_name": args.repository,
                                           "base_tree_sha": tree_sha,
                                           "tree_elements": [{"path": row["path"], "mode": "100644",
                                                              "type": "blob", "sha": row["git_sha"]}
                                                             for row in files[start:start + TREE_BATCH_SIZE]]})
        tree_sha = tree["sha"]
        print(json.dumps({"tree_entries_applied": min(start + TREE_BATCH_SIZE, len(files)),
                          "tree_entries_total": len(files)}), flush=True)
    state.update(status="tree_created", tree=tree_sha)
    write_checkpoint(checkpoint, state)
    commit = github.call("create_commit", {"repository_full_name": args.repository,
                                           "tree_sha": tree_sha, "parent_sha": current,
                                           "message": args.message})
    state.update(status="commit_created", commit=commit["sha"])
    write_checkpoint(checkpoint, state)
    if github.head(args.repository, args.branch) != current:
        raise ValueError("Branch advanced before publication; commit is checkpointed but not published")
    github.call("update_ref", {"repository_full_name": args.repository, "branch_name": args.branch,
                               "sha": commit["sha"], "force": False})
    if github.head(args.repository, args.branch) != commit["sha"]:
        raise ValueError("Branch verification failed; inspect the checkpoint and remote branch")
    state["status"] = "published"
    write_checkpoint(checkpoint, state)
    return commit["sha"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-json", type=Path, required=True)
    parser.add_argument("--repository", default="parthchvn/poly_world_cup")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--hosted-apps-helper", type=Path)
    parser.add_argument("--expected-head")
    parser.add_argument("--message", default="Publish prepared World Cup decision dataset")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--stage-only", action="store_true",
                        help="With --publish, upload blobs without creating a tree/commit or updating a branch")
    args = parser.parse_args()
    if args.stage_only and not args.publish:
        parser.error("--stage-only requires --publish and the same authorization arguments")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        parser.error("Repository must be owner/name")
    files = load_files(args.files_json.resolve())
    print(json.dumps({"files": len(files), "bytes": sum(row["bytes"] for row in files),
                      "max_blob_bytes": max(row["bytes"] for row in files),
                      "publish": args.publish, "stage_only": args.stage_only}), flush=True)
    if not args.publish:
        return
    if not args.checkpoint or not args.hosted_apps_helper or not args.expected_head:
        parser.error("--publish requires --checkpoint, --hosted-apps-helper and --expected-head")
    if not re.fullmatch(r"[0-9a-f]{40}", args.expected_head):
        parser.error("--expected-head must be a full commit SHA")
    commit = publish(files, args, GitHubTools(args.hosted_apps_helper))
    if commit is None:
        print(json.dumps({"status": "blobs_staged", "files": len(files),
                          "branch_unchanged": True}))
        return
    print(json.dumps({"status": "published", "commit": commit,
                      "url": f"https://github.com/{args.repository}/commit/{commit}"}))


if __name__ == "__main__":
    main()
