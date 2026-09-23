#!/usr/bin/env python3
"""Package an audited observation corpus without redistributing news bodies.

Default output is a portable SQLite corpus plus bibliographic news metadata.
Raw trade provenance is optional and always written to a separate archive.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.attribution import read_trade_context
from poly_world_cup.io import write_json, write_jsonl

FORBIDDEN_NEWS_FIELDS = {"body", "story", "html", "raw_html", "article_body", "full_text", "description",
                         "text", "content", "bodyhtml", "bodytext", "storybody", "articlebody"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def metadata_only(value, location="news") -> None:
    """Fail closed if a bibliographic export has acquired article body fields."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in FORBIDDEN_NEWS_FIELDS and item not in (None, "", [], {}):
                raise ValueError(f"News body field is prohibited: {location}.{key}")
            metadata_only(item, location + "." + key)
    elif isinstance(value, list):
        for item in value:
            metadata_only(item, location)


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                metadata_only(row, f"{path.name}:{number}")
                rows.append(row)
    return rows


def _safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Unsafe manifest member path")
    return path


def _regular(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular input file: {path}")
    return path


def _manifests(registry: dict, trades_root: Path) -> list[tuple[Path, dict]]:
    conditions = sorted({row["condition_id"].lower() for row in registry["contracts"]})
    if not conditions:
        raise ValueError("Registry has no contracts")
    result = []
    for condition in conditions:
        if len(condition) != 66 or not condition.startswith("0x") or any(c not in "0123456789abcdef" for c in condition[2:]):
            raise ValueError("Malformed registry condition ID")
        path = _regular(trades_root / condition / "manifest.json")
        item = json.loads(path.read_text())
        if item.get("condition_id") != condition or item.get("api_traversal_status") != "exhausted":
            raise ValueError(f"Release requires exhausted traversal for {condition}")
        pages = item.get("pages", [])
        if item.get("page_count") != len(pages) or item.get("row_count") != sum(p["row_count"] for p in pages):
            raise ValueError("Manifest page/row totals disagree")
        if len({page["file"] for page in pages}) != len(pages):
            raise ValueError("Manifest repeats a normalized page")
        for page in pages:
            relative = _safe_relative(page["file"])
            _regular(path.parent / relative)
        result.append((path, item))
    return result


def _readonly(path: Path) -> sqlite3.Connection:
    _regular(path)
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def _verify_database(db: sqlite3.Connection, manifests, news_records: dict[str, dict]) -> dict:
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    required = {"trades", "news", "fixture_news", "global_news", "source_pages", "metadata", "context_states"}
    if not required.issubset(tables):
        raise ValueError("SQLite is missing corpus tables")
    actual_news = set()
    for identity, serialized in db.execute("SELECT news_id,record_json FROM news"):
        record = json.loads(serialized)
        metadata_only(record, "SQLite.news")
        if news_records.get(identity) != record:
            raise ValueError("SQLite news content differs from supplied metadata catalogs")
        actual_news.add(identity)
    if actual_news != set(news_records):
        raise ValueError("SQLite news IDs do not match supplied metadata catalogs")
    expected_pages = {}
    for manifest_path, manifest in manifests:
        for page in manifest["pages"]:
            identity = str((manifest_path.parent / page["file"]).resolve())
            if identity in expected_pages:
                raise ValueError("Multiple manifests reference the same source page")
            expected_pages[identity] = (page["normalized_sha256"], page["row_count"])
    actual_pages = {path: (sha, count) for path, sha, count in db.execute("SELECT path,uncompressed_sha256,row_count FROM source_pages")}
    if actual_pages != expected_pages:
        raise ValueError("SQLite source pages do not match every committed manifest page")
    count = db.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    if count != sum(value[1] for value in expected_pages.values()):
        raise ValueError("SQLite does not contain every manifested trade observation")
    check = db.execute("PRAGMA quick_check").fetchone()[0]
    if check != "ok":
        raise ValueError("SQLite integrity check failed: " + str(check))
    report_row = db.execute("SELECT value_json FROM metadata WHERE key='report'").fetchone()
    report = json.loads(report_row[0]) if report_row else {}
    return {"trade_observations": count, "news_versions": len(actual_news), "source_pages": len(actual_pages),
            "conditions": len(manifests), "database_report": report,
            "source_completeness_certified": False, "training_ready": False}


def _sample_contexts(database: Path, limit: int) -> tuple[list[dict], dict]:
    if limit < 0:
        raise ValueError("Sample fixture count cannot be negative")
    with _readonly(database) as db:
        fixtures = [row[0] for row in db.execute("SELECT DISTINCT fixture_id FROM trades WHERE fixture_id IS NOT NULL ORDER BY fixture_id")]
        take = min(limit, len(fixtures))
        indices = sorted({round(i * (len(fixtures) - 1) / (take - 1)) for i in range(take)}) if take > 1 else [0] if take else []
        chosen, samples = [fixtures[i] for i in indices], []
        for fixture in chosen:
            bounds = db.execute("SELECT MIN(query_us),MAX(query_us) FROM trades WHERE fixture_id=?", (fixture,)).fetchone()
            queries = [
                ("first_execution", "SELECT trade_row_id FROM trades WHERE fixture_id=? ORDER BY query_us,trade_row_id LIMIT 1", (fixture,)),
                ("last_execution", "SELECT trade_row_id FROM trades WHERE fixture_id=? ORDER BY query_us DESC,trade_row_id DESC LIMIT 1", (fixture,)),
                ("at_or_after_time_midpoint", "SELECT trade_row_id FROM trades WHERE fixture_id=? AND query_us>=? ORDER BY query_us,trade_row_id LIMIT 1", (fixture, (bounds[0]+bounds[1])//2)),
            ]
            selected = {}
            for label, sql, params in queries:
                row = db.execute(sql, params).fetchone()
                if row:
                    selected.setdefault(row[0], []).append(label)
            for identity, labels in sorted(selected.items()):
                context = read_trade_context(database, identity, retrospective_limit=3, wallet_history_limit=3, global_context_limit=3)
                samples.append({"sampling_strata": labels, "context": context})
    return samples, {"method": "evenly_spaced_fixture_ids_then_first_last_and_time_midpoint_execution",
                     "fixtures": chosen, "sample_count": len(samples), "statistically_representative": False,
                     "training_examples": False, "display_limits": {"retrospective_news": 3, "wallet_history": 3, "global_news": 3}}


def _archive(root: Path, output: Path, *, compression_level=3) -> dict:
    if output.exists():
        raise ValueError(f"Refusing to overwrite existing release archive: {output}")
    temporary = output.with_name("." + output.name + ".tmp")
    try:
        with temporary.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=compression_level, mtime=0, filename="") as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for path in sorted(root.rglob("*")):
                    if not path.is_file():
                        continue
                    _regular(path)
                    info = archive.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ""
                    info.mode = 0o644
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"file": output.name, "bytes": output.stat().st_size, "sha256": digest(output)}


def _manifest(root: Path, facts: dict) -> dict:
    members = [{"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)}
               for path in sorted(root.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"]
    value = {"format_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
             **facts, "members": members, "manifest_self_hash_included": False}
    write_json(root / "MANIFEST.json", value)
    return value


def _raw_members(manifests, cache: Path, layout: str) -> list[tuple[Path, str]]:
    members = {}
    for path, manifest in manifests:
        condition = manifest["condition_id"]
        members[f"trades/{condition}/manifest.json"] = path
        condition_cache = cache / condition if layout == "per_condition" else cache
        archive_cache = "cache/" + condition if layout == "per_condition" else "cache"
        wanted_captures = {(p["body_sha256"], p["request_url"]) for p in manifest["pages"]}
        for capture in sorted((condition_cache / "captures").glob("*.json")):
            data = json.loads(capture.read_text())
            if (data.get("body_sha256"), data.get("url")) in wanted_captures:
                members[archive_cache + "/captures/" + capture.name] = _regular(capture)
        for page in manifest["pages"]:
            if not page.get("request_url", "").startswith("https://data-api.polymarket.com/v2/trades?"):
                raise ValueError("Raw export accepts only Polymarket trade responses")
            relative = _safe_relative(page["file"])
            members[f"trades/{condition}/{relative}"] = _regular(path.parent / relative)
            body_hash = page["body_sha256"]
            if len(body_hash) != 64 or any(x not in "0123456789abcdef" for x in body_hash):
                raise ValueError("Malformed trade body digest")
            body = condition_cache / "bodies" / (body_hash + ".json.gz")
            if not body.exists():
                body = condition_cache / "bodies" / (body_hash + ".json")
            members[archive_cache + "/bodies/" + body.name] = _regular(body)
    return [(path, name) for name, path in sorted(members.items())]


def export_corpus(*, database: Path, registry: Path, news: Path, archived_news: Iterable[Path],
                  trades_root: Path, output_dir: Path, reports: Iterable[Path] = (),
                  include_raw_trades: bool = False, trade_cache: Path | None = None,
                  sample_fixtures: int = 12, dry_run: bool = False,
                  cache_layout: str = "per_condition", immutable_database: bool = False,
                  provenance_report: Path | None = None) -> dict:
    database, registry, news, trades_root, output_dir = map(Path, (database, registry, news, trades_root, output_dir))
    archived_news, reports = list(map(Path, archived_news)), list(map(Path, reports))
    provenance = None
    if provenance_report is not None:
        provenance_report = _regular(Path(provenance_report))
        provenance = json.loads(provenance_report.read_text())
        if provenance.get("raw_provenance_verified") is not True:
            raise ValueError("Raw provenance report did not verify the collection")
        reports.append(provenance_report)
    for path in [database, registry, news, *archived_news, *reports]:
        _regular(path)
    for path in reports:
        if path.suffix != ".json":
            raise ValueError("Release reports must be JSON audit metadata")
        metadata_only(json.loads(path.read_text()), "report:" + path.name)
    manifests = _manifests(json.loads(registry.read_text()), trades_root)
    if include_raw_trades and trade_cache is None:
        raise ValueError("--raw-trades requires --trade-cache")
    if cache_layout not in {"per_condition", "shared"}:
        raise ValueError("Unknown trade cache layout")
    raw_members = _raw_members(manifests, Path(trade_cache), cache_layout) if include_raw_trades else []
    if immutable_database and Path(str(database) + "-wal").exists() and Path(str(database) + "-wal").stat().st_size:
        raise ValueError("Immutable database export requires a checkpointed, closed database with no WAL contents")
    wal_size = Path(str(database) + "-wal").stat().st_size if Path(str(database) + "-wal").exists() else 0
    inputs_size = sum(path.stat().st_size for path in [database, registry, news, *archived_news, *reports]) + sum(path.stat().st_size for path, _ in manifests) + wal_size
    raw_size = sum(path.stat().st_size for path, _ in raw_members)
    # Snapshot plus worst-case compressed output (compression can slightly grow
    # an input), modest samples, and separate raw staging/archives. Conservative.
    estimate = {"normalized_input_bytes": inputs_size, "raw_trade_input_bytes": raw_size,
                "estimated_extra_disk_required_bytes": 2 * inputs_size - (database.stat().st_size if immutable_database else 0) + 2 * raw_size + 256 * 1024**2,
                "compression_ratio_assumed": None, "raw_trade_export_requested": include_raw_trades,
                "immutable_database_requested": immutable_database}
    if dry_run:
        return {"dry_run": True, **estimate}
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_dir).free < estimate["estimated_extra_disk_required_bytes"]:
        raise ValueError("Insufficient free space for a conservative snapshot/archive estimate; run --dry-run")
    if any((output_dir / name).exists() for name in ["world_cup_corpus.tar.gz", "trade_provenance.tar.gz", "release_manifest.json"]):
        raise ValueError("Use a fresh output directory; existing release files are never overwritten")
    artifacts = []
    with tempfile.TemporaryDirectory(prefix=".corpus-export-", dir=output_dir) as tmp:
        stage = Path(tmp) / "corpus"
        stage.mkdir()
        def copy(source: Path, target: str):
            destination = stage / target
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            return destination
        copy(registry, "registry.json")
        catalogs = [copy(news, "news/news.jsonl")]
        for index, path in enumerate(archived_news):
            catalogs.append(copy(path, f"news/archived_headlines_{index:02d}.jsonl"))
        catalog_records = {}
        for path in catalogs:
            for row in _read_jsonl(path):
                if row["news_id"] in catalog_records:
                    raise ValueError("Duplicate news IDs across release catalogs")
                catalog_records[row["news_id"]] = row
        if immutable_database:
            try:
                (stage / "attribution.sqlite").hardlink_to(database.resolve())
            except OSError as error:
                raise ValueError("Immutable export requires output on the database filesystem") from error
        else:
            with _readonly(database) as source, sqlite3.connect(stage / "attribution.sqlite") as destination:
                source.backup(destination, pages=16384)
        immutable_stat = (database.stat().st_size, database.stat().st_mtime_ns)
        with _readonly(stage / "attribution.sqlite") as snapshot:
            facts = _verify_database(snapshot, manifests, catalog_records)
            schema = ";\n".join(row[0] for row in snapshot.execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY type,name")) + ";\n"
        facts["raw_provenance_checked"] = provenance is not None
        if provenance is not None:
            if (provenance.get("verified_observation_count") != facts["trade_observations"]
                    or provenance.get("verified_condition_count", provenance.get("condition_count")) != facts["conditions"]
                    or provenance.get("verified_raw_page_count", provenance.get("raw_page_count")) != facts["source_pages"]):
                raise ValueError("Provenance report counts do not match the exported corpus")
        (stage / "schema.sql").write_text(schema)
        for path, manifest in manifests:
            copy(path, "trade_manifests/" + manifest["condition_id"] + ".json")
        for index, path in enumerate(reports):
            copy(path, f"reports/{index:02d}_{path.name}")
        samples, sampling = _sample_contexts(stage / "attribution.sqlite", sample_fixtures)
        write_jsonl(stage / "context_samples.jsonl", samples)
        write_json(stage / "sample_method.json", sampling)
        (stage / "README.md").write_text(_README)
        _manifest(stage, {"kind": "normalized_observation_corpus", "counts": facts,
                          "news_full_text_included": False, "raw_trade_provenance_is_separate": include_raw_trades})
        corpus_artifact = _archive(stage, output_dir / "world_cup_corpus.tar.gz")
        if immutable_database and ((database.stat().st_size, database.stat().st_mtime_ns) != immutable_stat
                or (Path(str(database) + "-wal").exists() and Path(str(database) + "-wal").stat().st_size)):
            (output_dir / "world_cup_corpus.tar.gz").unlink()
            raise ValueError("Immutable database changed during export; the incomplete archive was removed")
        artifacts.append(corpus_artifact)
        if include_raw_trades:
            raw_stage = Path(tmp) / "trade_provenance"
            raw_stage.mkdir()
            for source, member in raw_members:
                target = raw_stage / member
                target.parent.mkdir(parents=True, exist_ok=True)
                # Hardlinks avoid another large copy; if unsupported, copy.
                try:
                    target.hardlink_to(source.resolve())
                except OSError:
                    shutil.copyfile(source, target)
            (raw_stage / "README.md").write_text("# Trade provenance\n\nOnly committed manifest-referenced normalized observations and original Polymarket trade response bodies are included. No news responses or HTML are included. Stored response SHA256 values refer to uncompressed bytes; archive member hashes refer to the actual stored bytes. API exhaustion is not a certificate of complete on-chain history.\n")
            _manifest(raw_stage, {"kind": "trade_provenance", "news_files_included": False})
            artifacts.append(_archive(raw_stage, output_dir / "trade_provenance.tar.gz"))
        summary = {"format_version": 1, "artifacts": artifacts, "counts": facts, "disk_estimate": estimate,
                   "news_full_text_included": False, "training_ready": False}
        write_json(output_dir / "release_manifest.json", summary)
        return summary


_README = """# World Cup observation corpus

Open `attribution.sqlite` with SQLite, Python's standard sqlite3 module, or a
SQLite database browser. The database contains every normalized trade observation
listed by the included exhausted API collection manifests, along with news
metadata and context links. It does not require the original machine's files.

Tables: trades, news, fixture_news, global_news, context_states, source_pages,
metadata. See schema.sql for exact column names. Prices/shares/token IDs are TEXT
so decimal precision and token identity survive; SQL CAST(... AS REAL) is only
for approximate exploration. query_us is an execution block-time proxy in UTC
microseconds, not an observed order-decision timestamp.

Example SQL:

    SELECT fixture_id, COUNT(*) AS observations
    FROM trades GROUP BY fixture_id ORDER BY observations DESC;

    SELECT trade_row_id, wallet, fixture_id, side, shares, price, block_timestamp
    FROM trades ORDER BY trade_row_id LIMIT 20;

    SELECT news_id, record_json FROM news LIMIT 5;

From a checkout of github.com/parthchvn/poly_world_cup, run:

    python -m poly_world_cup inspect-context --database /path/to/attribution.sqlite --row 1

context_samples.jsonl contains small stratified browsing examples, not a
statistically representative sample or SFT training data. News JSONL files retain
headlines, source URLs, timestamps, evidence descriptors, and linkage metadata;
article text and raw archived HTML are excluded. Current headlines with old
publication dates are not historically verified. Archived headlines only become
available at the independently recorded capture upper bound, and broad World Cup
relevance does not establish direct fixture relevance or actor exposure.

MANIFEST.json lists the SHA256 and size of every other member. Optional raw trade
provenance is distributed in a separate archive. source_pages.path retains the
original collection path for provenance; a relocated SQLite file still works.
Raw archive normalized pages can be found under trades/CONDITION_ID/pages/.

API observations are not certified complete exchange logs. Wallet histories
cover only these tournament contracts; holdings, external activity, and actual
private beliefs are not reconstructed. NO_TRADE labels cannot be inferred from
missing observations. This corpus is not yet ready for SFT.
"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("data/full/attribution.sqlite"))
    parser.add_argument("--registry", type=Path, default=Path("data/registry/registry.json"))
    parser.add_argument("--news", type=Path, default=Path("data/news/news.jsonl"))
    parser.add_argument("--archive-news", type=Path, action="append", default=[])
    parser.add_argument("--trades-root", type=Path, default=Path("data/full/trades"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--raw-trades", action="store_true")
    parser.add_argument("--immutable-database", action="store_true", help="Avoid a database copy; requires a closed immutable DB on the output filesystem")
    parser.add_argument("--provenance-report", type=Path)
    parser.add_argument("--trade-cache", type=Path, default=Path("data/full/cache"))
    parser.add_argument("--sample-fixtures", type=int, default=12)
    parser.add_argument("--cache-layout", choices=["per_condition", "shared"], default="per_condition")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = export_corpus(database=args.database, registry=args.registry, news=args.news,
            archived_news=args.archive_news, trades_root=args.trades_root, output_dir=args.output,
            reports=args.report, include_raw_trades=args.raw_trades, trade_cache=args.trade_cache,
            sample_fixtures=args.sample_fixtures, dry_run=args.dry_run, cache_layout=args.cache_layout,
            immutable_database=args.immutable_database, provenance_report=args.provenance_report)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, OSError, sqlite3.Error, KeyError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
