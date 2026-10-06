#!/usr/bin/env python3
"""Package an exact raw-row budget using whole actor/market histories, offline.

These are raw interval/trade rows, NOT scheduled five-minute SFT examples.
The ZIP preserves the shared inputs needed to build those examples on RunPod.
Python 3.11+, standard library only; Windows, macOS and Linux.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import tempfile
import zipfile

import derive_actor_metrics as metrics

require = metrics.require


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_file(root, relative):
    path = root / relative
    require(not path.is_symlink() and path.is_file()
            and path.resolve().is_relative_to(root.resolve()), f'Missing/unsafe file: {path}')
    return path


def choose(candidates, target, seed):
    require(type(target) is int and target > 0 and target % 2 == 0,
            'Target raw rows must be a positive even integer (complete interval/trade pairs).')
    candidates = list(candidates)
    random.Random(seed).shuffle(candidates)
    target //= 2
    mask, reachable, parents = (1 << (target + 1)) - 1, 1, {}
    for index, item in enumerate(candidates):
        weight = item['entry']['rows'] // 2
        fresh = ((reachable << weight) & mask) & ~reachable
        reachable |= fresh
        while fresh:
            bit = fresh & -fresh
            total = bit.bit_length() - 1
            parents[total] = (total - weight, index)
            fresh ^= bit
        if (reachable >> target) & 1:
            break
    require((reachable >> target) & 1,
            'Cannot reach the exact row target with whole eligible histories; no ZIP created.')
    selected = []
    while target:
        target, index = parents[target]
        selected.append(candidates[index])
    return sorted(selected, key=lambda item: (item['market_id'], item['entry']['actor_id']))


def package(input_root, output, target=200000, seed=42, max_trades=20):
    require(type(target) is int and target > 0 and target % 2 == 0, 'Target must be positive and even.')
    require(type(max_trades) is int and max_trades > 0, 'Trade cap must be positive.')
    input_root, output = Path(input_root).resolve(), Path(output).absolute()
    require(not output.exists(), f'Output already exists: {output}; choose a new --out.')
    sources = metrics.discover_exports([], input_root)
    candidates = []
    for source in sources:
        directory, manifest = source['path'], source['manifest']
        require(manifest.get('market_context_version') == 2, f'Official price history missing: {directory}')
        require(manifest.get('source', {}).get('api_traversal_status') == 'exhausted',
                f'Finish the API capture before packaging: {directory}')
        source_cap = manifest.get('max_trades_per_actor')
        require(source_cap is None or source_cap >= max_trades,
                f'Export already removed actors below the requested trade cap: {directory}')
        for name in ('market.json', 'espn_events.jsonl', 'market_price_history.jsonl'):
            safe_file(directory, name)
        expected_price_sha = manifest.get('market_price_history', {}).get('sha256')
        require(expected_price_sha and digest(directory / 'market_price_history.jsonl') == expected_price_sha,
                f'Price-history checksum mismatch: {directory}')
        source['manifest_sha256'] = digest(directory / 'manifest.json')
        index_path = safe_file(directory, 'actor_index.jsonl')
        source['index_sha256'] = digest(index_path)
        counts, seen, paths = Counter(), set(), set()
        for entry in metrics.iter_jsonl(index_path):
            actor = entry['actor_id']
            require(isinstance(actor, str) and metrics.ADDRESS.fullmatch(actor), 'Invalid actor ID in index')
            require(actor.lower() not in seen, f'Duplicate actor in index: {directory}/{actor}')
            seen.add(actor.lower())
            path = safe_file(directory, entry['path'])
            require(path.parent == directory / 'actors' and path.name in (actor + '.jsonl', actor + '.jsonl.gz'),
                    f'Invalid actor path in index: {path}')
            require(path not in paths, f'Duplicate actor path: {path}')
            paths.add(path)
            for key in ('rows', 'distinct_trade_times', 'trade_observations'):
                require(type(entry.get(key)) is int and entry[key] > 0, f'Invalid {key}: {path}')
                counts[key] += entry[key]
            counts['actors'] += 1
            require(entry['rows'] == 2 * entry['distinct_trade_times']
                    and entry['distinct_trade_times'] <= entry['trade_observations'], f'Invalid pair counts: {path}')
            if entry['trade_observations'] <= max_trades:
                candidates.append(dict(market_id=source['market_id'], source=source, entry=entry, path=path))
        for key in ('actors', 'rows', 'distinct_trade_times', 'trade_observations'):
            require(counts[key] == manifest['counts'].get(key, 0), f'Index/manifest {key} mismatch: {directory}')
        require(paths == set((directory / 'actors').glob('*.jsonl*')), f'Actor index is incomplete: {directory}')
        print(f"Scanned market {source['market_id']}: {counts['actors']:,} histories", flush=True)

    print(f"Eligible: {len(candidates):,} histories / {sum(x['entry']['rows'] for x in candidates):,} raw rows", flush=True)
    selected = choose(candidates, target, seed)
    grouped = defaultdict(list)
    for item in selected:
        grouped[item['market_id']].append(item)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Same-filesystem temporary archive; failures never publish a partial ZIP.
    with tempfile.TemporaryDirectory(prefix='actor-package-', dir=output.parent) as temp:
        partial = Path(temp) / 'package.zip'
        inventory, written = [], 0
        with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            def write_json(name, value):
                archive.writestr(name, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

            for market_id, items in grouped.items():
                source = items[0]['source']
                directory = source['path']
                prefix = f'exports/market_{market_id}'
                counts, coverage, entries, references = Counter(), Counter(), [], set()
                for item in items:
                    path, entry = item['path'], item['entry']
                    fingerprint = digest(path)
                    actor, groups, actual = metrics.actor_trade_groups(path, source)
                    require(actor == entry['actor_id'].lower(), f'Actor mismatch: {path}')
                    require(all(actual[key] == entry[key] for key in ('rows', 'distinct_trade_times', 'trade_observations')),
                            f'Actor index differs from file: {path}')
                    require(1 <= actual['trade_observations'] <= max_trades, f'Actor exceeds trade cap: {path}')
                    rows = list(metrics.iter_jsonl(path))
                    require(not any(row.get('in_market_pnl_opening_history') for row in rows),
                            f'Export has truncated opening history: {path}; use the full actor export.')
                    actual['news_entries'] = sum(len(row.get('news', [])) for row in rows)
                    for row in rows:
                        if row.get('actor_snapshot_ref'):
                            references.add(row['actor_snapshot_ref'])
                        if row['record_type'] == 'trade' and row.get('market_context'):
                            context = row['market_context']
                            coverage['trade_rows'] += 1
                            coverage['both_outcomes_available'] += int(all(context[o] is not None for o in ('yes', 'no')))
                            for outcome in ('yes', 'no'):
                                available = context[outcome] is not None
                                coverage[outcome + ('_available' if available else '_missing')] += 1
                                if not available:
                                    coverage[outcome + '_' + context['missing_reasons'][outcome]] += 1
                    raw = path.read_bytes()
                    require(hashlib.sha256(raw).hexdigest() == fingerprint, f'Actor changed while packaging: {path}')
                    relative = 'actors/' + path.name
                    archive.writestr(f'{prefix}/{relative}', raw,
                                     compress_type=zipfile.ZIP_STORED if path.suffix == '.gz' else zipfile.ZIP_DEFLATED)
                    counts.update(actual)
                    entries.append({**entry, 'path': relative, **{k: v for k, v in actual.items() if k != 'actors'}})
                    inventory.append({'market_id': market_id, 'actor_id': actor, 'path': f'{prefix}/{relative}',
                                      'rows': actual['rows'], 'trade_observations': actual['trade_observations'], 'sha256': fingerprint})
                    written += actual['rows']
                for ref in references:
                    snapshot = safe_file(directory, ref)
                    require(snapshot.parent == directory / 'actor_snapshots', f'Invalid snapshot reference: {ref}')
                    archive.write(snapshot, f'{prefix}/actor_snapshots/{snapshot.name}')
                # Preserve news, price provenance, and other shared export files byte-for-byte.
                for path in sorted(directory.iterdir()):
                    if path.name not in ('manifest.json', 'actor_index.jsonl', 'source_manifest.json') and path.is_file():
                        safe_file(directory, path.name)
                        archive.write(path, f'{prefix}/{path.name}')
                require(digest(directory / 'manifest.json') == source['manifest_sha256']
                        and digest(directory / 'actor_index.jsonl') == source['index_sha256'],
                        f'Source changed while packaging: {directory}')
                archive.write(directory / 'manifest.json', f'{prefix}/source_manifest.json')
                # The validator reads floats as Decimal. Reload the original
                # JSON types for metadata serialization; actor bytes stay exact.
                manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
                manifest['counts'] = dict(counts)
                manifest['market_price_coverage'] = dict(coverage)
                manifest['price_context_complete_for_exported_rows'] = coverage['both_outcomes_available'] == coverage['trade_rows']
                manifest['experiment_selection'] = {
                    'whole_actor_conversations': True, 'seed': seed, 'target_raw_rows': target,
                    'max_captured_executions_per_actor_market': max_trades,
                    'original_manifest_sha256': source['manifest_sha256'],
                    'original_full_export_counts': source['manifest']['counts'],
                    'original_source_capture_statistics_retained': True,
                }
                write_json(f'{prefix}/manifest.json', manifest)
                archive.writestr(f'{prefix}/actor_index.jsonl', ''.join(json.dumps(e) + '\n' for e in entries))
                print(f'Packaged {written:,}/{target:,} raw rows', flush=True)
            require(written == target, 'Internal selected-row count mismatch')
            receipt = {'format': 'whole_actor_market_subset_v2', 'seed': seed, 'rows': written,
                       'row_unit': 'raw interval/trade record, not scheduled prediction window',
                       'actor_market_histories': len(selected), 'markets': len(grouped),
                       'max_captured_executions_per_actor_market': max_trades,
                       'includes_no_trade_rows': True, 'groups': inventory}
            write_json('selection.json', receipt)
        with zipfile.ZipFile(partial) as archive:
            require(archive.testzip() is None, 'ZIP integrity check failed')
        require(not output.exists(), f'Output appeared during packaging: {output}')
        partial.rename(output)
    print(f'READY TO UPLOAD: {output}\nRaw actor rows: {written:,}; size: {output.stat().st_size / 1024**2:,.1f} MiB', flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path.home() / 'wc_collection_v1' / 'exports')
    parser.add_argument('--out', type=Path, default=Path.home() / 'wc_collection_v1' / 'wc_200k_by_actor_v2.zip')
    parser.add_argument('--target-rows', type=int, default=200000)
    parser.add_argument('--max-trades-per-actor', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    package(args.input_root, args.out, args.target_rows, args.seed, args.max_trades_per_actor)


if __name__ == '__main__':
    main()
