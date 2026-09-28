#!/usr/bin/env python3
"""Inspect a World Cup experiment locally and budget remaining data collection.

No network requests or model loading. Cached manifests describe observations,
not an unbiased sample of the wallets that remain. Scenario inputs are explicit
assumptions; neither an ETA nor historical-accounting completeness is promised.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics


ADDRESS = re.compile(r'0x[0-9a-fA-F]{40}')
SPLITS = ('train', 'validation', 'test')
GIB = 1024 ** 3


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def lines(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def timestamp_seconds(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('Target timestamps must include a timezone')
    return math.floor(stamp.timestamp())


def directory_size(path):
    """Logical file bytes, counting each path, without following symlinks."""
    size = count = 0
    if not path.is_dir() or path.is_symlink():
        return {'bytes': 0, 'files': 0, 'exists': False}
    for directory, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in files:
            item = Path(directory) / name
            if item.is_symlink():
                continue
            try:
                size += item.stat().st_size
                count += 1
            except FileNotFoundError:  # a collector may atomically replace a file
                continue
    return {'bytes': size, 'gib': round(size / GIB, 4), 'files': count, 'exists': True}


def target_cutoffs(prepared):
    """Use local audit filenames, never absolute source paths from another host."""
    cutoffs = {}
    audit = prepared / 'basic/source_audit.jsonl'
    if audit.is_file():
        for row in lines(audit):
            actor = Path(row['source_actor_file']).name.split('.', 1)[0].lower()
            if not ADDRESS.fullmatch(actor):
                raise ValueError(f'Invalid actor filename in {audit}')
            cutoff = timestamp_seconds(row['last_query_time'])
            cutoffs[actor] = max(cutoffs.get(actor, 0), cutoff)
        return cutoffs, 'basic/source_audit.jsonl'
    for path in sorted((prepared / 'selected_exports').glob('*/actors/*.jsonl*')):
        for row in lines(path):
            if row.get('record_type') != 'trade':
                continue
            actor = row['actor_id'].lower()
            if not ADDRESS.fullmatch(actor):
                raise ValueError(f'Invalid actor in {path}')
            cutoff = timestamp_seconds(row['timestamp'])
            cutoffs[actor] = max(cutoffs.get(actor, 0), cutoff)
    return cutoffs, 'selected_exports trade timestamps' if cutoffs else None


def dataset_state(path):
    manifest_path = path / 'manifest.json'
    if not manifest_path.is_file():
        return {'published_files_present': False, 'targets': None, 'path': str(path)}
    manifest = read_json(manifest_path)
    split_files = [sum((path / (split + suffix)).is_file()
                       for suffix in ('.jsonl', '.jsonl.gz')) == 1 for split in SPLITS]
    stats = manifest.get('stats', {})
    counts = [stats.get(split, {}).get('targets') for split in SPLITS]
    return {'published_files_present': all(split_files), 'path': str(path),
            'targets': sum(counts) if all(type(n) is int and n >= 0 for n in counts) else None,
            'split_targets': dict(zip(SPLITS, counts)),
            'verification': 'manifest and split files present; checksums not revalidated'}


def wallet_inventory(directory, cutoffs, warnings):
    captures, matches = [], {}
    for path in sorted(directory.glob('*/*/manifest.json')):
        try:
            state = read_json(path)
            parameters = state['parameters']
            pages = state['pages']
            if state.get('schema') != 'wallet_execution_capture_v1' or not isinstance(pages, list):
                raise ValueError('Unrecognized wallet manifest')
            page_bytes = 0
            for index, item in enumerate(pages):
                if item.get('file') not in (f'pages/{index:08d}.json', f'pages/{index:08d}.json.gz'):
                    raise ValueError('Unexpected page filename')
                page_path = path.parent / item['file']
                if page_path.is_symlink() or not page_path.is_file():
                    raise ValueError('Missing page file')
                page_bytes += page_path.stat().st_size
            if len(pages) != state.get('page_count'):
                raise ValueError('Page count differs from manifest')
            rows = state['row_count']
            if type(rows) is not int or rows < 0:
                raise ValueError('Invalid row count')
            status = state.get('api_traversal_status')
            if status not in ('paused', 'exhausted'):
                raise ValueError('Invalid traversal status')
            actor = parameters['user'].lower()
            record = {'pages': len(pages), 'rows': rows, 'page_bytes': page_bytes,
                      'status': status, 'actor': actor}
            captures.append(record)
            expected = {'user': actor, 'start': 1, 'end': cutoffs.get(actor),
                        'limit': 1000, 'taker_only': False,
                        'filter_type': 'TOKENS', 'filter_amount': '0.000001'}
            if actor in cutoffs and parameters == expected:
                if actor in matches:
                    raise ValueError('Duplicate capture for required wallet parameters')
                matches[actor] = record
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append(f'Could not count {path}: {error}')
    completed = sum(row['status'] == 'exhausted' for row in matches.values())
    page_counts = [row['pages'] for row in captures]
    return {'all_capture_count': len(captures),
            'all_captured_pages': sum(page_counts),
            'all_captured_executions': sum(row['rows'] for row in captures),
            'normalized_page_bytes': sum(row['page_bytes'] for row in captures),
            'required_wallets': len(cutoffs) if cutoffs else None,
            'matching_wallets_exhausted': completed,
            'matching_wallets_paused': len(matches) - completed,
            'not_started_wallets': len(cutoffs) - len(matches) if cutoffs else None,
            'remaining_wallets': len(cutoffs) - completed if cutoffs else None,
            'matching_captured_pages': sum(row['pages'] for row in matches.values()),
            'matching_captured_executions': sum(row['rows'] for row in matches.values()),
            'observed_pages_per_capture_median': statistics.median(page_counts) if page_counts else None,
            'observed_pages_per_capture_max': max(page_counts, default=None),
            'sample_warning': 'Alphabetical/prefix and partial captures are not a random sample; do not extrapolate their mean to remaining wallets.',
            'verification': 'Manifest parameters and page-file presence checked, not page checksums or historical completeness'}


def http_inventory(directory, warnings):
    times = []
    count = 0
    for path in (directory / 'requests').glob('*.json'):
        try:
            metadata = read_json(path)
            times.append(timestamp_seconds(metadata['retrieved_at']))
            count += 1
        except (OSError, ValueError, KeyError, TypeError) as error:
            warnings.append(f'Could not read request metadata {path}: {error}')
    return {'request_index_entries': count,
            'retrieval_timestamp_span_hours': round((max(times) - min(times)) / 3600, 4) if times else None,
            'request_duration_seconds': None,
            'timing_note': 'Retrieval timestamps are not request durations; their span may include pauses and multiple runs.'}


def number_list(value):
    try:
        values = [float(item) for item in value.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('Use comma-separated positive numbers') from error
    if not values or any(not math.isfinite(item) or item <= 0 for item in values):
        raise argparse.ArgumentTypeError('Use comma-separated positive numbers')
    return values


def estimate(args):
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Experiment root does not exist: {root}')
    for name in ('workers', 'min_interval', 'fallback_page_mib', 'output_bytes_per_target'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'--{name.replace("_", "-")} must be positive')
    if any(value < 1 for value in args.pages_per_wallet):
        raise ValueError('--pages-per-wallet must be at least 1 for every unresolved wallet')
    if args.elapsed_hours is not None and (not math.isfinite(args.elapsed_hours) or args.elapsed_hours <= 0):
        raise ValueError('--elapsed-hours must be positive')
    prepared = root / 'common/prepared'
    warnings = []
    cutoffs, cutoff_source = target_cutoffs(prepared)
    basic = dataset_state(prepared / 'basic')
    inmarket = dataset_state(root / 'inmarket/sft')
    global_state = dataset_state(root / 'global/sft')
    names = ('common', 'market_cache', 'wallet_cache', 'inmarket', 'global')
    sizes = {name: directory_size(root / name) for name in names}
    sizes['basic_sft'] = directory_size(prepared / 'basic')
    sizes['selected_exports'] = directory_size(prepared / 'selected_exports')
    sizes['wallet_http'] = directory_size(root / 'wallet_cache/http')
    sizes['wallet_histories'] = directory_size(root / 'wallet_cache/wallet_histories')
    total = directory_size(root)
    wallets = wallet_inventory(root / 'wallet_cache/wallet_histories', cutoffs, warnings)
    http = http_inventory(root / 'wallet_cache/http', warnings)
    unresolved = 0 if global_state['published_files_present'] else wallets['remaining_wallets']
    request_count = http['request_index_entries']
    page_count = wallets['all_captured_pages']
    http_per_page = sizes['wallet_http']['bytes'] / request_count if request_count else None
    normalized_per_page = sizes['wallet_histories']['bytes'] / page_count if page_count else None
    if http_per_page is not None and normalized_per_page is not None:
        bytes_per_page = http_per_page + normalized_per_page
        storage_basis = 'Observed cached HTTP bytes/request plus normalized capture bytes/page, including metadata; page fullness and compression can change.'
    else:
        bytes_per_page = args.fallback_page_mib * 1024 ** 2
        storage_basis = 'User-adjustable combined HTTP+normalized bytes/page assumption; no measured cache pair available.'
    targets = basic['targets']
    if inmarket['published_files_present'] and targets:
        output_allowance = sizes['inmarket']['bytes']
        output_basis = 'Uses complete In-market output size as a proxy for Global output; actual may differ.'
    elif targets is not None:
        output_allowance = sizes['basic_sft']['bytes'] + targets * args.output_bytes_per_target
        output_basis = 'Basic SFT bytes plus explicit additional output bytes/target assumption.'
    else:
        output_allowance = None
        output_basis = 'Unknown: no prepared target count is available.'
    scenarios = []
    if unresolved is not None:
        for pages in args.pages_per_wallet:
            requests = math.ceil(unresolved * pages)
            extra_cache = math.ceil(requests * bytes_per_page)
            extra_output = 0 if global_state['published_files_present'] else output_allowance
            for seconds in args.seconds_per_page:
                parallel_floor = max(requests * args.min_interval,
                                     requests * seconds / args.workers,
                                     pages * seconds if unresolved else 0)
                serial = requests * max(seconds, args.min_interval)
                scenarios.append({'assumed_additional_pages_per_remaining_wallet': pages,
                    'assumed_seconds_per_request': seconds, 'additional_requests': requests,
                    'ideal_parallel_network_hours': round(parallel_floor / 3600, 3),
                    'serial_network_hours': round(serial / 3600, 3),
                    'additional_cache_gib': round(extra_cache / GIB, 3),
                    'additional_output_gib': round(extra_output / GIB, 3) if extra_output is not None else None,
                    'projected_total_root_gib': round((total['bytes'] + extra_cache + extra_output) / GIB, 3) if extra_output is not None else None})
    global_state.update(pagination_reads_remaining_lower_bound=unresolved,
                        network_requests_remaining=0 if global_state['published_files_present'] else None,
                        local_output_allowance_bytes=output_allowance,
                        output_allowance_basis=output_basis,
                        duration_estimate='Scenarios only; the number of remaining pages is unknown.')
    basic.update(network_requests_remaining=0 if basic['published_files_present'] else None,
                 duration_estimate='Already prepared; no new fetch needed.' if basic['published_files_present'] else
                 'Unknown until required market captures and price-history coverage are measured.')
    inmarket.update(network_requests_remaining=0,
                    duration_estimate='Already prepared; no new fetch needed.' if inmarket['published_files_present'] else
                    'Offline computation only once Basic is ready; CPU/disk time is not measured here.')
    pace = None if args.elapsed_hours is None else {
        'elapsed_hours_supplied': args.elapsed_hours,
        'cached_exhausted_wallets_per_supplied_hour': round(wallets['matching_wallets_exhausted'] / args.elapsed_hours, 3),
        'note': 'Descriptive ratio only. Preexisting caches and unequal wallet sizes prevent treating this as an ETA.'}
    return {'format': 'actor_collection_estimate_v1', 'root': str(root), 'network_requests_made': 0,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'scope': 'Data collection and derived datasets only; excludes models, tokenized training caches, checkpoints and training time.',
            'variants': {'basic': basic, 'inmarket': inmarket, 'global': global_state},
            'target_cutoff_source': cutoff_source, 'wallets': wallets, 'http_cache': http,
            'storage': {'root_total': total, 'components': sizes,
                        'component_note': 'basic_sft/selected_exports overlap common; wallet_http/wallet_histories overlap wallet_cache. Do not sum overlapping components.',
                        'free_bytes_on_root_filesystem': shutil.disk_usage(root).free,
                        'units': 'Logical bytes; filesystem allocation, snapshots and temporary peak usage can be larger.'},
            'observed_pace': pace,
            'scenario_assumptions': {'workers': args.workers, 'global_min_interval_seconds': args.min_interval,
                'pages_per_wallet': args.pages_per_wallet, 'seconds_per_page': args.seconds_per_page,
                'combined_bytes_per_page': round(bytes_per_page), 'storage_basis': storage_basis,
                'fallback_page_mib': args.fallback_page_mib, 'output_bytes_per_target': args.output_bytes_per_target,
                'caveat': 'These are planning scenarios, not a confidence interval or an upper bound. Each assumed remaining page is treated as an HTTP cache miss; already cached but uncommitted responses reduce downloads. More pages, retries, throttling, slow disks, cache replay and feature computation can make time/storage larger. Ideal parallel assumes balanced independent wallets; pagination within each wallet stays sequential.'},
            'global_remaining_scenarios': scenarios, 'warnings': warnings}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Existing world_cup_40k_transfer directory')
    parser.add_argument('--out', type=Path, help='Also write this JSON report to a file')
    parser.add_argument('--summary', action='store_true', help='Print a short terminal summary; --out still saves full JSON')
    parser.add_argument('--elapsed-hours', type=float, help='Optional known elapsed time; descriptive pace only')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--min-interval', type=float, default=.25, help='Assumed shared minimum interval between request starts')
    parser.add_argument('--pages-per-wallet', type=number_list, default=[1., 5., 20.], help='Scenarios for ADDITIONAL pages per unresolved wallet')
    parser.add_argument('--seconds-per-page', type=number_list, default=[1., 3., 10.], help='Assumed wall seconds per request, excluding retries and CPU processing')
    parser.add_argument('--fallback-page-mib', type=float, default=2., help='Combined HTTP plus normalized cache MiB/page if no observed pair exists')
    parser.add_argument('--output-bytes-per-target', type=float, default=2000., help='Extra derived output bytes/target when no In-market proxy exists')
    return parser.parse_args(argv)


def summary(report):
    storage = report['storage']
    sizes = storage['components']
    text = ['Offline collection estimate', '']
    for name, component in (('basic', 'basic_sft'), ('inmarket', 'inmarket'), ('global', 'global')):
        state = report['variants'][name]
        status = 'ready' if state['published_files_present'] else 'not published'
        targets = f"{state['targets']:,}" if state['targets'] is not None else 'unknown'
        text.append(f"{name.capitalize():9} {status:14} targets={targets:>7}  current={sizes[component]['bytes'] / GIB:.3f} GiB")
    wallets = report['wallets']
    remaining = wallets['remaining_wallets']
    if remaining is None:
        text.append('Wallet cohort: unknown; prepare Basic first.')
    else:
        text.append(f"Wallets: {wallets['matching_wallets_exhausted']:,}/{wallets['required_wallets']:,} API traversals complete; "
                    f"{remaining:,} remaining ({wallets['matching_wallets_paused']:,} partly cached).")
    text.append(f"All current data/cache: {storage['root_total']['bytes'] / GIB:.3f} GiB; "
                f"filesystem free: {storage['free_bytes_on_root_filesystem'] / GIB:.1f} GiB.")
    if report['variants']['global']['published_files_present']:
        text.append('Global is already published; no further collection needed.')
    elif report['global_remaining_scenarios']:
        assumptions = report['scenario_assumptions']
        text.extend(['', f"Global scenarios: {assumptions['workers']} workers, shared minimum request interval {assumptions['global_min_interval_seconds']}s.",
                     'Assumptions, not ETAs: pages below are ADDITIONAL pages per unresolved wallet.',
                     'Ideal parallel hours exclude retries, cache replay and feature computation; actual time may be longer.',
                     '', 'Pages/wallet  Sec/request  Requests  Ideal hours  Extra cache GiB'])
        for row in report['global_remaining_scenarios']:
            text.append(f"{row['assumed_additional_pages_per_remaining_wallet']:12g}  "
                        f"{row['assumed_seconds_per_request']:11g}  "
                        f"{row['additional_requests']:8,}  {row['ideal_parallel_network_hours']:11.2f}  "
                        f"{row['additional_cache_gib']:15.3f}")
        allowance = report['variants']['global']['local_output_allowance_bytes']
        text.append(f"Additional Global output allowance: {allowance / GIB:.3f} GiB." if allowance is not None
                    else 'Additional Global output size: unknown.')
        text.append('Storage uses observed cache bytes/page when available; page sizes and final output may differ.')
    text.extend(['', 'When Basic is ready, no further fetch is needed. In-market adds no API fetches.',
                 'Excludes model weights, tokenized training caches, checkpoints and training time.'])
    if report['warnings']:
        text.append(f"Warnings: {len(report['warnings'])}; inspect the full JSON report before relying on coverage counts.")
    return '\n'.join(text) + '\n'


def main(argv=None):
    args = parse_args(argv)
    try:
        report = estimate(args)
        text = json.dumps(report, indent=2, allow_nan=False) + '\n'
        if args.out:
            args.out.expanduser().write_text(text, encoding='utf-8')
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f'Error: {error}') from error
    print(summary(report) if args.summary else text, end='')


if __name__ == '__main__':
    main()
