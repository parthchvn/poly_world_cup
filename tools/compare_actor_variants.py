#!/usr/bin/env python3
"""Check a fair basic/inmarket/global SFT comparison, or purge split overlap.

python3 tools/compare_actor_variants.py --basic BASE --inmarket LOCAL --global GLOBAL
python3 tools/compare_actor_variants.py purge --input BASE --out PURGED_BASE

Character counts describe stored system/user prompts; they are not tokenizer
counts and do not include repetition of prior conversation turns by training.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import sys
import tempfile


_SPEC = importlib.util.spec_from_file_location(
    '_actor_variant_metric_io', Path(__file__).resolve().parents[1] / 'scripts/derive_actor_metrics.py')
_METRICS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _METRICS
_SPEC.loader.exec_module(_METRICS)
require, loads, timestamp_us = _METRICS.require, _METRICS.loads, _METRICS.timestamp_us
read_json, write_json = _METRICS.read_json, _METRICS.write_json
SPLITS = ('train', 'validation', 'test')
SCOPES = {'inmarket': 'actor_and_binary_market', 'global': 'actor_across_all_markets'}


def split_file(root, split):
    paths = [root / (split + suffix) for suffix in ('.jsonl', '.jsonl.gz')]
    paths = [path for path in paths if path.exists() or path.is_symlink()]
    require(len(paths) == 1, f'{root}: {split} needs exactly one .jsonl or .jsonl.gz file')
    require(paths[0].is_file() and not paths[0].is_symlink(), f'Unsafe split file: {paths[0]}')
    return paths[0]


def lines(path):
    opener = gzip.open if path.name.endswith('.gz') else open
    with opener(path, 'rb') as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f'{path}:{number}: blank JSONL row')
            try:
                record = loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError(f'{path}:{number}: {error}') from error
            require(isinstance(record, dict), f'{path}:{number}: expected a JSON object')
            yield line, record


def conversation(record):
    """Validate one conversation and return query instants and execution count."""
    for key in ('sequence_id', 'actor_id', 'market_id', 'fixture_id'):
        require(isinstance(record.get(key), str) and record[key], f'Missing conversation {key}')
    messages = record.get('messages')
    require(isinstance(messages, list) and len(messages) >= 3 and len(messages) % 2 == 1,
            'Expected a system message followed by user/assistant pairs')
    require(all(isinstance(message, dict) and isinstance(message.get('content'), str)
                for message in messages), 'Invalid conversation messages')
    require(messages[0].get('role') == 'system', 'First conversation message must be system')
    queries, executions = [], 0
    for offset in range(1, len(messages), 2):
        user, assistant = messages[offset:offset + 2]
        require(user.get('role') == 'user' and assistant.get('role') == 'assistant',
                'Conversation roles must alternate user/assistant')
        context, label = loads(user['content']), loads(assistant['content'])
        require(isinstance(context, dict), 'User context must be a JSON object')
        if 'actor_id' in context:
            require(context.get('actor_id') == record['actor_id'], 'User actor_id differs from conversation')
        if 'market' in context:
            require(isinstance(context['market'], dict), 'User market must be an object')
            if 'market_id' in context['market']:
                require(context['market']['market_id'] == record['market_id'],
                        'User market_id differs from conversation')
        query = timestamp_us(context.get('query_time'))
        require(not queries or query > queries[-1], 'Conversation queries must strictly increase')
        require(isinstance(label, dict) and label.get('action') == 'TRADE'
                and isinstance(label.get('trades'), list) and label['trades'],
                'Expected a nonempty TRADE execution target')
        require(all(isinstance(trade, dict) for trade in label['trades']), 'Invalid target trade')
        queries.append(query)
        executions += len(label['trades'])
    require(type(record.get('target_count')) is int and record['target_count'] == len(queries),
            'Conversation target_count mismatch')
    require(type(record.get('execution_count')) is int and record['execution_count'] == executions,
            'Conversation execution_count mismatch')
    return queries, executions


def scan_dataset(source_dir):
    """Validate raw-byte hashes, role/order/identity contracts and declared counts."""
    root = Path(source_dir)
    require(root.is_dir() and not root.is_symlink(), f'Expected a real dataset directory: {root}')
    manifest = read_json(root / 'manifest.json')
    require(isinstance(manifest, dict) and manifest.get('format') == 'actor_market_trade_messages_v1',
            f'{root}: expected actor_market_trade_messages_v1')
    enabled = manifest.get('enabled_splits', list(SPLITS))
    require(enabled in (list(SPLITS), list(SPLITS[:2])), 'Invalid enabled_splits')
    require(isinstance(manifest.get('stats'), dict) and isinstance(manifest.get('split_sha256'), dict),
            'Dataset manifest must contain stats and split_sha256')
    seen_sequences, seen_actor_markets, fixture_splits = set(), set(), {}
    summary, paths = {}, {}
    for split in SPLITS:
        path = paths[split] = split_file(root, split)
        digest = hashlib.sha256()
        targets = executions = count = 0
        earliest = latest = None
        fixtures = set()
        for line, record in lines(path):
            digest.update(line)
            queries, fills = conversation(record)
            sequence = record['sequence_id']
            actor_market = (record['actor_id'].lower(), record['market_id'])
            require(sequence not in seen_sequences, f'Duplicate sequence_id: {sequence}')
            require(actor_market not in seen_actor_markets, f'Duplicate actor/market conversation: {actor_market}')
            seen_sequences.add(sequence)
            seen_actor_markets.add(actor_market)
            fixture = record['fixture_id']
            require(fixture not in fixture_splits or fixture_splits[fixture] == split,
                    f'Fixture appears in more than one split: {fixture}')
            fixture_splits[fixture] = split
            fixtures.add(fixture)
            earliest = queries[0] if earliest is None else min(earliest, queries[0])
            latest = queries[-1] if latest is None else max(latest, queries[-1])
            count += 1
            targets += len(queries)
            executions += fills
        observed = {'conversations': count, 'targets': targets, 'executions': executions}
        declared = manifest['stats'].get(split)
        require(isinstance(declared, dict), f'{split}: missing split stats')
        for key, value in observed.items():
            require(type(declared.get(key, 0)) is int and declared.get(key, 0) == value,
                    f'{split}: {key} count differs from manifest')
        require((count > 0) if split in enabled else (count == 0),
                f'{split}: contents disagree with enabled_splits')
        require(manifest['split_sha256'].get(split) == digest.hexdigest(),
                f'{split}: uncompressed JSONL hash differs from manifest')
        summary[split] = {**observed, 'first_query_us': earliest, 'last_query_us': latest,
                          'fixtures': sorted(fixtures), 'sha256': digest.hexdigest()}
    declared_mapping = manifest.get('fixture_to_split')
    if declared_mapping is not None:
        require(isinstance(declared_mapping, dict), 'Invalid fixture_to_split mapping')
        for fixture, split in fixture_splits.items():
            require(declared_mapping.get(fixture) == split, f'Fixture split differs from manifest: {fixture}')
    return {'root': root, 'manifest': manifest, 'paths': paths, 'splits': summary}


def _chronology(dataset):
    enabled = dataset['manifest'].get('enabled_splits', list(SPLITS))
    for left, right in zip(enabled, enabled[1:]):
        end = dataset['splits'][left]['last_query_us']
        start = dataset['splits'][right]['first_query_us']
        require(end is not None and start is not None and end < start,
                f'{left}/{right} query times overlap or touch. Before enriching either variant, run: '
                'python3 tools/compare_actor_variants.py purge --input BASIC_SFT --out PURGED_BASIC_SFT. '
                'Use that same purged base for all three models. If no holdout remains, collect later markets.')
    return {split: {'first_query_us': data['first_query_us'], 'last_query_us': data['last_query_us'],
                    'targets': data['targets']} for split, data in dataset['splits'].items()}


def validate_chronological_splits(source_dir):
    """Require max(train queries) < min(validation) and max(validation) < min(test)."""
    return _chronology(scan_dataset(source_dir))


def _variant_manifest(dataset, variant):
    manifest = dataset['manifest']
    require(manifest.get('feature_variant') == variant,
            f'{variant}: manifest feature_variant must be {variant!r}')
    if variant == 'basic':
        require('actor_metrics' not in manifest, 'Basic manifest already declares actor metrics')
    else:
        info = manifest.get('actor_metrics')
        require(isinstance(info, dict) and isinstance(info.get('config'), dict),
                f'{variant}: manifest actor_metrics.config is missing')
        require(info['config'].get('history_scope') == SCOPES[variant],
                f'{variant}: incorrect actor_metrics.config.history_scope')
        require(info.get('history_scope') == SCOPES[variant],
                f'{variant}: incorrect actor_metrics.history_scope')
        require(info.get('strict_prior') is True and info.get('target_messages_unchanged') is True,
                f'{variant}: manifest must declare strict prior metrics and unchanged targets')


def _metric_context(context, variant):
    if variant == 'basic':
        require('actor_metrics' not in context, 'Basic context already contains actor_metrics')
        return context
    require(isinstance(context.get('actor_metrics'), dict), f'{variant}: missing actor_metrics context')
    info = context['actor_metrics']
    require({'values', 'sample_counts', 'scope'} <= set(info)
            and set(info) <= {'values', 'sample_counts', 'scope', 'return_period_seconds'}
            and isinstance(info['values'], dict)
            and isinstance(info['sample_counts'], dict), f'{variant}: invalid compact actor_metrics')
    require(info['scope'] == ('current_market' if variant == 'inmarket' else 'global_wallet'),
            f'{variant}: incorrect actor_metrics prompt scope')
    if 'return_period_seconds' in info:
        require(_METRICS.number(info['return_period_seconds'], 'return_period_seconds') > 0,
                f'{variant}: return_period_seconds must be positive')
    return {key: value for key, value in context.items() if key != 'actor_metrics'}


def compare_variants(basic, inmarket, global_dataset):
    datasets = {name: scan_dataset(path) for name, path in
                (('basic', basic), ('inmarket', inmarket), ('global', global_dataset))}
    for variant, dataset in datasets.items():
        _variant_manifest(dataset, variant)
        _chronology(dataset)
    reference = datasets['basic']
    for name in ('inmarket', 'global'):
        other = datasets[name]['manifest']
        for key in ('enabled_splits', 'fixture_to_split'):
            require(reference['manifest'].get(key) == other.get(key), f'{name}: {key} changed')
    defaults = {'selected_features': list(_METRICS.ACTOR_METRIC_NAMES), 'lookback_seconds': None,
                'min_return_periods': 30, 'metric_significant_digits': 10}
    local_config = datasets['inmarket']['manifest']['actor_metrics']['config']
    global_config = datasets['global']['manifest']['actor_metrics']['config']
    for key, default in defaults.items():
        left, right = local_config.get(key, default), global_config.get(key, default)
        if key == 'selected_features':
            require(isinstance(left, list) and isinstance(right, list), 'Invalid selected_features')
            left, right = sorted(left), sorted(right)
        require(left == right, f'Inmarket/global {key} differs; use identical metric settings for a scope comparison')
    report = {'status': 'comparable', 'strict_split_chronology': True,
              'target_messages_identical': True, 'basic_context_identical': True,
              'character_count_semantics': 'system_and_user_message_characters_once_per_conversation_not_tokens',
              'splits': {}}
    for split in SPLITS:
        target_digest = hashlib.sha256()
        prompt_lengths = {name: [] for name in datasets}
        sentinel = object()
        rows = [lines(dataset['paths'][split]) for dataset in datasets.values()]
        for triple in zip_longest(*rows, fillvalue=sentinel):
            require(all(item is not sentinel for item in triple), f'{split}: conversation counts differ')
            records = {name: item[1] for name, item in zip(datasets, triple)}
            base = records['basic']
            metadata = {key: value for key, value in base.items() if key != 'messages'}
            for name, record in records.items():
                require({key: value for key, value in record.items() if key != 'messages'} == metadata,
                        f'{split}/{name}: conversation identity, order or metadata differs')
                require(len(record['messages']) == len(base['messages']), f'{split}/{name}: message counts differ')
                prompt_lengths[name].append(sum(len(message['content']) for message in record['messages']
                                                if message['role'] in ('system', 'user')))
            for offset, basic_message in enumerate(base['messages']):
                if basic_message['role'] != 'user':
                    for name, record in records.items():
                        require(record['messages'][offset] == basic_message,
                                f'{split}/{name}: system or assistant message differs')
                    if basic_message['role'] == 'assistant':
                        query_time = loads(base['messages'][offset - 1]['content'])['query_time']
                        identity = [base['sequence_id'], base['actor_id'], base['market_id'], query_time,
                                    basic_message]
                        target_digest.update(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                                        separators=(',', ':')).encode() + b'\n')
                    continue
                expected = _metric_context(loads(basic_message['content']), 'basic')
                for name, record in records.items():
                    message = record['messages'][offset]
                    require({key: value for key, value in message.items() if key != 'content'}
                            == {key: value for key, value in basic_message.items() if key != 'content'},
                            f'{split}/{name}: user message metadata differs')
                    require(_metric_context(loads(message['content']), name) == expected,
                            f'{split}/{name}: basic user context differs')
        summary = reference['splits'][split]
        report['splits'][split] = {key: summary[key] for key in ('conversations', 'targets', 'executions')}
        report['splits'][split]['target_sha256'] = target_digest.hexdigest()
        report['splits'][split]['prompt_characters'] = {}
        for name, values in prompt_lengths.items():
            ordered = sorted(values)
            report['splits'][split]['prompt_characters'][name] = {
                'mean': sum(values) / len(values) if values else None,
                'p95': ordered[(95 * len(ordered) + 99) // 100 - 1] if ordered else None,
                'total': sum(values)}
    return report


def _copy_tree(source, destination):
    require(not source.is_symlink(), f'Symlink in auxiliary files: {source}')
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            _copy_tree(child, destination / child.name)
    else:
        require(source.is_file(), f'Unexpected auxiliary file: {source}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def purge_overlapping_splits(input_path, output_path):
    """Retain whole conversations in original splits with strict chronological boundaries."""
    source = scan_dataset(input_path)
    root, output = source['root'].resolve(), Path(output_path)
    resolved = output.resolve()
    require(not output.exists() and not output.is_symlink(), f'Output already exists: {output}')
    require(resolved != root and root not in resolved.parents and resolved not in root.parents,
            'Purge output must be separate from source')
    require(source['manifest'].get('feature_variant') == 'basic' and 'actor_metrics' not in source['manifest'],
            'Purge the basic dataset before deriving either metrics variant')
    output.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f'.{output.name}.', dir=output.parent))
    enabled = source['manifest'].get('enabled_splits', list(SPLITS))
    retained, removed, counts, hashes, fixture_mapping = set(), {}, {}, {}, {}
    try:
        previous_max = None
        for split in SPLITS:
            count = targets = executions = 0
            latest = earliest = None
            dropped = []
            digest = hashlib.sha256()
            with (work / f'{split}.jsonl').open('wb') as stream:
                for line, record in lines(source['paths'][split]):
                    queries, fills = conversation(record)
                    if previous_max is not None and queries[0] <= previous_max:
                        dropped.append({'sequence_id': record['sequence_id'], 'fixture_id': record['fixture_id'],
                                        'actor_id': record['actor_id'], 'market_id': record['market_id'],
                                        'targets': len(queries), 'executions': fills,
                                        'first_query_us': queries[0], 'last_query_us': queries[-1],
                                        'must_start_after_us': previous_max})
                        continue
                    stream.write(line)
                    digest.update(line)
                    retained.add(record['sequence_id'])
                    fixture_mapping[record['fixture_id']] = split
                    count += 1
                    targets += len(queries)
                    executions += fills
                    earliest = queries[0] if earliest is None else min(earliest, queries[0])
                    latest = queries[-1] if latest is None else max(latest, queries[-1])
            require(count > 0 or split not in enabled,
                    f'Chronological purge would leave {split} empty. Collect later markets or rebuild '
                    'the basic fixture split with enough nonoverlapping conversations. No output was published.')
            removed[split] = dropped
            counts[split] = {'conversations': count, 'targets': targets, 'executions': executions}
            hashes[split] = digest.hexdigest()
            if split in enabled:
                previous_max = latest
        for name in ('source_audit.jsonl', 'split_plan.json'):
            path = root / name
            require(path.is_file() and not path.is_symlink(), f'Missing or unsafe auxiliary file: {path}')
        audit = root / 'audit'
        if audit.exists() or audit.is_symlink():
            _copy_tree(audit, work / 'audit')
        (work / 'audit').mkdir(exist_ok=True)
        require(not (work / 'audit/purge_source_manifest.json').exists(),
                'Dataset was already purged; use the original basic dataset')
        for name in ('manifest.json', 'split_plan.json', 'source_audit.jsonl'):
            _copy_tree(root / name, work / 'audit' / ('purge_source_' + name))
        audit_sequences = set()
        with (work / 'source_audit.jsonl').open('wb') as stream:
            for line, record in lines(root / 'source_audit.jsonl'):
                require(isinstance(record.get('sequence_id'), str), 'Source audit lacks sequence_id')
                require(record['sequence_id'] not in audit_sequences, 'Duplicate source audit sequence_id')
                audit_sequences.add(record['sequence_id'])
                if record['sequence_id'] in retained:
                    stream.write(line)
        require(retained <= audit_sequences, 'Source audit is missing retained conversations')
        dropped_count = sum(len(values) for values in removed.values())
        with (work / 'audit/purged_conversations.jsonl').open('w', encoding='utf-8') as stream:
            for split, records in removed.items():
                for record in records:
                    stream.write(json.dumps({'split': split, **record}, separators=(',', ':')) + '\n')
        manifest = copy.deepcopy(source['manifest'])
        manifest.update(created_at=datetime.now(timezone.utc).isoformat(), stats=counts,
                        split_sha256=hashes, fixture_to_split=fixture_mapping,
                        token_lengths_checked=False, max_length_checked=None,
                        converter_sha256=_METRICS.sha256(Path(__file__)),
                        global_query_time_separation_enforced=True)
        manifest['chronological_purge'] = {
            'policy': 'keep_all_train_drop_entire_holdout_conversations_starting_at_or_before_prior_split_max_query',
            'conversations_dropped': dropped_count,
            'targets_dropped': sum(record['targets'] for values in removed.values() for record in values),
            'source_manifest_sha256': _METRICS.sha256(root / 'manifest.json'),
            'source_split_sha256': source['manifest']['split_sha256'],
            'source_stats': source['manifest']['stats'],
            'sources_describe_original_exports': True,
            'audit': 'audit/purged_conversations.jsonl',
            'train_unchanged': True, 'retained_conversations_unchanged': True}
        # This is cohort selection, not truncation of any retained target or turn.
        manifest['targets_truncated_or_dropped'] = (
            manifest.get('targets_truncated_or_dropped', 0) + manifest['chronological_purge']['targets_dropped'])
        plan = read_json(root / 'split_plan.json')
        require(isinstance(plan, dict), 'Invalid split plan')
        plan.update(fixture_to_split=fixture_mapping, chronological_purge=manifest['chronological_purge'])
        write_json(work / 'split_plan.json', plan)
        write_json(work / 'manifest.json', manifest)
        ranges = validate_chronological_splits(work)
        # Reject source mutation during this operation before publishing output.
        checked_source = scan_dataset(root)
        require(checked_source['manifest'] == source['manifest'], 'Source manifest changed during purge')
        require(not output.exists(), f'Output appeared during purge: {output}')
        work.rename(output)
        return {'output': str(output), 'splits': counts, 'ranges': ranges,
                'conversations_dropped': dropped_count,
                'targets_dropped': manifest['chronological_purge']['targets_dropped'],
                'train_unchanged': True}
    finally:
        if work.exists():
            shutil.rmtree(work)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == 'purge':
        parser = argparse.ArgumentParser(description='Purge whole holdout conversations to enforce chronological splits.')
        parser.add_argument('--input', type=Path, required=True)
        parser.add_argument('--out', type=Path, required=True)
        args = parser.parse_args(argv[1:])
        report = purge_overlapping_splits(args.input, args.out)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--basic', type=Path, required=True)
        parser.add_argument('--inmarket', type=Path, required=True)
        parser.add_argument('--global', dest='global_dataset', type=Path, required=True)
        args = parser.parse_args(argv)
        report = compare_variants(args.basic, args.inmarket, args.global_dataset)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f'Error: {error}', file=sys.stderr)
        raise SystemExit(1)
