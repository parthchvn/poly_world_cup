import builtins
import copy
import fcntl
import gzip
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
import test_actor_activity as cli
import actor_activity_common as activity
from tests import test_actor_variant_comparison as comparison_fixture


class ActivityTests(unittest.TestCase):
    def setUp(self):
        self.fixture = comparison_fixture.ActorVariantComparisonTests()
        self.fixture.setUp()
        self.root = self.fixture.root
        self.config = {'history_scope': 'actor_and_binary_market',
            'selected_features': sorted(activity.FEATURES), 'metric_significant_digits': 10}
        manifest = self.fixture.datasets['inmarket'] / 'manifest.json'
        data = json.loads(manifest.read_text())
        data['actor_metrics']['config'] = self.config
        manifest.write_text(json.dumps(data))
        self.source = self.root / 'exports/market_99'
        (self.source / 'actors').mkdir(parents=True)
        self.kickoff = activity.timestamp_us('2026-06-12T17:00:00Z')
        condition = '0x' + '9' * 64
        market = {'market_id': '99', 'condition_id': condition, 'question': 'Will this match be a draw?',
            'kickoff_utc': activity.utc(self.kickoff), 'fixture_id': 'espn:999',
            'tokens': [{'outcome': 'Yes', 'token_id': '1'}, {'outcome': 'No', 'token_id': '2'}]}
        (self.source / 'market.json').write_text(json.dumps(market))
        for number in (0, 1, 10, 11):
            actor = '0x' + f'{number:040x}'
            rows = []
            previous = self.kickoff - 3600_000_000
            for i, offset in enumerate((-30, 60, 240)):
                stamp = activity.utc(self.kickoff + offset * 1_000_000)
                interval = {'start': activity.utc(previous), 'end': stamp, 'start_inclusive': False, 'end_inclusive': False}
                base = {'actor_id': actor, 'market_id': '99', 'condition_id': condition,
                        'news': [{'time': '2099-01-01T00:00:00Z', 'text': 'DO NOT COPY RAW INTERVAL CONTEXT'}],
                        'market_context': {'future': 'DO NOT COPY'}, 'actor_snapshot_ref': 'DO NOT READ'}
                rows.extend([{**base, 'row_index': 2*i, 'record_type': 'interval', 'interval': interval,
                              'label': {'action': 'NO_TRADE'}},
                             {**base, 'row_index': 2*i+1, 'record_type': 'trade', 'timestamp': stamp,
                              'context_interval': interval, 'label': {'action': 'TRADE', 'trades': [
                                  {'time': stamp, 'side': 'BUY', 'outcome': 'Yes', 'shares': '2', 'price': '0.4'}]}}])
                previous = self.kickoff + offset * 1_000_000
            (self.source / 'actors' / (actor + '.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        prices = []
        for offset in (-10, 0, 30, 60, 120, 240):
            for outcome, value in (('yes', '0.4'), ('no', '0.6')):
                prices.append({'outcome': outcome, 'observed_at': activity.utc(self.kickoff + offset*1_000_000),
                    'source': 'polymarket_clob_prices_history', 'price': value})
        price_path = self.source / 'market_price_history.jsonl'
        price_path.write_text(''.join(json.dumps(r)+'\n' for r in prices))
        news = [{'time_utc': activity.utc(self.kickoff + offset*1_000_000), 'kind': 'foul',
                 'text': f'news at {offset}'} for offset in (-10, 0, 60, 120, 240, 999)]
        (self.source / 'espn_events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in news))
        manifest = {'format': 'actor_market_intervals_v1', 'market_id': '99', 'condition_id': condition,
            'created_at': '2026-06-13T17:00:00Z', 'espn_event_id': '999', 'origin_utc': activity.utc(self.kickoff-3600_000_000),
            'max_trades_per_actor': 20, 'market_context_version': 2, 'market_price_max_age_seconds': 300,
            'source': {'api_traversal_status': 'exhausted'},
            'market_price_history': {'sha256': activity.sha(price_path)},
            'counts': {'actors': 4, 'rows': 24, 'distinct_trade_times': 12, 'trade_observations': 12}}
        (self.source / 'manifest.json').write_text(json.dumps(manifest))

    def tearDown(self):
        self.fixture.tearDown()

    def args(self, *extra):
        return cli.parse_args(['prepare', '--basic-sft', str(self.fixture.datasets['basic']),
            '--inmarket-sft', str(self.fixture.datasets['inmarket']), '--input-root', str(self.root / 'exports'),
            '--out', str(self.root / 'activity'), '--targets', '12', '--match-minutes', '6', *extra])

    def prepare(self, args=None):
        with patch('sys.stdout', new_callable=io.StringIO), patch('urllib.request.urlopen', side_effect=AssertionError('network')):
            return activity.prepare(args or self.args())

    def raw(self):
        source = activity.metrics.discover_exports([self.source], None)[0]
        source['market'] = json.loads((self.source / 'market.json').read_text())
        actor_file = self.source / 'actors' / ('0x' + f'{10:040x}' + '.jsonl')
        _, groups, _ = activity.metrics.actor_trade_groups(actor_file, source)
        return source, groups

    def change_manifest(self, **fields):
        path = self.source / 'manifest.json'
        value = json.loads(path.read_text())
        value.update(fields)
        path.write_text(json.dumps(value))

    def test_preparation_excludes_union_of_train_and_validation_and_pairs_contexts(self):
        meta = self.prepare()
        self.assertEqual(meta['actor_overlap_train_validation'], 0)
        self.assertEqual(meta['excluded_actor_market_files'], 2)
        self.assertEqual(meta['positive_windows'], 4)
        self.assertEqual(meta['negative_windows'], 8)
        self.assertEqual(meta['selected_wallets'], 2)
        self.assertTrue(meta['retrospectively_filtered_actor_cohort'])
        _, rows = activity.read_bundle(self.root / 'activity')
        self.assertEqual({int(r['actor_id'], 16) for r in rows['labels']}, {10, 11})
        for basic, inmarket in zip(rows['basic'], rows['inmarket']):
            self.assertNotIn('DO NOT', activity.dump(basic))
            b, m = [json.loads(r['messages'][1]['content']) for r in (basic, inmarket)]
            m.pop('actor_metrics')
            self.assertEqual(b, m)
            self.assertEqual([m['role'] for m in basic['messages']], ['system', 'user'])

    def test_current_and_future_trades_prices_news_excluded_at_input_cutoff(self):
        source, groups = self.raw()
        timeline = activity.load_timeline(source)
        q = self.kickoff + 60_000_000
        basic, enriched, _ = activity.context_at(source, timeline, groups, q, 60_000_000, self.config, 20, 1200)
        self.assertEqual(len(basic['prior_executions']), 1)
        self.assertEqual(enriched['actor_metrics']['sample_counts']['captured_executions'], 1)
        self.assertEqual(basic['market_context']['yes']['age_seconds'], '30.0')
        self.assertEqual([r['text'] for r in basic['news']], ['news at -10', 'news at 0'])
        self.assertEqual(activity.answer_at(groups, q, 60_000_000), 1)
        self.assertEqual(activity.answer_at(groups, self.kickoff, 60_000_000), 0)
        self.assertEqual(activity.answer_at(groups, self.kickoff + 180_000_000, 60_000_000), 0)

    def test_future_values_cannot_change_prior_features_or_input(self):
        source, groups = self.raw()
        timeline = activity.load_timeline(source)
        before = activity.context_at(source, timeline, groups, self.kickoff, 60_000_000, self.config, 20, 1200)
        for group in groups[1:]:
            group['trades'][0]['shares'] = activity.metrics.number('999999', 'shares')
            group['expected'][0]['shares'] = '999999'
        timeline['news'][-1][1]['text'] = 'FUTURE CHANGED'
        timeline['prices']['yes'][-1] = (timeline['prices']['yes'][-1][0], '0.99')
        after = activity.context_at(source, timeline, groups, self.kickoff, 60_000_000, self.config, 20, 1200)
        self.assertEqual(before, after)

    def test_future_execution_changes_do_not_move_selected_grid_windows(self):
        self.prepare(self.args('--targets', '5'))
        _, original = activity.read_bundle(self.root / 'activity')
        for path in (self.source / 'actors').glob('*'):
            path.write_text(path.read_text().replace('17:04:00', '17:04:30'))
        self.prepare(self.args('--targets', '5', '--out', str(self.root / 'changed')))
        _, changed = activity.read_bundle(self.root / 'changed')
        self.assertEqual([r['id'] for r in original['labels']], [r['id'] for r in changed['labels']])

    def test_enrollment_requires_strictly_prior_first_execution(self):
        q = list(activity.eligible_queries(self.kickoff, self.kickoff, 180_000_000, 60_000_000))
        self.assertEqual(q, [self.kickoff + 60_000_000, self.kickoff + 120_000_000])
        self.assertEqual(list(activity.eligible_queries(self.kickoff + 999_000_000, self.kickoff, 180_000_000, 60_000_000)), [])

    def test_cap_defaults_to_twenty_and_cannot_recover_previously_removed_wallets(self):
        args = self.args('--max-trades-per-actor', '0')
        self.assertEqual(self.args().max_trades_per_actor, 20)
        with self.assertRaisesRegex(ValueError, 'cannot recover'):
            self.prepare(args)
        self.change_manifest(max_trades_per_actor=None)
        self.assertFalse(self.prepare(args)['retrospectively_filtered_actor_cohort'])

    def test_trade_cap_applied_to_unfiltered_source_before_window_sampling(self):
        self.change_manifest(max_trades_per_actor=None)
        args = self.args('--max-trades-per-actor', '2')
        with self.assertRaisesRegex(ValueError, 'No unseen wallets'):
            self.prepare(args)

    def test_partial_capture_is_not_silently_labeled_negative(self):
        self.change_manifest(source={'api_traversal_status': 'paused'})
        with self.assertRaisesRegex(ValueError, 'incomplete or uncertified'):
            self.prepare()
        self.assertFalse((self.root / 'activity').exists())

    def test_incomplete_export_counts_fail(self):
        (self.source / 'actors' / ('0x' + f'{10:040x}' + '.jsonl')).unlink()
        with self.assertRaisesRegex(ValueError, 'count mismatch'):
            self.prepare()

    def test_seen_match_and_temporally_early_match_excluded(self):
        market_path = self.source / 'market.json'
        market = json.loads(market_path.read_text())
        market['market_id'] = '0'
        self.change_manifest(market_id='0')
        market_path.write_text(json.dumps(market))
        with self.assertRaisesRegex(ValueError, 'No later held-out exports'):
            self.prepare()
        market['market_id'] = '99'
        self.change_manifest(market_id='99')
        market['kickoff_utc'] = '2026-06-06T10:00:00Z'
        market_path.write_text(json.dumps(market))
        with self.assertRaisesRegex(ValueError, 'No later held-out exports'):
            self.prepare()

    def test_fixture_cannot_be_renamed_to_bypass_holdout(self):
        path = self.source / 'market.json'
        value = json.loads(path.read_text())
        value['fixture_id'] = 'espn:888'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, 'fixture identity'):
            self.prepare()

    def test_capture_must_follow_complete_prediction_window(self):
        self.change_manifest(created_at='2026-06-12T17:02:00Z')
        with self.assertRaisesRegex(ValueError, 'Capture predates'):
            self.prepare()

    def test_stale_prices_null_without_dropping_negative_or_positive_windows(self):
        source, groups = self.raw()
        timeline = activity.load_timeline(source)
        basic, _, _ = activity.context_at(source, timeline, groups, self.kickoff + 1000_000_000,
                                         60_000_000, self.config, 1, 1200)
        self.assertEqual(basic['market_context'], {'yes': None, 'no': None})
        self.assertEqual(len(basic['prior_executions']), 1)
        self.assertEqual(basic['earlier_execution_groups_omitted'], 2)

    def test_bundle_hash_and_cutoff_tampering_detected(self):
        self.prepare()
        path = self.root / 'activity/basic.jsonl.gz'
        rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
        context = json.loads(rows[0]['messages'][1]['content'])
        context['prior_executions'][0]['time'] = '2099-01-01T00:00:00Z'
        rows[0]['messages'][1]['content'] = json.dumps(context)
        path.write_bytes(gzip.compress(('\n'.join(json.dumps(r) for r in rows)+'\n').encode()))
        with self.assertRaisesRegex(ValueError, 'checksum'):
            activity.read_bundle(self.root / 'activity')

    def test_prompt_cutoff_audited_even_if_file_checksums_are_updated(self):
        self.prepare()
        root = self.root / 'activity'
        meta = json.loads((root / 'manifest.json').read_text())
        for variant in ('basic', 'inmarket'):
            path = root / (variant + '.jsonl.gz')
            rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
            context = json.loads(rows[0]['messages'][1]['content'])
            context['news'].append({'time': context['query_time'], 'text': 'same-time event'})
            rows[0]['messages'][1]['content'] = json.dumps(context)
            path.write_bytes(gzip.compress(('\n'.join(json.dumps(r) for r in rows)+'\n').encode()))
            meta['files'][variant] = activity.sha(path)
        (root / 'manifest.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'news leaked'):
            activity.read_bundle(root)

    def test_sampling_is_deterministic_and_does_not_force_both_classes(self):
        first = self.prepare(self.args('--targets', '1'))
        second = self.prepare(self.args('--targets', '1', '--out', str(self.root / 'another')))
        self.assertEqual(first['target_sha256'], second['target_sha256'])
        self.assertEqual(first['files'], second['files'])
        self.assertEqual(first['positive_windows'] + first['negative_windows'], 1)

    def test_metrics_handle_ties_imbalance_absent_classes_and_nonfinite_scores(self):
        stats = activity.summarize([0, 0, 0, 1], [.2, .2, .2, .2])
        self.assertEqual(stats['accuracy'], .75)
        self.assertEqual(stats['balanced_accuracy'], .5)
        self.assertEqual(stats['trade_recall'], 0)
        self.assertIsNone(stats['trade_precision'])
        self.assertEqual(stats['average_precision'], .25)
        self.assertFalse(activity.summarize([0], [.1])['both_classes_present'])
        self.assertIsNone(activity.summarize([0], [.1])['average_precision'])
        perfect = activity.summarize([0, 1], [.1, .9])
        self.assertEqual(perfect['trade_f1'], 1)
        self.assertEqual(perfect['average_precision'], 1)
        with self.assertRaisesRegex(ValueError, 'finite'):
            activity.summarize([1], [float('nan')])

    def test_round_tripped_baseline_scores_are_numeric_and_summarizable(self):
        self.prepare()
        meta, records = activity.read_bundle(self.root / 'activity')
        rows = records['labels']
        self.assertTrue(all(type(r['prior_rate_score']) is float for r in rows))
        result = activity.summarize([r['label'] for r in rows], [r['prior_rate_score'] for r in rows])
        self.assertEqual(result['windows'], meta['targets'])
        self.assertEqual(result, activity.summarize([r['label'] for r in rows], [str(r['prior_rate_score']) for r in rows]))

    def test_probability_rejects_bad_saved_scores_and_accepts_decimal_strings(self):
        for value in (True, None, {}, [], 'nan', 'Infinity', '-0.1', '1.1', 'not a score', float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                activity.probability(value)
        for value in ('0.125', .125, activity.metrics.Decimal('.125')):
            self.assertEqual(activity.probability(value), .125)

    def saved_predictions(self, variant='basic', count=None, limit=0):
        meta, records = activity.read_bundle(self.root / 'activity')
        path = self.root / 'results' / variant
        path.mkdir(parents=True)
        targets, prompts = records['labels'][:limit or meta['targets']], records[variant][:limit or meta['targets']]
        identity = {'format': 'world_cup_activity_scores_v1', 'variant': variant,
            'bundle_sha256': activity.sha(self.root / 'activity/manifest.json'),
            'target_sha256': meta['target_sha256'], 'selected_ids_sha256': activity.digest([r['id'] for r in targets]),
            'limit': limit, 'threshold': .5,
            'score_protocol': 'single_forward_next_token_P(B)/(P(A)+P(B)); A=NO_TRADE; B=TRADE',
            'max_context': 16384, 'seed': 42, 'code': {'old_inference_code': 'preserve_this'}, 'versions': {}}
        activity.write_json(path / 'identity.json', identity)
        rows = []
        for target, prompt in zip(targets, prompts):
            rows.append({**target, 'prior_rate_score': str(target['prior_rate_score']),
                'trade_score': .8 if target['label'] else .2,
                'prediction': 'TRADE' if target['label'] else 'NO_TRADE', 'unconstrained_choice_mass': .02,
                'prompt_sha256': activity.digest(prompt['messages'])})
        if count is not None:
            rows = rows[:count]
        (path / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        return path

    def run_report(self, path=None):
        args = cli.parse_args(['report', '--bundle', str(self.root / 'activity'),
                              '--results', str(path or self.root / 'results')])
        with patch('sys.stdout', new_callable=io.StringIO):
            return args.func(args)

    def test_offline_report_recovers_old_string_scores_without_weights_or_rewriting_journals(self):
        self.prepare()
        roots = [self.saved_predictions(variant) for variant in ('basic', 'inmarket')]
        before = {p: p.read_bytes() for root in roots for p in (root / 'identity.json', root / 'predictions.jsonl')}
        real_import = builtins.__import__
        def cpu_only(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'transformers', 'peft', 'bitsandbytes'}:
                raise AssertionError('No ML imports allowed in offline reporting')
            return real_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=cpu_only), patch.object(cli, 'evaluate', side_effect=AssertionError('No inference')):
            result = self.run_report()
        self.assertEqual(set(result['completed']), {'basic', 'inmarket'})
        self.assertEqual(result['pending'], [])
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        for root in roots:
            summary = json.loads((root / 'summary.json').read_text())
            self.assertEqual(summary['metrics']['windows'], 12)
            self.assertEqual(summary['baselines']['prior_repeat_event_rate']['windows'], 12)
            self.assertTrue(summary['postprocessing']['offline_recovery'])
            self.assertEqual(summary['identity_sha256'], activity.sha(root / 'identity.json'))
        self.assertTrue((self.root / 'results/comparison.json').is_file())
        summaries = {root: (root / 'summary.json').read_bytes() for root in roots}
        self.run_report()
        self.assertEqual(summaries, {root: (root / 'summary.json').read_bytes() for root in roots})

    def test_offline_report_refuses_wrong_prompt_or_baseline_or_bundle(self):
        self.prepare()
        path = self.saved_predictions()
        journal = path / 'predictions.jsonl'
        original = journal.read_bytes()
        for field, value, reason in (('prompt_sha256', 'wrong', 'prompt mismatch'),
                                     ('prior_rate_score', '0.999999', 'baseline score mismatch'),
                                     ('label', 2, 'target/label mismatch')):
            rows = [json.loads(line) for line in original.splitlines()]
            rows[0][field] = value
            journal.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaisesRegex(ValueError, reason):
                self.run_report(path)
            self.assertFalse((path / 'summary.json').exists())
        journal.write_bytes(original)
        identity = json.loads((path / 'identity.json').read_text())
        identity['bundle_sha256'] = 'other_bundle'
        activity.write_json(path / 'identity.json', identity)
        with self.assertRaisesRegex(ValueError, 'different bundle'):
            self.run_report(path)

    def test_offline_report_leaves_incomplete_and_active_journals_untouched(self):
        self.prepare()
        path = self.saved_predictions(count=5)
        journal = path / 'predictions.jsonl'
        original = journal.read_bytes()
        self.assertEqual(self.run_report(path)['completed'], {})
        self.assertFalse((path / 'summary.json').exists())
        self.assertEqual(journal.read_bytes(), original)
        journal.write_bytes(original + b'{"id":')
        self.assertEqual(self.run_report(path)['completed'], {})
        self.assertEqual(journal.read_bytes(), original + b'{"id":')
        with (path / 'evaluation.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_report(path)
        self.assertEqual(result['completed'], {})
        self.assertEqual(result['pending'], [str(path)])

    def test_offline_report_respects_saved_pilot_limit(self):
        self.prepare()
        path = self.saved_predictions(limit=3)
        self.run_report(path)
        summary = json.loads((path / 'summary.json').read_text())
        self.assertEqual(summary['metrics']['windows'], 3)
        self.assertTrue(summary['pilot'])

    def test_resume_keeps_complete_lines_only_and_refuses_mismatched_prompt(self):
        prompt = {'id': 'one', 'messages': [{'role': 'user', 'content': 'past only'}]}
        label = {'id': 'one', 'label': 0}
        row = {**label, 'prompt_sha256': activity.digest(prompt['messages']), 'trade_score': .2}
        path = self.root / 'predictions.jsonl'
        path.write_text(json.dumps(row)+'\n{"id":')
        self.assertEqual(cli.recover(path, [prompt], [label], True), [row])
        self.assertTrue(path.read_bytes().endswith(b'\n'))
        with self.assertRaisesRegex(ValueError, 'prompt/label mismatch'):
            cli.recover(path, [{**prompt, 'messages': []}], [label], True)

    def test_choices_are_single_tokens_and_prompts_not_truncated(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids': {'A': [1], 'B': [2]}.get(text, [3]*10)}
            def apply_chat_template(self, messages, **kwargs):
                return 'prompt'
        tokenizer = Tokenizer()
        prompts = [{'id': 'one', 'messages': []}]
        self.assertEqual(cli.encode_prompts(tokenizer, prompts, 10), ([[3]*10], [1, 2]))
        with self.assertRaisesRegex(ValueError, 'no rows truncated'):
            cli.encode_prompts(tokenizer, prompts, 9)

    def test_run_checks_both_adapters_then_evaluates_sequentially(self):
        args = cli.parse_args(['run', '--bundle', 'bundle', '--basic-run-dir', 'basic_run',
                               '--inmarket-run-dir', 'inmarket_run', '--out', 'out'])
        calls = []
        with patch.object(cli, 'evaluate', side_effect=lambda a: calls.append((a.variant, a.check_only))), patch.object(cli, 'compare') as report:
            cli.run_both(args)
        self.assertEqual(calls, [('basic', True), ('inmarket', True), ('basic', False), ('inmarket', False)])
        report.assert_called_once()

    def test_comparison_requires_same_windows_and_reports_baselines(self):
        self.prepare()
        meta, records = activity.read_bundle(self.root / 'activity')
        for variant in ('basic', 'inmarket'):
            root = self.root / ('result_' + variant)
            root.mkdir()
            identity = {'variant': variant, 'bundle_sha256': 'bundle', 'target_sha256': meta['target_sha256'],
                'selected_ids_sha256': activity.digest([r['id'] for r in records['labels']]),
                'score_protocol': 'fixed', 'max_context': 16384, 'seed': 42, 'code': {}, 'versions': {}}
            activity.write_json(root / 'identity.json', identity)
            predictions = [{**r, 'trade_score': (.9 if r['label'] else .1) if variant == 'basic' else .1}
                           for r in records['labels']]
            (root / 'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
            activity.write_json(root / 'summary.json', {'status': 'completed', 'variant': variant,
                'identity_sha256': activity.sha(root / 'identity.json'),
                'predictions_sha256': activity.sha(root / 'predictions.jsonl'), 'pilot': False,
                'task': meta['task'], 'retrospectively_filtered_actor_cohort': True,
                'baselines': {'always_no_trade': activity.summarize([r['label'] for r in predictions], [0.0]*len(predictions))}})
        args = cli.parse_args(['compare', '--basic', str(self.root / 'result_basic'),
                              '--inmarket', str(self.root / 'result_inmarket'), '--out', str(self.root / 'compare.json')])
        with patch('sys.stdout', new_callable=io.StringIO):
            result = cli.compare(args)
        self.assertEqual(result['models']['basic']['balanced_accuracy'], 1)
        self.assertEqual(result['models']['inmarket']['balanced_accuracy'], .5)
        self.assertIn('always_no_trade', result['models'])
        path = self.root / 'result_inmarket/identity.json'
        identity = json.loads(path.read_text())
        identity['selected_ids_sha256'] = 'different'
        activity.write_json(path, identity)
        path = self.root / 'result_inmarket/summary.json'
        summary = json.loads(path.read_text())
        summary['identity_sha256'] = activity.sha(path.parent / 'identity.json')
        activity.write_json(path, summary)
        with self.assertRaisesRegex(ValueError, 'Comparison differs'):
            cli.compare(args)


if __name__ == '__main__':
    unittest.main()
