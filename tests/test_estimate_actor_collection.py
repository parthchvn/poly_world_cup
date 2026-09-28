import gzip
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location('estimate_actor_collection',
    Path(__file__).resolve().parents[1] / 'tools/estimate_actor_collection.py')
estimate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(estimate)
ACTOR = '0x' + '1' * 40
OTHER = '0x' + '2' * 40
QUERY = '2026-06-14T17:06:46Z'
CUTOFF = estimate.timestamp_seconds(QUERY)


class CollectionEstimateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / 'common/prepared'

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def dataset(self, path):
        self.write(path / 'manifest.json', {
            'stats': {split: {'targets': count} for split, count in
                      zip(estimate.SPLITS, (40, 2, 2))}})
        for split in estimate.SPLITS:
            (path / f'{split}.jsonl').write_text('{}\n')

    def cohort(self):
        self.dataset(self.prepared / 'basic')
        rows = [{'source_actor_file': actor + '.jsonl.gz',
                 'last_query_time': timestamp, 'source_export': '/another/computer/ignored'}
                for actor, timestamp in ((ACTOR, QUERY), (OTHER, QUERY),
                                         (ACTOR, '2026-06-13T00:00:00Z'))]
        (self.prepared / 'basic/source_audit.jsonl').write_text(
            '\n'.join(json.dumps(row) for row in rows))

    def capture(self, actor=ACTOR, status='exhausted', cutoff=CUTOFF,
                pages=1, name='capture', start=1):
        directory = self.root / 'wallet_cache/wallet_histories' / actor / name
        items = []
        for index in range(pages):
            filename = f'pages/{index:08d}.json'
            self.write(directory / filename, {'rows': [0] * 10})
            items.append({'file': filename, 'row_count': 10})
        self.write(directory / 'manifest.json', {
            'schema': 'wallet_execution_capture_v1',
            'parameters': {'user': actor, 'start': start, 'end': cutoff,
                'limit': 1000, 'taker_only': False, 'filter_type': 'TOKENS',
                'filter_amount': '0.000001'},
            'pages': items, 'page_count': pages, 'row_count': pages * 10,
            'api_traversal_status': status})
        return directory

    def run_report(self, *options):
        args = estimate.parse_args(['--root', str(self.root), *options])
        with patch.object(socket, 'socket', side_effect=AssertionError('No network allowed')):
            return estimate.estimate(args)

    def test_offline_unique_actor_max_cutoff_and_completed_variants(self):
        self.cohort()
        self.dataset(self.root / 'inmarket/sft')
        self.capture()
        self.capture(OTHER, status='paused', pages=3)
        report = self.run_report('--elapsed-hours', '7')
        self.assertEqual(report['network_requests_made'], 0)
        self.assertEqual(report['wallets']['required_wallets'], 2)
        self.assertEqual(report['wallets']['matching_wallets_exhausted'], 1)
        self.assertEqual(report['wallets']['matching_wallets_paused'], 1)
        self.assertEqual(report['variants']['basic']['network_requests_remaining'], 0)
        self.assertEqual(report['variants']['inmarket']['network_requests_remaining'], 0)
        self.assertEqual(report['variants']['global']['pagination_reads_remaining_lower_bound'], 1)
        self.assertIsNone(report['variants']['global']['network_requests_remaining'])
        self.assertEqual(report['variants']['basic']['targets'], 44)
        self.assertIn('not a random sample', report['wallets']['sample_warning'])
        self.assertNotIn('eta', report['observed_pace'])

    def test_wrong_cutoff_or_truncated_start_is_not_reusable(self):
        self.cohort()
        self.capture(cutoff=CUTOFF + 1)
        self.capture(OTHER, start=100)
        report = self.run_report()
        self.assertEqual(report['wallets']['all_capture_count'], 2)
        self.assertEqual(report['wallets']['not_started_wallets'], 2)
        self.assertEqual(report['wallets']['matching_wallets_exhausted'], 0)

    def test_missing_normalized_page_cannot_count_as_completed(self):
        self.cohort()
        directory = self.capture()
        (directory / 'pages/00000000.json').unlink()
        report = self.run_report()
        self.assertEqual(report['wallets']['matching_wallets_exhausted'], 0)
        self.assertTrue(report['warnings'])

    def test_compressed_normalized_pages_use_the_actual_compressed_size(self):
        self.cohort()
        directory = self.capture(pages=2)
        page = directory / 'pages/00000000.json'
        compressed = page.with_suffix('.json.gz')
        compressed.write_bytes(gzip.compress(page.read_bytes()))
        page.unlink()
        state = json.loads((directory / 'manifest.json').read_text())
        state['pages'][0]['file'] += '.gz'
        self.write(directory / 'manifest.json', state)
        report = self.run_report()
        self.assertEqual(report['wallets']['matching_wallets_exhausted'], 1)
        self.assertEqual(report['wallets']['normalized_page_bytes'], compressed.stat().st_size
                         + (directory / 'pages/00000001.json').stat().st_size)

    def test_cached_http_and_normalized_storage_both_count(self):
        self.cohort()
        self.capture()
        http = self.root / 'wallet_cache/http'
        self.write(http / 'requests/a.json', {'retrieved_at': '2026-09-28T01:00:00Z'})
        self.write(http / 'requests/b.json', {'retrieved_at': '2026-09-28T03:00:00Z'})
        (http / 'bodies').mkdir()
        (http / 'bodies/body.json.gz').write_bytes(gzip.compress(b'x' * 10000))
        self.write(http / 'captures/history.json', {'retrieved_at': 'not timing'})
        report = self.run_report()
        sizes = report['storage']['components']
        expected = sizes['wallet_http']['bytes'] / 2 + sizes['wallet_histories']['bytes']
        self.assertEqual(report['scenario_assumptions']['combined_bytes_per_page'], round(expected))
        self.assertEqual(report['http_cache']['retrieval_timestamp_span_hours'], 2)
        self.assertIsNone(report['http_cache']['request_duration_seconds'])
        self.assertGreater(sizes['wallet_cache']['bytes'], sizes['wallet_http']['bytes'])

    def test_explicit_parallel_scenarios_respect_rate_and_wallet_dependencies(self):
        self.cohort()
        report = self.run_report('--pages-per-wallet', '20', '--seconds-per-page', '10',
                                 '--workers', '4', '--min-interval', '0.25')
        row = report['global_remaining_scenarios'][0]
        self.assertEqual(row['additional_requests'], 40)
        # Two sequential 20-page wallets cannot realize 4-way page parallelism.
        self.assertEqual(row['ideal_parallel_network_hours'], round(200 / 3600, 3))
        self.assertEqual(row['serial_network_hours'], round(400 / 3600, 3))
        limited = self.run_report('--pages-per-wallet', '1', '--seconds-per-page', '1',
                                  '--min-interval', '60')['global_remaining_scenarios'][0]
        self.assertEqual(limited['ideal_parallel_network_hours'], round(120 / 3600, 3))

    def test_missing_cohort_reports_unknown_instead_of_zero(self):
        report = self.run_report()
        self.assertIsNone(report['wallets']['required_wallets'])
        self.assertIsNone(report['variants']['global']['pagination_reads_remaining_lower_bound'])
        self.assertEqual(report['global_remaining_scenarios'], [])
        self.assertIsNone(report['variants']['basic']['network_requests_remaining'])

    def test_fallback_selected_exports_handles_gzip(self):
        path = self.prepared / 'selected_exports/market_1/actors/a.jsonl.gz'
        path.parent.mkdir(parents=True)
        with gzip.open(path, 'wt') as stream:
            stream.write(json.dumps({'record_type': 'interval'}) + '\n')
            stream.write(json.dumps({'record_type': 'trade', 'actor_id': ACTOR,
                                     'timestamp': QUERY}) + '\n')
        report = self.run_report()
        self.assertEqual(report['wallets']['required_wallets'], 1)
        self.assertEqual(report['target_cutoff_source'], 'selected_exports trade timestamps')

    def test_complete_global_needs_no_refetch_even_without_cache(self):
        self.cohort()
        self.dataset(self.root / 'global/sft')
        report = self.run_report()
        self.assertEqual(report['variants']['global']['network_requests_remaining'], 0)
        for row in report['global_remaining_scenarios']:
            self.assertEqual(row['additional_requests'], 0)
            self.assertEqual(row['additional_output_gib'], 0)

    def test_symlinks_not_counted_as_owned_storage(self):
        elsewhere = self.root / 'elsewhere'
        elsewhere.mkdir()
        (elsewhere / 'large').write_bytes(b'x' * 1000)
        path = self.root / 'test'
        path.mkdir()
        (path / 'linked').symlink_to(elsewhere, target_is_directory=True)
        self.assertEqual(estimate.directory_size(path)['bytes'], 0)

    def test_report_can_be_written_and_does_not_modify_source(self):
        self.cohort()
        audit = self.prepared / 'basic/source_audit.jsonl'
        before = audit.read_bytes()
        output = self.root / 'estimate.json'
        with patch('builtins.print'):
            estimate.main(['--root', str(self.root), '--out', str(output)])
        self.assertEqual(json.loads(output.read_text())['network_requests_made'], 0)
        self.assertEqual(audit.read_bytes(), before)

    def test_summary_is_concise_and_out_remains_full_json(self):
        self.cohort()
        self.capture()
        output = self.root / 'estimate.json'
        with patch('builtins.print') as printed:
            estimate.main(['--root', str(self.root), '--summary', '--out', str(output)])
        displayed = printed.call_args.args[0]
        self.assertIn('1/2 API traversals complete', displayed)
        self.assertIn('Assumptions, not ETAs', displayed)
        self.assertIn('Extra cache GiB', displayed)
        self.assertLess(len(displayed.splitlines()), 30)
        saved = json.loads(output.read_text())
        self.assertEqual(saved['format'], 'actor_collection_estimate_v1')
        self.assertEqual(len(saved['global_remaining_scenarios']), 9)

    def test_invalid_assumptions_rejected(self):
        for flags in (('--workers', '0'), ('--elapsed-hours', '-1'),
                      ('--min-interval', 'nan'), ('--pages-per-wallet', '0.5')):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                self.run_report(*flags)


if __name__ == '__main__':
    unittest.main()
