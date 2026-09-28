"""Skipping audit-only snapshots must be explicit and preserve SFT examples."""
import io
import json
import unittest
from unittest.mock import patch

from tests import actor_snapshot_fixtures as snapshots
from tests import test_prepare_actor_sft as fixtures

builder = fixtures.builder


class SkipActorSnapshotsTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ActorSFTTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        quiet = patch('sys.stdout', new_callable=io.StringIO)
        quiet.start()
        self.addCleanup(quiet.stop)

    def export(self, skip=False, legacy_namespace=False):
        source, actor_file, rows = self.fixture.source(
            with_market_context=True, context_version=2)
        market = json.loads((source / 'market.json').read_text())
        actor = actor_file.stem
        origin = builder.timestamp_us('2026-06-01T16:00:00Z')
        context = {
            'event_id': '1', 'untimed_events': [],
            'timed_events': [{
                'timestamp_us': origin + 30_000_000, 'news_id': 'goal-1',
                'time_utc': builder.utc_time(origin + 30_000_000),
                'kind': 'goal', 'text': 'A goal.'}],
        }
        trades = [
            {'actor_id': actor, 'time_us': builder.timestamp_us(row['timestamp']),
             'time': row['timestamp'], 'trade': trade}
            for row in rows if row['record_type'] == 'trade'
            for trade in row['label']['trades']
        ]
        output = self.root / 'new_export'
        command = ['1', '--out', str(output), '--cache', str(self.root / 'cache'),
                   '--price-history-file', str(source / 'history_fixture.json')]
        if skip:
            command.append('--skip-actor-snapshots')
        with patch.object(builder, 'export') as capture:
            builder.main(command)
        args = capture.call_args.args[0]
        if legacy_namespace:
            del args.skip_actor_snapshots

        def collect(actors, current_market, options, work):
            self.assertEqual(actors, [actor])
            snapshots.write_snapshot(work / 'actor_snapshots', actor, current_market)
            return snapshots.report()

        with patch.object(builder, 'HttpClient', return_value=object()), \
                patch.object(builder, 'resolve_market', return_value=market), \
                patch.object(builder, 'collect_espn_context', return_value=context), \
                patch.object(builder, 'load_market_trades', return_value=(iter(trades), {})), \
                patch.object(builder, 'collect_actor_snapshots', side_effect=collect) as collect_call:
            manifest = builder.export(args)
        return output, manifest, collect_call

    def test_skip_export_has_no_snapshot_calls_files_or_references(self):
        output, manifest, collect = self.export(skip=True)
        collect.assert_not_called()
        self.assertIsNone(manifest['actor_snapshots'])
        self.assertTrue(manifest['actor_snapshots_skipped'])
        self.assertFalse((output / 'actor_snapshots').exists())
        for file in [output / 'actor_index.jsonl', *list((output / 'actors').glob('*.jsonl'))]:
            for line in file.read_text().splitlines():
                self.assertNotIn('actor_snapshot_ref', json.loads(line))
        source = builder.sft_discover([output], None)[0]
        record, audit, counts = builder.sft_convert_actor(next((output / 'actors').iterdir()), source)
        self.assertEqual(record['target_count'], 2)
        self.assertNotIn('actor_snapshot_ref', audit)

    def test_default_and_legacy_namespace_still_collect_snapshots(self):
        output, manifest, collect = self.export(legacy_namespace=True)
        collect.assert_called_once()
        self.assertEqual(manifest['actor_snapshots']['version'], 1)
        self.assertFalse(manifest['actor_snapshots_skipped'])
        for file in [output / 'actor_index.jsonl', *list((output / 'actors').glob('*.jsonl'))]:
            for line in file.read_text().splitlines():
                self.assertTrue((output / json.loads(line)['actor_snapshot_ref']).is_file())

    def test_incompatible_snapshot_options_fail_before_any_network(self):
        commands = [
            ['1', '--out', str(self.root / 'export')],
            ['sft', '1', '2', '3', '--out', str(self.root / 'sft')],
        ]
        for command in commands:
            with self.subTest(command=command), \
                    patch.object(builder, 'HttpClient', side_effect=AssertionError('network forbidden')), \
                    patch('sys.stderr', new_callable=io.StringIO) as error, self.assertRaises(SystemExit):
                builder.main([*command, '--skip-actor-snapshots',
                              '--actor-snapshots-dir', str(self.root / 'snapshots')])
            self.assertIn('cannot be combined', error.getvalue())

    def test_sft_reuse_requires_explicit_skip_flag_and_recorded_skip(self):
        for i in range(1, 4):
            path, _, _ = self.fixture.source(i, i, with_market_context=True, context_version=2)
            manifest = json.loads((path / 'manifest.json').read_text())
            manifest.update(actor_snapshots=None, actor_snapshots_skipped=True)
            (path / 'manifest.json').write_text(json.dumps(manifest))
        command = ['sft', '1', '2', '3', '--data-root', str(self.root),
                   '--out', str(self.root / 'sft'), '--reuse-existing']
        with patch.object(builder, 'HttpClient', side_effect=AssertionError('network forbidden')):
            with patch('sys.stderr', new_callable=io.StringIO) as error, self.assertRaises(SystemExit):
                builder.main(command)
            self.assertIn('lacks actor snapshots', error.getvalue())
            result = builder.main([*command, '--skip-actor-snapshots'])
        self.assertFalse(result['actor_snapshots_used_as_model_input'])
        self.assertEqual(result['stats']['train']['targets'], 2)

    def test_skip_flag_does_not_silently_accept_missing_snapshot_provenance(self):
        self.fixture.source(with_market_context=True, context_version=2)
        with patch('sys.stderr', new_callable=io.StringIO) as error, self.assertRaises(SystemExit):
            builder.main(['sft', '1', '2', '3', '--data-root', str(self.root),
                          '--out', str(self.root / 'sft'), '--reuse-existing', '--skip-actor-snapshots'])
        self.assertIn('lacks actor snapshots', error.getvalue())


if __name__ == '__main__':
    unittest.main()
