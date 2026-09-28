import io
import unittest
from unittest.mock import patch

from tests import test_prepare_actor_sft as fixtures

builder = fixtures.builder


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ActorSFTTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.output = self.root / 'sft'
        self.quiet = patch('sys.stdout', new_callable=io.StringIO)
        self.quiet.start()
        self.addCleanup(self.quiet.stop)

    def command(self, *extra, ids=('1', '2', '3')):
        return builder.main(['sft', *ids, '--data-root', str(self.root),
                             '--out', str(self.output), *extra])

    def source(self, market_id=1, fixture=1):
        return self.fixture.source(market_id, fixture, with_market_context=True, context_version=2,
                                   with_actor_snapshots=True)

    def metadata(self, market_id, **kwargs):
        return {'espn_event_id': market_id,
                'kickoff_utc': f'2026-06-{int(market_id):02d}T17:00:00Z'}

    def collect(self, args):
        self.assertEqual(args.out, self.root / ('market_' + args.market_id))
        self.assertEqual(args.cache, self.root / 'market_actor_cache')
        self.source(int(args.market_id), int(args.market_id))

    def test_combined_collection_matches_prepare_output(self):
        with patch.object(builder, 'resolve_market', side_effect=self.metadata), \
                patch.object(builder, 'export', side_effect=self.collect) as collect:
            self.command('--http-retries', '3')
        self.assertEqual(collect.call_count, 3)
        self.assertTrue(all(call.args[0].http_retries == 3 for call in collect.call_args_list))
        builder.main(['prepare', '--input-root', str(self.root), '--out', str(self.root / 'direct')])
        for split in ('train', 'validation', 'test'):
            self.assertEqual((self.output / (split + '.jsonl')).read_bytes(),
                             (self.root / 'direct' / (split + '.jsonl')).read_bytes())

    def test_reuse_and_prepare_need_no_http_or_collection(self):
        for i in range(1, 4):
            self.source(i, i)
        with patch.object(builder, 'HttpClient', side_effect=AssertionError('HTTP forbidden')), \
                patch.object(builder, 'export', side_effect=AssertionError('collection forbidden')):
            self.command('--reuse-existing')
            builder.main(['prepare', '--input-root', str(self.root), '--out', str(self.root / 'direct')])
        self.assertTrue((self.output / 'manifest.json').is_file())

    def test_failed_collection_can_resume_without_recollecting_completed_market(self):
        def fail_second(args):
            if args.market_id == '2':
                raise OSError('simulated network failure')
            self.collect(args)
        with patch.object(builder, 'resolve_market', side_effect=self.metadata), \
                patch.object(builder, 'export', side_effect=fail_second), \
                patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
            self.command()
        self.assertTrue((self.root / 'market_1/manifest.json').is_file())
        self.assertFalse(self.output.exists())
        with patch.object(builder, 'resolve_market', side_effect=self.metadata), \
                patch.object(builder, 'export', side_effect=self.collect) as collect:
            self.command('--reuse-existing')
        self.assertEqual([call.args[0].market_id for call in collect.call_args_list], ['2', '3'])

    def test_too_few_matches_fails_before_trade_collection(self):
        with patch.object(builder, 'resolve_market', side_effect=self.metadata), \
                patch.object(builder, 'export') as collect, \
                patch('sys.stderr', new_callable=io.StringIO) as error, self.assertRaises(SystemExit):
            self.command(ids=('1', '2'))
        collect.assert_not_called()
        self.assertIn('at least 3', error.getvalue())

    def test_existing_export_requires_explicit_reuse(self):
        self.source()
        with patch('sys.stderr', new_callable=io.StringIO) as error, self.assertRaises(SystemExit):
            self.command()
        self.assertIn('--reuse-existing', error.getvalue())
        self.assertTrue((self.root / 'market_1/manifest.json').is_file())

    def test_legacy_reuse_cannot_silently_skip_new_features(self):
        self.fixture.source()
        with patch('sys.stderr', new_callable=io.StringIO) as error, self.assertRaises(SystemExit):
            self.command('--reuse-existing')
        self.assertIn('lacks official market context', error.getvalue())
        self.assertIn('SAME --cache', error.getvalue())
        self.assertFalse(self.output.exists())

    def test_wrong_existing_market_is_rejected(self):
        path, _, _ = self.source(9, 1)
        path.rename(self.root / 'market_1')
        with patch('sys.stderr', new_callable=io.StringIO) as error, self.assertRaises(SystemExit):
            self.command('--reuse-existing')
        self.assertIn('does not match requested market', error.getvalue())

    def test_reuse_cannot_silently_skip_actor_snapshots(self):
        self.fixture.source(with_market_context=True, context_version=2)
        with patch('sys.stderr', new_callable=io.StringIO) as error, self.assertRaises(SystemExit):
            self.command('--reuse-existing')
        self.assertIn('actor snapshot', error.getvalue().lower())
        self.assertFalse(self.output.exists())

    def test_two_match_mode(self):
        for i in (1, 2):
            self.source(i, i)
        result = self.command('--reuse-existing', '--train-validation-only', ids=('1', '2'))
        self.assertFalse(result['held_out_test_available'])
        self.assertEqual((self.output / 'test.jsonl').read_text(), '')

    def test_invalid_inputs_fail_before_collection(self):
        for ids, extra in [(('1', '1', '2'), []), (('1', '2', '3'), ['--espn-file', 'match.json'])]:
            with self.subTest(ids=ids, extra=extra), patch.object(builder, 'export') as collect, \
                    patch('sys.stderr', new_callable=io.StringIO), self.assertRaises(SystemExit):
                self.command(*extra, ids=ids)
            collect.assert_not_called()

    def test_legacy_collection_options_still_work(self):
        with patch.object(builder, 'export') as collect:
            builder.main(['1897059', '--out', str(self.root / 'actor'), '--http-transport', 'curl'])
        args = collect.call_args.args[0]
        self.assertEqual(args.market_id, '1897059')
        self.assertEqual(args.out, self.root / 'actor')
        self.assertEqual(args.http_transport, 'curl')
        self.assertEqual(args.max_trades_per_actor, 20)


if __name__ == '__main__':
    unittest.main()
