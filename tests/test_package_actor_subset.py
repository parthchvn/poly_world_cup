import copy
import gzip
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import zipfile

from tests.test_interval_decision_data import IntervalDatasetTests, builder, data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import package_actor_subset as pack


class PackageActorSubsetTests(unittest.TestCase):
    def setUp(self):
        self.fixture = IntervalDatasetTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.inputs = self.fixture.inputs
        quiet = patch('sys.stdout', new_callable=io.StringIO)
        quiet.start()
        self.addCleanup(quiet.stop)
        for number, source in enumerate(self.fixture.sources):
            original = next((source / 'actors').glob('*.jsonl'))
            rows = self.fixture.rows(original)
            actor = '0x' + f'{100 + number:040x}'
            for row in rows:
                row['actor_id'] = actor
            extra = source / 'actors' / (actor + '.jsonl.gz')
            extra.write_bytes(gzip.compress((''.join(json.dumps(r) + '\n' for r in rows)).encode()))
            self.reindex(source)

    def reindex(self, source):
        manifest = json.loads((source / 'manifest.json').read_text())
        manifest['source']['elapsed_seconds'] = 0.5
        source_info = pack.metrics.discover_exports([source], None)[0]
        counts = pack.Counter()
        entries = []
        for path in sorted((source / 'actors').glob('*.jsonl*')):
            actor, _, actual = pack.metrics.actor_trade_groups(path, source_info)
            actual['news_entries'] = sum(len(r.get('news', [])) for r in pack.metrics.iter_jsonl(path))
            counts.update(actual)
            entries.append({'actor_id': actor, 'path': 'actors/' + path.name,
                            **{k: v for k, v in actual.items() if k != 'actors'}})
        manifest['counts'] = dict(counts)
        (source / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.fixture.jsonl(source / 'actor_index.jsonl', entries)

    def test_zip_has_exact_whole_histories_and_builds_future_intervals(self):
        out = self.root / 'package.zip'
        receipt = pack.package(self.inputs, out, target=30)
        self.assertEqual(receipt['rows'], 30)
        self.assertEqual(receipt['actor_market_histories'], 5)
        self.assertEqual(receipt['markets'], 3)
        extracted = self.root / 'extracted'
        with zipfile.ZipFile(out) as archive:
            self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))
            archive.extractall(extracted)
        for group in receipt['groups']:
            path = extracted / group['path']
            original = self.inputs / Path(group['path']).relative_to('exports')
            self.assertEqual(path.read_bytes(), original.read_bytes())
            self.assertEqual(pack.digest(path), group['sha256'])
        for source in self.fixture.sources:
            target = extracted / 'exports' / source.name
            self.assertEqual((target / 'espn_events.jsonl').read_bytes(), (source / 'espn_events.jsonl').read_bytes())
            self.assertEqual((target / 'market_price_history.jsonl').read_bytes(), (source / 'market_price_history.jsonl').read_bytes())
            self.assertEqual((target / 'source_manifest.json').read_bytes(), (source / 'manifest.json').read_bytes())
            self.assertEqual(json.loads((target / 'manifest.json').read_text())['source']['elapsed_seconds'], 0.5)
        result = data.prepare_interval_dataset(extracted / 'exports', self.root / 'prepared',
                                              window_seconds=60, match_minutes=6)
        self.assertTrue((self.root / 'prepared' / 'train.jsonl').is_file())
        self.assertTrue(result)

    def test_over_cap_actor_is_excluded_without_truncating(self):
        source = self.fixture.sources[0]
        market = json.loads((source / 'market.json').read_text())
        manifest = json.loads((source / 'manifest.json').read_text())
        manifest['max_trades_per_actor'] = None
        (source / 'manifest.json').write_text(json.dumps(manifest))
        kickoff = data.activity.timestamp_us(market['kickoff_utc'])
        actor = '0x' + 'f' * 40
        trades = [{'time_us': kickoff + i * 1_000_000, 'time': data.activity.utc(kickoff + i * 1_000_000),
                   'trade': {'side': 'BUY', 'outcome': 'Yes', 'shares': '1', 'price': '0.4'}} for i in range(21)]
        rows = list(builder.actor_records(actor, trades, market, [], [], kickoff - 3600_000_000))
        self.fixture.jsonl(source / 'actors' / (actor + '.jsonl'), rows)
        self.reindex(source)
        result = pack.package(self.inputs, self.root / 'package.zip', target=36)
        self.assertEqual(result['rows'], 36)
        self.assertNotIn(actor, {g['actor_id'] for g in result['groups']})

    def test_infeasible_exact_total_publishes_nothing(self):
        out = self.root / 'package.zip'
        with self.assertRaisesRegex(ValueError, 'exact row target'):
            pack.package(self.inputs, out, target=8)
        self.assertFalse(out.exists())

    def test_corrupt_selected_actor_publishes_nothing(self):
        source = self.fixture.sources[0]
        path = next((source / 'actors').glob('*.jsonl'))
        rows = self.fixture.rows(path)
        rows[0]['row_index'] = 99
        self.fixture.jsonl(path, rows)
        out = self.root / 'package.zip'
        with self.assertRaisesRegex(ValueError, 'row_index'):
            pack.package(self.inputs, out, target=36)
        self.assertFalse(out.exists())

    def test_rejects_missing_shared_news_and_preserves_existing_zip(self):
        out = self.root / 'package.zip'
        out.write_bytes(b'existing')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            pack.package(self.inputs, out, target=30)
        self.assertEqual(out.read_bytes(), b'existing')
        (self.fixture.sources[0] / 'espn_events.jsonl').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing/unsafe'):
            pack.package(self.inputs, self.root / 'another.zip', target=30)

    def test_selection_is_repeatable_and_never_reuses_a_history(self):
        candidates = [{'market_id': '1', 'entry': {'actor_id': str(i), 'rows': weight}}
                      for i, weight in enumerate([2, 4, 6, 10, 12])]
        selected = pack.choose(candidates, 18, 42)
        self.assertEqual(sum(x['entry']['rows'] for x in selected), 18)
        self.assertEqual(len(selected), len({x['entry']['actor_id'] for x in selected}))
        self.assertEqual(selected, pack.choose(copy.deepcopy(candidates), 18, 42))


if __name__ == '__main__':
    unittest.main()
