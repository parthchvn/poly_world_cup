import base64
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from poly_world_cup.news_archive import ArchiveCollector, archive_version, parse_cdx, verify_replay
from poly_world_cup.io import atomic_write, write_json


class NewsArchiveTests(unittest.TestCase):
    def setUp(self):
        self.raw = b'<html><title>fallback</title><meta property="og:title" content="Scotland World Cup injury &amp; update"></html>'
        self.capture = {'timestamp': '20260601024726', 'original': 'https://www.espn.com/soccer/story/_/id/1/a',
                        'mimetype': 'text/html', 'statuscode': '200',
                        'digest': base64.b32encode(hashlib.sha1(self.raw).digest()).decode()}
        self.url = 'https://web.archive.org/web/20260601024726id_/' + self.capture['original']
        self.headers = {'Content-Type': 'text/html', 'Memento-Datetime': 'Mon, 01 Jun 2026 02:47:26 GMT',
                        'Link': '<' + self.capture['original'] + '>; rel="original"'}

    def test_verified_replay_extracts_archived_headline(self):
        body, title = verify_replay(self.capture, self.url, self.headers, self.raw)
        self.assertEqual(body, self.raw)
        self.assertEqual(title, 'Scotland World Cup injury & update')

    def test_redirect_to_later_capture_rejected(self):
        with self.assertRaisesRegex(ValueError, 'redirected'):
            verify_replay(self.capture, self.url.replace('20260601024726', '20260901024726'), self.headers, self.raw)

    def test_forged_memento_timestamp_rejected(self):
        headers = {**self.headers, 'Memento-Datetime': 'Mon, 01 Jun 2026 02:47:27 GMT'}
        with self.assertRaisesRegex(ValueError, 'timestamp mismatch'):
            verify_replay(self.capture, self.url, headers, self.raw)

    def test_wrong_original_rejected(self):
        headers = {**self.headers, 'Link': '<https://example.com/other>; rel="original"'}
        with self.assertRaisesRegex(ValueError, 'original URL mismatch'):
            verify_replay(self.capture, self.url, headers, self.raw)

    def test_body_digest_must_match_index(self):
        with self.assertRaisesRegex(ValueError, 'does not match CDX digest'):
            verify_replay(self.capture, self.url, self.headers, self.raw + b'changed')

    def test_transport_compression_digest_supported(self):
        raw = gzip.compress(self.raw)
        capture = {**self.capture, 'digest': base64.b32encode(hashlib.sha1(raw).digest()).decode()}
        headers = {**self.headers, 'Content-Encoding': 'gzip'}
        body, _ = verify_replay(capture, self.url, headers, raw)
        self.assertEqual(body, self.raw)

    def test_cdx_excludes_later_or_wrong_source(self):
        keys = ['timestamp', 'original', 'mimetype', 'digest', 'statuscode']
        later = {**self.capture, 'timestamp': '20260801000000'}
        wrong = {**self.capture, 'original': 'https://example.com/other'}
        rows = [keys] + [[row[k] for k in keys] for row in (later, wrong, self.capture)]
        self.assertEqual(parse_cdx(rows, self.capture['original'], '20260719235959'), [self.capture])

    def test_current_headline_never_promoted_by_archive(self):
        article = {'news_id': 'current-version', 'source_article_id': '1', 'title': 'New September information',
                   'published_at_utc': '20260530160000', 'fixture_ids': ['espn:2']}
        replay = {'headline': 'Historical World Cup headline', 'archive_url': self.url,
                  'body_sha256': hashlib.sha256(self.raw).hexdigest(), 'retrieved_at': '2026-09-23T00:00:00Z'}
        row = archive_version(article, self.capture, replay)
        self.assertEqual(row['title'], replay['headline'])
        self.assertEqual(row['availability_upper_utc'], '2026-06-01T02:47:26Z')
        self.assertIsNone(row['body'])
        self.assertFalse(row['fixture_links'][0]['historical_link_verified'])
        self.assertEqual(row['context_scope'], 'tournament')
        self.assertTrue(row['historical_tournament_scope_verified'])

    def test_direct_game_url_verifies_only_that_fixture(self):
        capture = {**self.capture, 'original': 'https://www.espn.com/soccer/report/_/gameId/760415'}
        replay = {'headline': 'Mexico report', 'archive_url': self.url, 'body_sha256': 'a' * 64, 'retrieved_at': '2026-09-23T00:00:00Z'}
        row = archive_version({'fixture_ids': ['espn:760415', 'espn:999']}, capture, replay)
        links = {link['fixture_id']: link for link in row['fixture_links']}
        self.assertTrue(links['espn:760415']['historical_link_verified'])
        self.assertFalse(links['espn:999']['historical_link_verified'])
        self.assertFalse(row['historical_tournament_scope_verified'])

    def test_unknown_game_url_does_not_create_registry_fixture(self):
        capture = {**self.capture, 'original': 'https://www.espn.com/soccer/report/_/gameId/401123456'}
        replay = {'headline': 'Old report', 'archive_url': self.url, 'body_sha256': 'a' * 64, 'retrieved_at': '2026-09-23T00:00:00Z'}
        row = archive_version({'fixture_ids': []}, capture, replay)
        self.assertEqual(row['fixture_ids'], [])
        self.assertEqual(row['fixture_links'], [])


    def _checkpoint(self, directory):
        collector = ArchiveCollector(directory, interval=0)
        article = {'news_id': 'current', 'source_article_id': '1', 'source_url': self.capture['original'], 'fixture_ids': []}
        body_hash = hashlib.sha256(self.raw).hexdigest()
        replay = {'archive_url': self.url, 'body_sha256': body_hash, 'headline': 'Scotland World Cup injury & update',
                  'retrieved_at': '2026-09-23T00:00:00Z', 'capture_timestamp': self.capture['timestamp'], 'cdx_digest': self.capture['digest'], 'response_headers': self.headers, 'payload_sha256': body_hash}
        replay_key = hashlib.sha256(self.url.encode()).hexdigest()
        write_json(directory / 'replays' / f'{replay_key}.json', replay)
        atomic_write(directory / 'html_cache' / f'{body_hash}.html.gz', gzip.compress(self.raw))
        atomic_write(directory / 'payload_cache' / f'{body_hash}.bin.gz', gzip.compress(self.raw))
        fields = ['timestamp', 'original', 'mimetype', 'digest', 'statuscode']
        cdx = json.dumps([fields, [self.capture[k] for k in fields]]).encode()
        cdx_hash = hashlib.sha256(cdx).hexdigest()
        atomic_write(directory / 'cdx_cache' / 'bodies' / f'{cdx_hash}.json.gz', gzip.compress(cdx))
        version = archive_version(article, self.capture, replay)
        key = hashlib.sha256((article['source_url'] + collector.start + collector.cutoff).encode()).hexdigest()
        checkpoint = directory / 'articles' / f'{key}.json'
        saved = {'source_url': article['source_url'], 'current_news_id': 'current', 'status': 'verified_headline',
                 'cdx_body_sha256': cdx_hash, 'versions': [version]}
        write_json(checkpoint, saved)
        return collector, article, checkpoint, saved, cdx_hash

    def test_resume_rebuilds_backdated_checkpoint_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector, article, checkpoint, saved, _ = self._checkpoint(Path(temporary))
            saved['versions'][0]['availability_upper_utc'] = '2020-01-01T00:00:00Z'
            saved['versions'][0]['availability_evidence'][0]['captured_at_utc'] = '2020-01-01T00:00:00Z'
            write_json(checkpoint, saved)
            restored = collector.collect_article(article)['versions'][0]
            self.assertEqual(restored['availability_upper_utc'], '2026-06-01T02:47:26Z')
            self.assertEqual(restored['availability_evidence'][0]['captured_at_utc'], '2026-06-01T02:47:26Z')

    def test_resume_rejects_capture_not_in_original_cdx(self):
        with tempfile.TemporaryDirectory() as temporary:
            collector, article, checkpoint, saved, _ = self._checkpoint(Path(temporary))
            saved['versions'][0]['archive_capture_timestamp'] = '20200101000000'
            write_json(checkpoint, saved)
            with self.assertRaisesRegex(ValueError, 'no matching cached CDX proof'):
                collector.collect_article(article)

    def test_resume_rejects_corrupt_cdx_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            collector, article, _, _, digest = self._checkpoint(directory)
            atomic_write(directory / 'cdx_cache' / 'bodies' / f'{digest}.json.gz', gzip.compress(b'[]'))
            with self.assertRaisesRegex(ValueError, 'CDX checkpoint cache integrity failure'):
                collector.collect_article(article)


    def test_resume_rejects_replay_repointed_to_other_cached_article(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            collector, _, _, _, _ = self._checkpoint(directory)
            metadata_path = directory / 'replays' / (hashlib.sha256(self.url.encode()).hexdigest() + '.json')
            metadata = json.loads(metadata_path.read_text())
            wrong = b'<html><meta property="og:title" content="Later World Cup result"></html>'
            wrong_hash = hashlib.sha256(wrong).hexdigest()
            atomic_write(directory / 'html_cache' / f'{wrong_hash}.html.gz', gzip.compress(wrong))
            metadata['body_sha256'] = wrong_hash
            metadata['headline'] = 'Later World Cup result'
            write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, 'headline cache integrity failure'):
                collector.replay(self.capture)

    def test_resume_rechecks_memento_headers(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            collector, _, _, _, _ = self._checkpoint(directory)
            metadata_path = directory / 'replays' / (hashlib.sha256(self.url.encode()).hexdigest() + '.json')
            metadata = json.loads(metadata_path.read_text())
            metadata['response_headers']['Memento-Datetime'] = 'Mon, 01 Jun 2026 03:47:26 GMT'
            write_json(metadata_path, metadata)
            with self.assertRaisesRegex(ValueError, 'timestamp mismatch'):
                collector.replay(self.capture)


    def test_other_world_cup_edition_remains_candidate(self):
        replay = {'headline': 'Spain will host 2030 World Cup final, federation says', 'archive_url': self.url,
                  'body_sha256': hashlib.sha256(self.raw).hexdigest(), 'retrieved_at': '2026-09-23T00:00:00Z'}
        row = archive_version({'source_article_id': '1'}, self.capture, replay)
        self.assertTrue(row['historical_availability_verified'])
        self.assertEqual(row['context_scope'], 'candidate')
        self.assertFalse(row['historical_tournament_scope_verified'])


if __name__ == '__main__':
    unittest.main()
