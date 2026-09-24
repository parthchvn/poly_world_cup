"""Regression guards for cohort recovery, query binding and atomic raw replay."""
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from poly_world_cup.http import FetchResult, request_url
from poly_world_cup.trades import TradeIngestionError, ingest_condition
from scripts.recover_sequence_trades import SCHEMA, import_condition, verify_selected_counts, inherit_archive, sha256

CONDITION='0x'+'a'*64
WALLET='0x'+'b'*40


def trade(**updates):
    return {'proxy_wallet':WALLET,'condition_id':CONDITION,'token_id':'1234',
            'side':'BUY','size':'0.000001','price':'0.4399999648',
            'timestamp':1780000000,'transaction_hash':'0x'+'c'*64,**updates}


def page(rows,cursor=None):
    return {'data':rows,'pagination':{'has_more':cursor is not None,'next_cursor':cursor}}


class CachedClient:
    def __init__(self,path,pages):
        self.path=path;self.pages=list(pages)

    def get_json(self,url,params=None):
        url=request_url(url,params);body=json.dumps(self.pages.pop(0)).encode()
        digest=hashlib.sha256(body).hexdigest();when='2026-09-24T12:00:00Z'
        (self.path/'requests').mkdir(parents=True,exist_ok=True)
        (self.path/'bodies').mkdir(parents=True,exist_ok=True)
        (self.path/'bodies'/(digest+'.json.gz')).write_bytes(gzip.compress(body,mtime=0))
        metadata={'url':url,'body_sha256':digest,'retrieved_at':when,'body_compression':'gzip'}
        (self.path/'requests'/(hashlib.sha256(url.encode()).hexdigest()+'.json')).write_text(json.dumps(metadata))
        return FetchResult(json.loads(body,parse_float=Decimal),url,when,digest,False)


class SourceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name);self.captures=self.path/'trades';self.cache=self.path/'cache'
        self.db=sqlite3.connect(':memory:',uri=True);self.db.executescript(SCHEMA);self.addCleanup(self.db.close)

    def capture(self,pages,**kwargs):
        return ingest_condition(CachedClient(self.cache/CONDITION,pages),condition_id=CONDITION,
                                output_dir=self.captures,compress=True,minimum_size='0.000001',**kwargs)

    def ledger(self,rows):
        counts=Counter(r['proxy_wallet'] for r in rows)
        self.db.executemany('INSERT INTO wallet_market_counts VALUES(?,?,?)',[(w,CONDITION,n) for w,n in counts.items()]);self.db.commit()

    def replay(self):
        import_condition(self.db,CONDITION,self.captures,self.cache,{'1234','5678'})

    def test_inclusive_twenty_and_repeated_rows_preserve_decimal_identity(self):
        rows=[trade() for _ in range(20)]+[trade(proxy_wallet='0x'+'d'*40) for _ in range(21)]+[trade(proxy_wallet='0x'+'e'*40)]
        self.ledger(rows);self.capture([page(rows)]);self.replay()
        result=verify_selected_counts(self.db)
        self.assertEqual(result['selected_observation_count'],21)
        self.assertEqual(self.db.execute('SELECT COUNT(DISTINCT observation_id) FROM trades').fetchone()[0],21)
        self.assertEqual(self.db.execute('SELECT DISTINCT price FROM trades').fetchall(),[('0.4399999648',)])
        self.assertEqual(self.db.execute('SELECT full_count FROM recovery_conditions').fetchone()[0],42)

    def test_cursor_pages_and_unchanged_replay_resume_are_idempotent(self):
        rows=[trade(),trade(timestamp=1779999999)]
        self.ledger(rows);self.capture([page(rows[:1],'next'),page(rows[1:])]);self.replay();self.replay()
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM trades').fetchone()[0],2)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM source_pages').fetchone()[0],2)

    def test_paused_capture_cannot_supply_complete_labels(self):
        self.ledger([trade()]);self.capture([page([trade()],'next')],max_pages=1)
        with self.assertRaisesRegex(TradeIngestionError,'nonexhausted'):self.replay()
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM trades').fetchone()[0],0)

    def test_same_total_count_but_wrong_wallet_is_rejected(self):
        self.ledger([trade()]);self.capture([page([trade(proxy_wallet='0x'+'e'*40)])])
        with self.assertRaisesRegex(TradeIngestionError,'count ledger'):self.replay()
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM source_pages').fetchone()[0],0)

    def test_raw_query_binding_rejects_other_query_even_if_normalized_page_valid(self):
        self.ledger([trade()]);self.capture([page([trade()])])
        path=self.captures/CONDITION/'manifest.json';state=json.loads(path.read_text())
        state['pages'][0]['request_url']+='&user=another';path.write_text(json.dumps(state))
        with self.assertRaisesRegex(TradeIngestionError,'intended query'):self.replay()

    def test_corrupt_later_raw_page_rolls_back_entire_contract(self):
        rows=[trade(),trade(timestamp=1779999999)];self.ledger(rows)
        manifest=self.capture([page(rows[:1],'next'),page(rows[1:])])
        path=self.cache/CONDITION/'bodies'/(manifest['pages'][1]['body_sha256']+'.json.gz')
        path.write_bytes(gzip.compress(b'{}'))
        with self.assertRaisesRegex(TradeIngestionError,'checksum'):self.replay()
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM trades').fetchone()[0],0)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM source_pages').fetchone()[0],0)

    def test_unknown_token_is_rejected_atomically(self):
        rows=[trade(),trade(token_id='9999')];self.ledger(rows);self.capture([page(rows)])
        with self.assertRaisesRegex(TradeIngestionError,'Unknown token'):self.replay()
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM trades').fetchone()[0],0)

    def test_pair_validator_rejects_missing_pair(self):
        self.ledger([trade()])
        with self.assertRaisesRegex(TradeIngestionError,'Selected pair counts'):verify_selected_counts(self.db)

    def test_inherited_archive_readonly_uri_and_count_binding(self):
        archive=self.path/'world_cup_lt20.sqlite'
        old=sqlite3.connect(archive)
        old.executescript(SCHEMA)
        old.execute('CREATE TABLE selected_trades AS SELECT * FROM trades')
        old.execute('INSERT INTO wallet_market_counts VALUES(?,?,?)',(WALLET,CONDITION,1))
        old.execute("INSERT INTO source_pages VALUES(1,'oldpage','digest',1,'inherited',NULL,NULL,NULL,NULL)")
        old.execute('INSERT INTO selected_trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (44,'old-observation',WALLET,CONDITION,1780000000000000,'2026-05-28T20:26:40Z','0x'+'c'*64,'1234','BUY','1','0.4',1,1))
        old.commit();old.close()
        (self.path/'MANIFEST.json').write_text(json.dumps({'members':[{'file':archive.name,'bytes':archive.stat().st_size,'sha256':sha256(archive)}]}))
        self.ledger([trade()]);before=sha256(archive)
        report=inherit_archive(self.db,archive,set(),self.path)
        self.assertEqual(report['inherited_observation_count'],1)
        self.assertEqual(sha256(archive),before)
        self.assertEqual(verify_selected_counts(self.db)['selected_pair_count'],1)

    def test_imported_manifest_change_is_not_silently_resumed(self):
        self.ledger([trade()]);self.capture([page([trade()])]);self.replay()
        path=self.captures/CONDITION/'manifest.json';state=json.loads(path.read_text())
        state['extra_audit_field']='changed';path.write_text(json.dumps(state))
        with self.assertRaisesRegex(TradeIngestionError,'manifest changed'):self.replay()


if __name__=='__main__':unittest.main()
