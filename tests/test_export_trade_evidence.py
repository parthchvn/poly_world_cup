import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import unittest

from tests import test_complete_tournament_wallets as fixtures
from scripts.export_trade_evidence import export_trade_evidence, Shards


class EvidenceExportTests(unittest.TestCase):
 def setUp(self):
  self.case=fixtures.CompletedTournamentTests();self.case.setUp();self.case.run_completion()
  self.output=self.case.root/'evidence'
 def tearDown(self):self.case.tearDown()
 def test_full_counts_include_excluded_wallets_readback_recomputes_threshold(self):
  before=hashlib.sha256(self.case.output.read_bytes()).hexdigest()
  report=export_trade_evidence(self.case.output,self.output,reports={})
  rows=[json.loads(line) for path in sorted((self.output/'trade_count_ledger').glob('*.gz')) for line in gzip.open(path,'rt')]
  self.assertIn('excluded',{row['wallet'] for row in rows})
  self.assertEqual(report['source_observations'],47)
  self.assertEqual(report['retained_observations'],6)
  self.assertEqual(report['retained_wallets'],2)
  self.assertTrue(report['gzip_crc_and_full_readback_verified'])
  self.assertFalse(report['inherited_raw_pages_included'])
  self.assertEqual(before,hashlib.sha256(self.case.output.read_bytes()).hexdigest())
 def test_bytes_deterministic_and_existing_output_refused(self):
  other=self.case.root/'evidence2'
  export_trade_evidence(self.case.output,self.output,reports={'custom':{'ok':True}})
  export_trade_evidence(self.case.output,other,reports={'custom':{'ok':True}})
  for file in self.output.rglob('*'):
   if file.is_file():self.assertEqual(file.read_bytes(),(other/file.relative_to(self.output)).read_bytes())
  with self.assertRaises(FileExistsError):export_trade_evidence(self.case.output,self.output,reports={})
 def test_frozen_report_bytes_preserve_upstream_checkpoint_hashes(self):
  frozen=self.case.root/'frozen.json';frozen.write_bytes(b'{"z": 1, "a": 2}\n')
  export_trade_evidence(self.case.output,self.output,reports={'frozen':frozen})
  self.assertEqual(frozen.read_bytes(),(self.output/'frozen.json').read_bytes())
 def test_tampered_count_ledger_fails_without_publication(self):
  with sqlite3.connect(self.case.output) as db:db.execute("UPDATE wallet_market_counts SET observation_count=observation_count+1 WHERE wallet='keep' AND condition_id=?",(fixtures.C2,))
  with self.assertRaisesRegex(ValueError,'Ledger totals'):export_trade_evidence(self.case.output,self.output,reports={})
  self.assertFalse(self.output.exists())
 def test_shards_are_bounded_and_crc_readable(self):
  writer=Shards(self.case.root/'shards','counts',1024)
  rows=[{'wallet':str(i),'condition_id':'x'*80,'observation_count':i+1} for i in range(30)]
  for row in rows:writer.write(row)
  writer.close();self.assertGreater(len(writer.files),1)
  recovered=[]
  for info in writer.files:
   path=self.case.root/'shards'/info['path'];self.assertLess(path.stat().st_size,1024)
   with gzip.open(path,'rt') as stream:recovered.extend(json.loads(line) for line in stream)
  self.assertEqual(recovered,rows)

if __name__=='__main__':unittest.main()
