import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from poly_world_cup.attribution import build_attribution_index
from scripts.filter_wallet_activity import filter_database
from scripts.complete_tournament_wallets import complete_tournament_database

C1='0x'+'1'*64
C2='0x'+'2'*64
REGISTRY={'fixtures':[{'fixture_id':'fixture:1'},{'fixture_id':'fixture:2'}],
 'contracts':[{'condition_id':c,'fixture_id':f,'selection':'home','tokens':[{'token_id':'123','outcome':'Yes'},{'token_id':'456','outcome':'No'}]}
 for c,f in [(C1,'fixture:1'),(C2,'fixture:2')]]}

class CompletedTournamentTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  self.original=self.root/'original.sqlite';self.source=self.root/'subset.sqlite'
  self.fresh=self.root/'fresh.sqlite';self.output=self.root/'completed.sqlite'
  self.old=[]
  for wallet,condition,count in [('keep',C1,2),('keep',C2,2),('crosses20',C1,10),('crosses20',C2,9),('excluded',C1,20),('excluded',C2,1)]:
   for i in range(count): self.old.append(self.row(len(self.old)+1,wallet,condition))
  self.fresh_rows=[dict(row) for row in self.old if row['condition_id']==C2]
  self.fresh_rows += [self.row(100,'keep',C2),self.row(101,'crosses20',C2),self.row(102,'new',C2)]
  self.build(self.original,self.old,'old')
  with sqlite3.connect(self.original) as db:
   d=json.loads(db.execute("select value_json from metadata where key='report'").fetchone()[0]);d.update(partial_api_collection=True,api_traversals_exhausted=1,registry_condition_count=2)
   db.execute("update metadata set value_json=? where key='report'",(json.dumps(d),))
  filter_database(self.original,self.source)
  self.build(self.fresh,self.fresh_rows,'fresh')
  self.checkpoint={'unfinished_condition_ids':[C2],'scope':{'contracts':2},'trades':{'requested_minimum_size_tokens':'0.000001','taker_only':False,'saved_observations':len(self.old),'contract_status_counts':{'exhausted':1,'paused':1}},'integrity':{'all_saved_pages_replayed_from_raw_captures':True}}
  self.collection={'conditions':[{'condition_id':C2,'status':'exhausted','integrity_validated':True}], 'status':'api_exhausted','requested_minimum_size_tokens':'0.000001','taker_only':False,'validated_observation_count':len(self.fresh_rows)}
  self.original_provenance={'raw_provenance_verified':True,'api_exhausted_conditions':1,'verified_condition_count':2,'verified_observation_count':len(self.old),'condition_manifest_sha256':{C1:'1'*64,C2:'2'*64}}
  self.provenance={'raw_provenance_verified':True,'verified_condition_count':1,'api_exhausted_conditions':1,'condition_selection':'explicit','requested_condition_count':1,'verified_observation_count':len(self.fresh_rows),'verified_raw_page_count':1,'condition_manifest_sha256':{C2:'3'*64}}
  self.collection['condition_manifest_sha256']={C2:'3'*64}
  self.provenance['verified_condition_ids']=[C2]
  self.provenance['verified_pages']=self.latest_fresh_ledger
 def tearDown(self): self.temp.cleanup()
 def row(self,i,wallet,condition):
  return {'observation_id':f'obs:{i}','condition_id':condition,'token_id':'123','side':'BUY','size':'1','price':'0.5','proxy_wallet':wallet,'block_timestamp':'2026-06-10T12:00:00Z','transaction_hash':f'tx:{i}'}
 def build(self,path,rows,label):
  page=self.root/(label+'.jsonl');page.write_text(''.join(json.dumps(r)+'\n' for r in rows))
  build_attribution_index(registry=REGISTRY,trade_pages=[page],news_records=[],output_path=path,
    expected_page_hashes={page:hashlib.sha256(page.read_bytes()).hexdigest()},expected_page_row_counts={page:len(rows)})
  if label=='fresh':
   self.latest_fresh_ledger=[{'path':str(page.resolve()),'uncompressed_sha256':hashlib.sha256(page.read_bytes()).hexdigest(),'row_count':len(rows),'condition_id':C2}]
   if hasattr(self,'provenance'):self.provenance['verified_pages']=self.latest_fresh_ledger
 def run_completion(self,**kwargs):
  return complete_tournament_database(self.source,self.fresh,self.output,original_checkpoint=self.checkpoint,original_provenance=self.original_provenance,fresh_collection=self.collection,fresh_provenance=self.provenance,**kwargs)
 def test_counts_replaced_global_threshold_recomputed_sources_unchanged(self):
  hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.source,self.fresh)]
  report=self.run_completion()
  self.assertEqual(report['net_added_observations'],3)
  self.assertEqual(report['api_traversals_exhausted'],2)
  self.assertFalse(report['partial_api_collection']);self.assertFalse(report['source_completeness_certified'])
  self.assertEqual(report['old_retained_observations_checked_for_multiset_containment'],12)
  with sqlite3.connect(self.output) as db:
   self.assertEqual(db.execute('select wallet,count(*) from trades group by wallet order by wallet').fetchall(),[('keep',5),('new',1)])
   self.assertEqual(db.execute("select observed_count from wallet_counts where wallet='crosses20'").fetchone()[0],20)
   self.assertEqual(db.execute('select count(*) from source_pages').fetchone()[0],2)
   self.assertEqual(db.execute('pragma foreign_key_check').fetchall(),[])
   ledger=json.loads(db.execute("select value_json from metadata where key='contract_exhaustion_ledger'").fetchone()[0]);self.assertEqual([(r['condition_id'],r['provenance_scope']) for r in ledger],[(C1,'inherited'),(C2,'fresh')])
  self.assertEqual(hashes,[hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.source,self.fresh)])
 def test_same_transaction_distinct_observations_not_collapsed(self):
  rows=self.fresh_rows+[dict(self.fresh_rows[0],observation_id='extra:same-semantic')]
  self.build(self.fresh,rows,'fresh');self.collection['validated_observation_count']=len(rows);self.provenance['verified_observation_count']=len(rows)
  self.run_completion()
  with sqlite3.connect(self.output) as db:
   self.assertEqual(db.execute("select count(*) from trades where wallet='keep'").fetchone()[0],6)
 def test_lower_count_rejected(self):
  rows=[r for r in self.fresh_rows if r['proxy_wallet']!='excluded']
  self.build(self.fresh,rows,'fresh');self.collection['validated_observation_count']=len(rows);self.provenance['verified_observation_count']=len(rows)
  with self.assertRaisesRegex(ValueError,'loses observations'):self.run_completion()
  self.assertFalse(self.output.exists())
 def test_equal_count_changed_economic_observation_rejected(self):
  rows=[dict(r) for r in self.fresh_rows];rows[0]['price']='0.4'
  self.build(self.fresh,rows,'fresh')
  with self.assertRaisesRegex(ValueError,'full multiset'):self.run_completion()
 def test_lost_duplicate_multiplicity_rejected(self):
  with sqlite3.connect(self.source) as db:
   values=db.execute("select * from selected_trades where wallet='keep' and condition_id=? limit 1",(C2,)).fetchone()
   other=db.execute("select trade_row_id from selected_trades where wallet='keep' and condition_id=? order by trade_row_id desc limit 1",(C2,)).fetchone()[0]
   db.execute('update selected_trades set transaction_hash=? where trade_row_id=?',(values[12],other))
  with self.assertRaisesRegex(ValueError,'full multiset'):self.run_completion()
 def test_incomplete_unverified_wrong_threshold_rejected(self):
  for key,value in [('status','paused'),('requested_minimum_size_tokens','0.01')]:
   old=self.collection[key];self.collection[key]=value
   with self.assertRaisesRegex(ValueError,'exhaust'):self.run_completion()
   self.collection[key]=old
  self.provenance['raw_provenance_verified']=False
  with self.assertRaisesRegex(ValueError,'provenance'):self.run_completion()
 def test_unrelated_provenance_pages_rejected(self):
  self.provenance['verified_pages'][0]['uncompressed_sha256']='0'*64
  with self.assertRaisesRegex(ValueError,'raw-replayed provenance ledger'):self.run_completion()
 def test_unknown_replacement_condition_rejected(self):
  c3='0x'+'3'*64
  self.checkpoint['unfinished_condition_ids']=[c3];self.collection['conditions'][0]['condition_id']=c3
  with self.assertRaises(ValueError):self.run_completion()
 def test_output_overwrite_refused(self):
  self.output.write_text('existing')
  with self.assertRaises(FileExistsError):self.run_completion()
  self.assertEqual(self.output.read_text(),'existing')
 def test_coverage_gate(self):
  with self.assertRaisesRegex(ValueError,'104 fixtures'):self.run_completion(require_tournament_coverage=True)

if __name__=='__main__':unittest.main()
