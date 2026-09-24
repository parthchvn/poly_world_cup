#!/usr/bin/env python3
"""Recover the <=20 cohort from preserved normalized data and verified API captures.

The old archive is immutable inherited evidence, not a new raw replay. Nineteen
replacement contracts are replayed from complete fresh condition traversals;
exactly-20 pairs in other contracts are separately replayed from user queries.
All selected pair counts must equal the complete published v2 ledger. Coverage
bounds deliberately describe only the narrower selected-observation interval.
Build in private temporary storage, close/check the DB, then atomically publish.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poly_world_cup.http import FetchResult, request_url
from poly_world_cup.io import write_json
from poly_world_cup.trades import API_URL, TradeIngestionError, _json_bytes, _validated_page, validate_collection

SCHEMA = """
CREATE TABLE trades(
 trade_row_id INTEGER PRIMARY KEY, observation_id TEXT NOT NULL,
 wallet TEXT NOT NULL, condition_id TEXT NOT NULL, query_us INTEGER NOT NULL,
 block_timestamp TEXT NOT NULL, transaction_hash TEXT NOT NULL,
 token_id TEXT NOT NULL, side TEXT NOT NULL, shares TEXT NOT NULL, price TEXT NOT NULL,
 source_page_id INTEGER NOT NULL, source_line INTEGER NOT NULL,
 UNIQUE(source_page_id,source_line));
CREATE TABLE wallet_market_counts(wallet TEXT NOT NULL,condition_id TEXT NOT NULL,
 observation_count INTEGER NOT NULL CHECK(observation_count>0),
 PRIMARY KEY(wallet,condition_id)) WITHOUT ROWID;
CREATE INDEX counts_condition ON wallet_market_counts(condition_id,wallet);
CREATE TABLE source_pages(source_page_id INTEGER PRIMARY KEY,path TEXT NOT NULL UNIQUE,
 uncompressed_sha256 TEXT NOT NULL,row_count INTEGER NOT NULL,
 provenance_scope TEXT NOT NULL,request_url TEXT,body_sha256 TEXT,retrieved_at TEXT,
 raw_cache_path TEXT);
CREATE TABLE condition_coverage(condition_id TEXT PRIMARY KEY,fixture_id TEXT NOT NULL,
 status TEXT NOT NULL,observation_count INTEGER NOT NULL,page_count INTEGER NOT NULL,
 earliest_query_us INTEGER,latest_query_us INTEGER,capture_started_at TEXT,
 capture_finished_at TEXT,manifest_sha256 TEXT,selected_observation_count INTEGER NOT NULL,
 coverage_bound_scope TEXT NOT NULL);
CREATE TABLE metadata(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);
CREATE TABLE recovery_conditions(condition_id TEXT PRIMARY KEY,manifest_sha256 TEXT NOT NULL,
 full_count INTEGER NOT NULL,selected_count INTEGER NOT NULL,details_json TEXT NOT NULL);
"""
COLUMNS = ('observation_id,wallet,condition_id,query_us,block_timestamp,transaction_hash,'
           'token_id,side,shares,price,source_page_id,source_line')
INSERT = 'INSERT INTO trades('+COLUMNS+') VALUES('+','.join('?' for _ in range(12))+')'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')


def read_raw(cache, page, expected_url):
    """Bind immutable bytes to the exact intended query, not just a valid hash."""
    if page['request_url'] != expected_url:
        raise TradeIngestionError('Saved request URL differs from intended query')
    key = hashlib.sha256(expected_url.encode()).hexdigest()
    metadata = json.loads((Path(cache)/'requests'/f'{key}.json').read_text())
    if any(metadata.get(k) != v for k,v in (
        ('url',expected_url),('body_sha256',page['body_sha256']),
        ('retrieved_at',page['retrieved_at']))):
        raise TradeIngestionError('Raw cache metadata differs from saved page')
    suffix = '.json.gz' if metadata.get('body_compression') == 'gzip' else '.json'
    path = Path(cache)/'bodies'/(page['body_sha256']+suffix)
    raw = path.read_bytes()
    if suffix.endswith('gz'):
        raw = gzip.decompress(raw)
    if hashlib.sha256(raw).hexdigest() != page['body_sha256']:
        raise TradeIngestionError('Raw response checksum mismatch')
    return FetchResult(json.loads(raw,parse_float=Decimal),expected_url,
                       page['retrieved_at'],page['body_sha256'],True), path


def trade_values(row, page_id, line):
    return (row['observation_id'],row['proxy_wallet'],row['condition_id'],
            row['block_timestamp_seconds']*1000000,row['block_timestamp'],
            row['transaction_hash'],row['token_id'],row['side'],row['size'],
            row['price'],page_id,line)


def verify_selected_counts(db):
    """Exact pairwise equality, including missing pairs, overcounts and extras."""
    db.execute('DROP TABLE IF EXISTS actual_selected_counts')
    db.execute('CREATE TEMP TABLE actual_selected_counts AS SELECT wallet,condition_id,COUNT(*) n FROM trades GROUP BY wallet,condition_id')
    db.execute('CREATE UNIQUE INDEX actual_pairs ON actual_selected_counts(wallet,condition_id)')
    bad = db.execute('''SELECT c.wallet,c.condition_id,c.observation_count,a.n
        FROM wallet_market_counts c LEFT JOIN actual_selected_counts a
        ON a.wallet=c.wallet AND a.condition_id=c.condition_id
        WHERE c.observation_count<=20 AND (a.n IS NULL OR a.n!=c.observation_count) LIMIT 1''').fetchone()
    extra = db.execute('''SELECT a.wallet,a.condition_id,a.n FROM actual_selected_counts a
        LEFT JOIN wallet_market_counts c ON a.wallet=c.wallet AND a.condition_id=c.condition_id
        WHERE c.wallet IS NULL OR c.observation_count>20 LIMIT 1''').fetchone()
    if bad or extra:
        raise TradeIngestionError(f'Selected pair counts do not match complete ledger: missing/mismatch={bad}, extra={extra}')
    return {'selected_observation_count':db.execute('SELECT COUNT(*) FROM trades').fetchone()[0],
            'selected_pair_count':db.execute('SELECT COUNT(*) FROM actual_selected_counts').fetchone()[0],
            'selected_wallet_count':db.execute('SELECT COUNT(DISTINCT wallet) FROM trades').fetchone()[0]}


def import_condition(db, condition, captures, cache, valid_tokens):
    """Replay every raw page, filter by full counts, and commit a whole contract."""
    directory=Path(captures)/condition
    manifest=validate_collection(captures,condition_id=condition)
    if manifest['api_traversal_status'] != 'exhausted':
        raise TradeIngestionError('Cannot import a nonexhausted condition')
    if manifest['parameters']['filter_amount'] != '0.000001':
        raise TradeIngestionError('Unexpected source minimum-size query')
    digest=sha256(directory/'manifest.json')
    existing=db.execute('SELECT manifest_sha256 FROM recovery_conditions WHERE condition_id=?',(condition,)).fetchone()
    if existing:
        if existing[0] != digest:
            raise TradeIngestionError('Already imported manifest changed')
        return
    expected=dict(db.execute('SELECT wallet,observation_count FROM wallet_market_counts WHERE condition_id=?',(condition,)))
    counts=Counter(); selected=0; cursors=[]
    try:
        db.execute('BEGIN')
        for page in manifest['pages']:
            params=dict(manifest['parameters']); params['cursor']=page['requested_cursor']
            result,raw_path=read_raw(Path(cache)/condition,page,request_url(API_URL,params))
            normalized,nxt,more=_validated_page(result,condition,page['requested_cursor'],cursors)
            if nxt != page['next_cursor'] or more != page['has_more']:
                raise TradeIngestionError('Raw pagination differs from saved journal')
            normalized_bytes=b''.join(_json_bytes(row) for row in normalized)
            page_path=directory/page['file']
            saved=gzip.decompress(page_path.read_bytes()) if page_path.suffix=='.gz' else page_path.read_bytes()
            if normalized_bytes != saved or hashlib.sha256(saved).hexdigest()!=page['normalized_sha256']:
                raise TradeIngestionError('Raw replay differs from saved normalized observations')
            cursor=db.execute('''INSERT INTO source_pages(path,uncompressed_sha256,row_count,provenance_scope,
                request_url,body_sha256,retrieved_at,raw_cache_path) VALUES(?,?,?,?,?,?,?,?)''',
                (str(page_path.resolve()),page['normalized_sha256'],len(normalized),'fresh_condition_raw_replay',
                 result.url,result.body_sha256,result.retrieved_at,str(raw_path.resolve())))
            values=[]
            for line,row in enumerate(normalized,1):
                if row['token_id'] not in valid_tokens:
                    raise TradeIngestionError('Unknown token in captured condition')
                wallet=row['proxy_wallet']; counts[wallet]+=1
                if expected.get(wallet,21)<=20:
                    values.append(trade_values(row,cursor.lastrowid,line))
            db.executemany(INSERT,values); selected+=len(values); cursors.append(page['requested_cursor'])
        if counts != Counter(expected):
            raise TradeIngestionError('Fresh full condition count ledger differs from published v2 evidence')
        details={'full_earliest_block_timestamp':manifest['earliest_block_timestamp'],
                 'full_latest_block_timestamp':manifest['latest_block_timestamp'],
                 'page_count':len(manifest['pages']),
                 'capture_started_at':min(p['retrieved_at'] for p in manifest['pages']),
                 'capture_finished_at':max(p['retrieved_at'] for p in manifest['pages'])}
        db.execute('INSERT INTO recovery_conditions VALUES(?,?,?,?,?)',
                   (condition,digest,sum(counts.values()),selected,json.dumps(details,sort_keys=True)))
        db.commit()
    except BaseException:
        db.rollback(); raise


def load_ledger(db,evidence,conditions):
    files=[]; total=0
    for path in sorted((Path(evidence)/'trade_count_ledger').glob('*.jsonl.gz')):
        values=[]
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line)
                if row['condition_id'] not in conditions or type(row['observation_count']) is not int or row['observation_count']<=0:
                    raise TradeIngestionError('Invalid full count ledger row')
                values.append((row['wallet'],row['condition_id'],row['observation_count']))
                total+=row['observation_count']
        db.executemany('INSERT INTO wallet_market_counts VALUES(?,?,?)',values)
        files.append({'path':str(path),'sha256':sha256(path)})
    db.commit()
    return {'files':files,'observation_count':total,'pair_count':db.execute('SELECT COUNT(*) FROM wallet_market_counts').fetchone()[0]}


def inherit_archive(db,archive,replacements,evidence):
    """The archive hash binds unchanged normalized rows to prior evidence."""
    archive=Path(archive).resolve()
    manifest_path=archive.parent/'MANIFEST.json'
    manifest=json.loads(manifest_path.read_text())
    member=next(x for x in manifest['members'] if x['file']==archive.name)
    digest=sha256(archive)
    if archive.stat().st_size!=member['bytes'] or digest!=member['sha256']:
        raise TradeIngestionError('Original archive DB differs from preserved manifest')
    db.execute('ATTACH DATABASE ? AS old',('file:'+str(archive)+'?mode=ro',))
    if db.execute('PRAGMA old.quick_check').fetchone()[0] != 'ok':
        raise TradeIngestionError('Original archive SQLite failed integrity check')
    db.execute('CREATE TEMP TABLE replacement_ids(condition_id TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO replacement_ids VALUES(?)',[(c,) for c in replacements])
    mismatch=db.execute('''SELECT c.wallet,c.condition_id FROM wallet_market_counts c
        LEFT JOIN old.wallet_market_counts o ON c.wallet=o.wallet AND c.condition_id=o.condition_id
        WHERE c.condition_id NOT IN(SELECT condition_id FROM replacement_ids)
        AND (o.observation_count IS NULL OR o.observation_count!=c.observation_count) LIMIT 1''').fetchone()
    extra=db.execute('''SELECT o.wallet,o.condition_id FROM old.wallet_market_counts o
        LEFT JOIN wallet_market_counts c ON c.wallet=o.wallet AND c.condition_id=o.condition_id
        WHERE o.condition_id NOT IN(SELECT condition_id FROM replacement_ids) AND c.wallet IS NULL LIMIT 1''').fetchone()
    if mismatch or extra:
        raise TradeIngestionError('Original complete-condition counts differ from v2 ledger')
    # Retain every inherited page reference for verifiable source-line coordinates;
    # superseded/unused pages are explicitly tagged and never counted as new replay.
    db.execute('''INSERT INTO source_pages(source_page_id,path,uncompressed_sha256,row_count,provenance_scope)
        SELECT source_page_id,path,uncompressed_sha256,row_count,'inherited_normalized_archive_reference' FROM old.source_pages''')
    db.execute('INSERT INTO trades('+COLUMNS+') SELECT '+','.join('t.'+c for c in COLUMNS.split(','))+'''
        FROM old.selected_trades t JOIN wallet_market_counts c ON t.wallet=c.wallet AND t.condition_id=c.condition_id
        WHERE t.condition_id NOT IN(SELECT condition_id FROM replacement_ids) AND c.observation_count<20''')
    db.commit(); count=db.execute('SELECT COUNT(*) FROM trades').fetchone()[0]
    db.execute('DETACH DATABASE old')
    return {'archive_path':str(archive),'archive_bytes':archive.stat().st_size,'archive_sha256':digest,
            'archive_manifest_sha256':sha256(manifest_path),'inherited_observation_count':count,
            'provenance_scope':'inherited_normalized_archive_not_new_raw_replay'}


def import_exact20(db,root,cache,valid_tokens):
    # Companion collector documents each single-page user+condition query. Never
    # infer completeness from a short page without terminal pagination and counts.
    from scripts.recover_exact20_pairs import verify_saved_pair
    pairs=list(db.execute('''SELECT wallet,condition_id FROM wallet_market_counts
        WHERE observation_count=20 AND condition_id NOT IN(SELECT condition_id FROM recovery_conditions)'''))
    count=0
    with db:
        for wallet,condition in pairs:
            rows,proof=verify_saved_pair(Path(root),Path(cache),wallet,condition,valid_tokens[condition])
            page=proof['page']
            cur=db.execute('''INSERT INTO source_pages(path,uncompressed_sha256,row_count,provenance_scope,
                request_url,body_sha256,retrieved_at,raw_cache_path) VALUES(?,?,?,?,?,?,?,?)''',
                (proof['normalized_path'],page['normalized_sha256'],len(rows),'fresh_exact20_raw_replay',
                 page['request_url'],page['body_sha256'],page['retrieved_at'],proof['raw_path']))
            db.executemany(INSERT,[trade_values(row,cur.lastrowid,i) for i,row in enumerate(rows,1)])
            count+=len(rows)
    return count


def finalize(db,evidence,contracts,archive_info,ledger_info,output):
    print(now(),'creating chronological and identity indexes',flush=True)
    db.executescript('''CREATE UNIQUE INDEX trades_observation_id ON trades(observation_id);
        CREATE INDEX trades_wallet_time ON trades(wallet,query_us,observation_id,trade_row_id);
        CREATE INDEX trades_wallet_condition_time ON trades(wallet,condition_id,query_us);
        CREATE INDEX trades_condition_time ON trades(condition_id,query_us);''')
    stats=verify_selected_counts(db)
    exhaustion={x['condition_id']:x for x in json.loads((Path(evidence)/'contract_exhaustion_ledger.json').read_text())}
    bounds={r[0]:r[1:] for r in db.execute('SELECT condition_id,MIN(query_us),MAX(query_us),COUNT(*) FROM trades GROUP BY condition_id')}
    full_counts=dict(db.execute('SELECT condition_id,SUM(observation_count) FROM wallet_market_counts GROUP BY condition_id'))
    original_prov=json.loads((Path(evidence)/'original_provenance.json').read_text())
    for condition,contract in sorted(contracts.items()):
        rec=db.execute('SELECT manifest_sha256,details_json FROM recovery_conditions WHERE condition_id=?',(condition,)).fetchone()
        if rec:
            digest,details=rec[0],json.loads(rec[1]); status='recaptured_api_exhausted_selected_cohort_verified'
            start,end=details['capture_started_at'],details['capture_finished_at']; pages=details['page_count']
        else:
            if exhaustion[condition]['status']!='api_exhausted':
                raise TradeIngestionError('Inherited condition lacks exhausted evidence')
            digest=exhaustion[condition]['manifest_sha256'];status='inherited_api_exhausted_selected_cohort_verified'
            # Verification time of inherited evidence is not a raw capture time.
            start=end=None
            pages=db.execute('SELECT COUNT(DISTINCT source_page_id) FROM trades WHERE condition_id=?',(condition,)).fetchone()[0]
        if condition not in bounds:
            raise TradeIngestionError('Condition has no recovered selected observations')
        lo,hi,selected=bounds[condition]
        db.execute('INSERT INTO condition_coverage VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                   (condition,contract['fixture_id'],status,full_counts[condition],pages,lo,hi,start,end,digest,selected,'selected_cohort_observed_interval'))
    report={'schema_version':3,'finished_at_utc':now(),
        'source_scope':'selected_cohort_with_full_count_ledger',
        'coverage_bound_scope':'selected_cohort_observed_interval',
        'selected_pair_counts_exactly_match_full_ledger':True,
        'source_completeness_certified':False,'canonical_fill_identity_available':False,
        'training_coverage_certified':False,'minimum_size_tokens':'0.000001',
        'filter_scope':['wallet','condition_id'],'maximum_observations_inclusive':20,
        'full_ledger':ledger_info,'archive_recovery':archive_info,
        'raw_replayed_replacement_contracts':db.execute('SELECT COUNT(*) FROM recovery_conditions').fetchone()[0],
        'condition_count':len(contracts),'fixture_count':len({x['fixture_id'] for x in contracts.values()}),
        'source_page_semantics':'Inherited ledger includes unused and superseded references; only fresh page entries are newly raw-replayed.',
        'capture_time_semantics':'Inherited condition capture timestamps unknown; source verification time is not substituted.',
        'limitations':['API observations are not canonical fills or deliberate decisions',
            'The <=20 selection is retrospective per binary contract, not proof of market-maker status',
            'Only selected actor-contract histories are present; excluded actor-contract histories are absent',
            'Coverage endpoints are narrower selected-cohort observations, not full-condition endpoints',
            'API exhaustion is not chain completeness; trades below 0.000001 tokens are excluded'],**stats}
    for key,value in [('report',report),('source_scope',report['source_scope']),('coverage_bound_scope',report['coverage_bound_scope'])]:
        db.execute('INSERT INTO metadata VALUES(?,?)',(key,json.dumps(value,sort_keys=True)))
    db.commit()
    if db.execute('PRAGMA quick_check').fetchone()[0]!='ok':
        raise TradeIngestionError('Final SQLite integrity check failed')
    db.execute('PRAGMA wal_checkpoint(TRUNCATE)'); db.execute('PRAGMA journal_mode=DELETE'); db.close()
    return report


def publish_recovery_audit(recovery):
    """Write compact capture proofs after the immutable source DB is published."""
    recovery=Path(recovery)
    source=recovery/'source.sqlite'
    db=sqlite3.connect('file:'+str(source.resolve())+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    report=json.loads((recovery/'recovery_report.json').read_text())
    ledger_digest=hashlib.sha256()
    for row in db.execute('SELECT wallet,condition_id,observation_count FROM wallet_market_counts ORDER BY wallet,condition_id'):
        ledger_digest.update(_json_bytes(list(row)))
    replacements=[]
    for row in db.execute('SELECT * FROM recovery_conditions ORDER BY condition_id'):
        entry=dict(row);entry.update(json.loads(entry.pop('details_json')))
        entry['raw_replay_verified']=True
        replacements.append(entry)
    condition_rows=[dict(r) for r in db.execute('SELECT * FROM condition_coverage ORDER BY condition_id')]
    proof={
        'schema_version':1,'created_at_utc':now(),
        'source_database_sha256':report['source_database_sha256'],
        'source_scope':'selected_cohort_with_full_count_ledger',
        'coverage_bound_scope':'selected_cohort_observed_interval',
        'canonical_pair_count_ledger_sha256':ledger_digest.hexdigest(),
        'canonical_pair_count_ledger_hash_encoding':'UTF8 compact JSON arrays [wallet,condition_id,count] with LF, sorted by wallet then condition',
        'replacement_capture_raw_replays':replacements,
        'full_replacement_observation_count':sum(r['full_count'] for r in replacements),
        'selected_replacement_observation_count':sum(r['selected_count'] for r in replacements),
        'replacement_page_count':sum(r['page_count'] for r in replacements),
        'exact20_collection_report_sha256':sha256(recovery/'exact20'/'collection_report.json'),
        'replacement_collection_report_sha256':sha256(recovery/'trades'/'batch_progress.json'),
        'condition_coverage':condition_rows,
        'api_exhaustion_is_not_chain_completeness':True,
        'inherited_raw_response_bodies_replayed_in_this_recovery':False,
        'page_count_semantics':'Fresh conditions: full replayed page count. Inherited conditions: distinct selected-source page references, including exactly20 user query pages.'}
    db.close()
    write_json(recovery/'recovery_audit.json',proof)
    return proof


def package_capture_inputs(recovery, output_dir, chunk_bytes=8*1024*1024):
    """Persist expensive fresh captures separately from the much larger DB."""
    import shutil
    import tarfile
    recovery,output_dir=Path(recovery),Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Recovery bundle output must be empty')
    report=json.loads((recovery/'recovery_report.json').read_text())
    fresh=json.loads((recovery/'trades'/'batch_progress.json').read_text())
    exact=json.loads((recovery/'exact20'/'collection_report.json').read_text())
    if fresh['status']!='api_exhausted' or exact['status']!='complete':
        raise TradeIngestionError('Only complete captures may be bundled')
    if not (recovery/'recovery_audit.json').exists():
        publish_recovery_audit(recovery)
    directories=('cache','trades','exact20','exact20_cache')
    files=[path for name in directories for path in (recovery/name).rglob('*')
           if path.is_file() and not path.name.startswith('.')]
    files.extend(recovery/name for name in ('recovery_report.json','recovery_audit.json'))
    output_dir.mkdir(parents=True,exist_ok=True)
    fd,temporary=tempfile.mkstemp(prefix='worldcup_capture_inputs_',suffix='.tar.gz');os.close(fd)
    archive=Path(temporary)
    try:
        with tarfile.open(archive,'w:gz',compresslevel=3) as bundle:
            for path in sorted(files):
                bundle.add(path,arcname=str(path.relative_to(recovery)),recursive=False)
        archive_hash=sha256(archive);archive_bytes=archive.stat().st_size;parts=[]
        with archive.open('rb') as stream:
            index=1
            while content:=stream.read(chunk_bytes):
                path=output_dir/f'capture_inputs.tar.gz.part{index:03d}'
                path.write_bytes(content)
                parts.append({'path':path.name,'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
                index+=1
        for name in ('recovery_report.json','recovery_audit.json'):
            shutil.copyfile(recovery/name,output_dir/name)
        manifest={
            'schema_version':1,'archive_format':'tar+gzip split into ordered binary parts',
            'archive_sha256':archive_hash,'archive_bytes':archive_bytes,'archive_member_count':len(files),
            'parts':parts,'captured_condition_count':fresh['condition_count'],
            'full_replacement_observations':fresh['validated_observation_count'],
            'exact20_pairs':exact['recovered_pair_count'],'exact20_observations':exact['recovered_observation_count'],
            'old_filtered_archive_required_separately':True,
            'old_filtered_archive_sha256':report['archive_recovery']['archive_sha256'],
            'source_sqlite_included':False,'old_filtered_archive_included':False,
            'all_excluded_actor_contract_histories_included':False,
            'raw_captures_support_replay_not_chain_completeness':True}
        write_json(output_dir/'MANIFEST.json',manifest)
        (output_dir/'CHECKSUMS.sha256').write_text(''.join(f"{p['sha256']}  {p['path']}\n" for p in parts))
        readme="""# Fresh capture recovery inputs

This archive preserves the fresh19 replacement-contract API captures and1,700 exactly-20 actor–contract queries used to rebuild the actor-sequence dataset. It contains immutable raw response caches, normalized pages, cursor manifests, capture reports, and recovery audits. It does not contain `source.sqlite`, the original3.35GB filtered archive, or a complete set of histories for excluded actor–contract pairs across all 312 contracts. The 19 full-contract captures do include their excluded high-activity pairs.

The original `world_cup_lt20.sqlite` and its `MANIFEST.json` are required separately. Their verified SHA256 is recorded in this package manifest. The complete published count ledger and registry remain in `datasets/world_cup_2026_tournament_lt20_v2_evidence`.

From the repository root, first verify the parts inside this directory using `sha256sum -c CHECKSUMS.sha256`. Then concatenate parts in numeric order, extract into a new recovery directory, and rebuild offline:

```sh
cat datasets/world_cup_2026_actor_sequences_v3_recovery/capture_inputs.tar.gz.part* > /tmp/worldcup_capture_inputs.tar.gz
mkdir -p data/sequence_v3_recovery
tar -xzf /tmp/worldcup_capture_inputs.tar.gz -C data/sequence_v3_recovery
python scripts/recover_sequence_trades.py --archive /path/to/world_cup_lt20.sqlite
```

The recovery directory must not already contain a published `source.sqlite`. For a new online collection instead, add `--collect` to the recovery command. Every selected actor–contract pair must exactly match the full count ledger before publication. Inherited293-contract normalized evidence is never described as newly raw-replayed. Coverage bounds are the narrower selected-cohort observed interval. API exhaustion and repeated observations are not proofs of canonical fill identity, deliberate inactivity, or chain completeness.
"""
        # Keep prose readable while retaining exact technical identifiers.
        for old,new in [('fresh19','fresh 19'),('and1,700','and 1,700'),('original3.35GB','original 3.35 GB'),('Inherited293-contract','Inherited 293-contract')]:
            readme=readme.replace(old,new)
        readme = readme.replace('# Fresh capture recovery inputs\n', '# Fresh capture recovery inputs\n\nThe archive has ' + str(len(parts)) + ' ordered parts, each at most ' + str(chunk_bytes // (1024*1024)) + ' MiB, to fit GitHub API uploads.\n')
        (output_dir/'README.md').write_text(readme)
        return manifest
    finally:
        archive.unlink(missing_ok=True)


def collect_sources(evidence, recovery, replacements):
    """Reproduce both missing capture sets at a combined 12 logical requests/s."""
    from concurrent.futures import ThreadPoolExecutor
    from poly_world_cup.batch import run_batch
    from scripts.recover_exact20_pairs import main as exact20_main
    def replacements_job():
        report=run_batch(replacements,output_dir=recovery/'trades',cache_dir=recovery/'cache',
            workers=32,requests_per_second=6,minimum_size='0.000001',compress=True,
            on_progress=lambda r: print(now(),'replacement capture',r['status_counts'],r['committed_observation_count'],flush=True))
        if report['status']!='api_exhausted':
            raise TradeIngestionError('Replacement capture did not finish: inspect batch_progress.json')
    def exact20_job():
        status=exact20_main(['--evidence',str(evidence),'--output',str(recovery/'exact20'),
                            '--cache',str(recovery/'exact20_cache'),'--workers','32',
                            '--requests-per-second','6'])
        if status:
            raise TradeIngestionError('Exactly20 capture failed: inspect failed_collection_report.json')
    with ThreadPoolExecutor(max_workers=2) as executor:
        pending=[executor.submit(replacements_job),executor.submit(exact20_job)]
        for future in pending:
            future.result()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--evidence',type=Path,default=Path('datasets/world_cup_2026_tournament_lt20_v2_evidence'))
    p.add_argument('--recovery',type=Path,default=Path('data/sequence_v3_recovery'))
    p.add_argument('--wait-for-captures',action='store_true')
    p.add_argument('--collect',action='store_true',help='Collect or resume the 19 missing full conditions and exactly20 pairs before rebuilding')
    p.add_argument('--package-inputs',type=Path,help='After success, package fresh replay inputs in this empty output directory')
    args=p.parse_args(); args.recovery.mkdir(parents=True,exist_ok=True)
    contracts={c['condition_id']:c for c in json.loads((args.evidence/'registry.json').read_text())['contracts']}
    tokens={c:{t['token_id'] for t in row['tokens']} for c,row in contracts.items()}
    replacements={c['condition_id'] for c in json.loads((args.evidence/'replacement_collection.json').read_text())['conditions']}
    output=args.recovery/'source.sqlite'
    if output.exists():
        raise SystemExit('Refusing to replace an existing published source DB; choose a new recovery directory')
    if args.collect:
        collect_sources(args.evidence,args.recovery,replacements)
    fd,private_name=tempfile.mkstemp(prefix='worldcup_selected_source_',suffix='.sqlite');os.close(fd)
    private=Path(private_name)
    print(now(),'building private DB',str(private),flush=True)
    db=sqlite3.connect(private,uri=True)
    db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=NORMAL')
    db.execute('PRAGMA cache_size=-180000');db.executescript(SCHEMA)
    ledger=load_ledger(db,args.evidence,contracts);print(now(),'ledger loaded',ledger['observation_count'],flush=True)
    inherited=inherit_archive(db,args.archive,replacements,args.evidence);print(now(),'inherited',inherited['inherited_observation_count'],flush=True)
    remaining=set(replacements)
    while remaining:
        advanced=False
        for condition in sorted(remaining):
            path=args.recovery/'trades'/condition/'manifest.json'
            if not path.exists() or json.loads(path.read_text())['api_traversal_status']!='exhausted':
                continue
            import_condition(db,condition,args.recovery/'trades',args.recovery/'cache',tokens[condition])
            remaining.remove(condition);advanced=True
            print(now(),'replayed condition',condition,'remaining',len(remaining),flush=True)
        if remaining and not args.wait_for_captures:
            raise SystemExit(f'Missing exhausted captures: {sorted(remaining)}')
        if remaining and not advanced:
            time.sleep(10)
    while not (args.recovery/'exact20'/'collection_report.json').exists():
        if not args.wait_for_captures: raise SystemExit('Missing exact20 collection_report.json')
        time.sleep(10)
    exact=import_exact20(db,args.recovery/'exact20',args.recovery/'exact20_cache',tokens)
    print(now(),'exact20 imported',exact,flush=True)
    report=finalize(db,args.evidence,contracts,inherited,ledger,output)
    if private.stat().st_dev != output.parent.stat().st_dev:
        raise SystemExit('Private and final files must share a filesystem for atomic publication')
    report['source_database_sha256']=sha256(private);report['source_database_bytes']=private.stat().st_size
    report['source_database_path']=str(output.resolve())
    os.replace(private,output)
    write_json(args.recovery/'recovery_report.json',report)
    publish_recovery_audit(args.recovery)
    if args.package_inputs:
        package_capture_inputs(args.recovery,args.package_inputs)
    print(now(),'PUBLISHED',json.dumps({k:report[k] for k in ('selected_observation_count','selected_pair_count','source_database_sha256')}),flush=True)


if __name__=='__main__':
    main()
