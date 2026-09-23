#!/usr/bin/env python3
"""Export reproducible count and provenance evidence from a completed cohort.

The evidence contains all observed wallet/contract counts, including excluded
wallets. It does not claim to restore the missing raw pages of the inherited
capture. The previous prepared datasets and the source database are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.filter_wallet_activity import _identity, _reject_live_wal


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()


def _sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


class Shards:
    def __init__(self,root,prefix,max_bytes):
        self.root,self.prefix,self.max_bytes=root,prefix,max_bytes
        self.raw_limit=max_bytes-max(256,max_bytes//100)
        self.files=[];self.raw=None;self.gz=None;self.rows=0;self.size=0;self.digest=None
    def write(self,row):
        payload=encoded(row)
        if len(payload)>self.raw_limit:raise ValueError('One evidence row exceeds the shard limit')
        if self.gz is not None and self.size+len(payload)>self.raw_limit:self.close()
        if self.gz is None:
            self.path=self.root/self.prefix/f'part-{len(self.files)+1:05d}.jsonl.gz'
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.raw=self.path.open('xb');self.gz=gzip.GzipFile(filename='',mode='wb',fileobj=self.raw,mtime=0,compresslevel=6)
            self.rows=0;self.size=0;self.digest=hashlib.sha256()
        self.gz.write(payload);self.digest.update(payload);self.rows+=1;self.size+=len(payload)
    def close(self):
        if self.gz is None:return
        self.gz.close();self.raw.close();self.gz=None;self.raw=None
        if self.path.stat().st_size>=self.max_bytes:raise ValueError('Compressed evidence shard exceeds its limit')
        self.files.append({'path':self.path.relative_to(self.root).as_posix(),'rows':self.rows,
                           'bytes':self.path.stat().st_size,'sha256':_sha(self.path),
                           'uncompressed_sha256':self.digest.hexdigest()})


def export_trade_evidence(database:Path,output:Path,*,reports:dict[str,dict | list | Path],max_bytes:int=20_000_000)->dict:
    if type(max_bytes) is not int or max_bytes<1024:raise ValueError('max_bytes must be an integer at least 1024')
    database=Path(database).resolve(strict=True);output=Path(output).absolute()
    if os.path.lexists(output):raise FileExistsError('Refusing to replace existing evidence output')
    _reject_live_wal(database);before=_identity(database)
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.',dir=output.parent))
    db=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    writer=None
    try:
        metadata={row['key']:json.loads(row['value_json']) for row in db.execute('SELECT * FROM metadata ORDER BY key')}
        report=metadata['report'];ledger=metadata['contract_exhaustion_ledger']
        if report.get('partial_api_collection') is not False or any(row.get('status')!='api_exhausted' for row in ledger):
            raise ValueError('Evidence export requires all captured contract traversals exhausted')
        if len(ledger)!=report.get('registry_condition_count'):raise ValueError('Exhaustion ledger has wrong contract count')
        writer=Shards(stage,'trade_count_ledger',max_bytes)
        for row in db.execute('SELECT wallet,condition_id,observation_count FROM wallet_market_counts ORDER BY wallet,condition_id'):
            writer.write(dict(row))
        writer.close();files=list(writer.files);writer=None
        # Source references include superseded pages as disclosed in the report.
        source_path=stage/'source_page_ledger.jsonl.gz';source_rows=0;source_digest=hashlib.sha256()
        with source_path.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as stream:
            for row in db.execute('SELECT source_page_id,path,uncompressed_sha256,row_count FROM source_pages ORDER BY source_page_id'):
                payload=encoded(dict(row));stream.write(payload);source_digest.update(payload);source_rows+=1
        if source_path.stat().st_size>=max_bytes:raise ValueError('Source-page evidence exceeds shard size; shard it before publishing')
        files.append({'path':source_path.name,'rows':source_rows,'bytes':source_path.stat().st_size,
                      'sha256':_sha(source_path),'uncompressed_sha256':source_digest.hexdigest()})
        provided={'completion_report':report,'contract_exhaustion_ledger':ledger,**reports}
        if 'completion_report' in reports or 'contract_exhaustion_ledger' in reports:
            raise ValueError('Caller reports cannot replace database-derived evidence')
        for name,value in provided.items():
            if not name or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for ch in name):
                raise ValueError('Evidence report names must be simple identifiers')
            path=stage/(name+'.json')
            payload=value.read_bytes() if isinstance(value,Path) else encoded(value)
            json.loads(payload)
            path.write_bytes(payload)
            if path.stat().st_size>=max_bytes:raise ValueError('One evidence report exceeds the publication size limit')
            files.append({'path':path.name,'bytes':path.stat().st_size,'sha256':_sha(path)})
        # Read back every gzip row: this checks trailer/CRC, JSON, count, digest,
        # ordering, and the full threshold decision independently of SQLite SUM.
        observations=0;pairs=0;wallets=0;kept_wallets=0;kept_rows=0
        last_key=None;current_wallet=None;current_count=0;conditions=set()
        for entry in files:
            path=stage/entry['path']
            if _sha(path)!=entry['sha256']:raise ValueError('Evidence checksum mismatch')
            if not entry['path'].endswith('.jsonl.gz'):
                json.loads(path.read_bytes());continue
            count=0;digest=hashlib.sha256()
            with gzip.open(path,'rb') as stream:
                for line in stream:
                    digest.update(line);row=json.loads(line);count+=1
                    if entry['path'].startswith('trade_count_ledger/'):
                        key=(row['wallet'],row['condition_id']);number=row['observation_count']
                        if last_key is not None and key<=last_key:raise ValueError('Count ledger is not uniquely ordered')
                        if type(number) is not int or number<1:raise ValueError('Invalid count ledger value')
                        if current_wallet is not None and current_wallet!=key[0]:
                            wallets+=1
                            if current_count<report['threshold_exclusive']:kept_wallets+=1;kept_rows+=current_count
                            current_count=0
                        current_wallet=key[0];current_count+=number;last_key=key
                        observations+=number;pairs+=1;conditions.add(key[1])
            if count!=entry['rows'] or digest.hexdigest()!=entry['uncompressed_sha256']:
                raise ValueError('Evidence row count or uncompressed digest mismatch')
        if current_wallet is not None:
            wallets+=1
            if current_count<report['threshold_exclusive']:kept_wallets+=1;kept_rows+=current_count
        for actual,expected,label in [(observations,report['source_observations'],'observations'),
            (wallets,report['source_wallets'],'wallets'),(kept_rows,report['retained_observations'],'retained observations'),
            (kept_wallets,report['retained_wallets'],'retained wallets')]:
            if actual!=expected:raise ValueError('Ledger totals disagree with '+label)
        if conditions!={row['condition_id'] for row in ledger}:raise ValueError('Count ledger and exhaustion identities differ')
        manifest={'schema_version':1,'wallet_contract_pair_count':pairs,'source_observations':observations,
                  'source_wallets':wallets,'retained_observations':kept_rows,'retained_wallets':kept_wallets,
                  'condition_count':len(conditions),'threshold_exclusive':report['threshold_exclusive'],
                  'max_artifact_bytes_exclusive':max_bytes,'files':files,'gzip_crc_and_full_readback_verified':True,
                  'inherited_raw_pages_included':False,'fresh_raw_pages_included':False,
                  'raw_source_scope':'Provenance and count evidence; raw capture bodies are not part of this export.',
                  'superseded_original_source_page_references_retained':True}
        (stage/'manifest.json').write_bytes(encoded(manifest))
        db.close();db=None
        _reject_live_wal(database)
        if _identity(database)!=before:raise ValueError('Source changed during evidence export')
        os.rename(stage,output)
        return manifest
    finally:
        if writer is not None and writer.gz is not None:writer.gz.close();writer.raw.close()
        if db is not None:db.close()
        if stage.exists():shutil.rmtree(stage)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',action='append',default=[],metavar='NAME=PATH')
    args=p.parse_args();reports={}
    for item in args.report:
        name,sep,path=item.partition('=')
        if not sep or name in reports:p.error('Each report needs a unique NAME=PATH')
        reports[name]=Path(path).resolve(strict=True)
    manifest=export_trade_evidence(args.database,args.output,reports=reports)
    print(json.dumps({k:v for k,v in manifest.items() if k!='files'},indent=2))

if __name__=='__main__':main()
