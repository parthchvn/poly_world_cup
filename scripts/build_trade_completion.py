#!/usr/bin/env python3
"""Replay completed replacement captures, attribute them, and rebuild the cohort.

Run after collect-tournament/ingest has exhausted the condition IDs recorded in
an original collection checkpoint. This reads all source files and publishes a
new database. It never replaces the previous cohort or prepared SFT dataset.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from poly_world_cup.attribution import build_attribution_index
from poly_world_cup.io import write_json
from poly_world_cup.provenance import verify_raw_provenance
from poly_world_cup.trades import validate_collection
from scripts.complete_tournament_wallets import complete_tournament_database


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--registry',type=Path,required=True)
    parser.add_argument('--original-checkpoint',type=Path,required=True)
    parser.add_argument('--original-provenance',type=Path,required=True)
    parser.add_argument('--trades-root',type=Path,required=True)
    parser.add_argument('--cache-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--require-tournament-coverage',action='store_true')
    args=parser.parse_args()
    output=args.output_dir.absolute()
    output.mkdir(parents=True,exist_ok=True)
    names=('fresh_attribution.sqlite','fresh_attribution_report.json','fresh_provenance.json',
           'fresh_collection.json','completed_tournament.sqlite','completion_report.json')
    if any(os.path.lexists(output/name) for name in names):
        parser.error('Output directory already contains completed artifacts; use a new output directory')
    checkpoint=json.loads(args.original_checkpoint.read_text())
    original_provenance=json.loads(args.original_provenance.read_text())
    if hashlib.sha256(args.original_provenance.read_bytes()).hexdigest()!=checkpoint['integrity']['input_sha256']['provenance']:
        raise ValueError('Original provenance bytes do not match original collection checkpoint')
    conditions=sorted(checkpoint['unfinished_condition_ids'])
    registry=json.loads(args.registry.read_text())
    if hashlib.sha256(args.registry.read_bytes()).hexdigest()!=checkpoint['integrity']['input_sha256']['registry']:
        raise ValueError('Registry bytes do not match original collection checkpoint')
    def progress(report):
        write_json(output/'provenance_progress.json',report)
        print(json.dumps({'phase':'raw_provenance','conditions_verified':report['verified_condition_count'],
                          'pages_verified':report['verified_raw_page_count'],
                          'observations_verified':report['verified_observation_count'],
                          'errors':len(report['errors'])}),flush=True)
    before_manifests={c:hashlib.sha256((args.trades_root/c/'manifest.json').read_bytes()).hexdigest() for c in conditions}
    provenance=verify_raw_provenance(args.trades_root,args.cache_root,condition_ids=conditions,
                                    cache_layout='per_condition',on_progress=progress)
    if provenance['raw_provenance_verified'] is not True or provenance['api_exhausted_conditions']!=len(conditions):
        write_json(output/'failed_provenance.json',provenance)
        raise ValueError('All replacement contracts must be exhausted and raw-replay verified')
    pages=[];hashes={};counts={};rows=[];manifest_hashes={};verified_pages=[]
    for condition in conditions:
        state=validate_collection(args.trades_root,condition_id=condition)
        if state['api_traversal_status']!='exhausted' or state['minimum_size_filter']!={'type':'TOKENS','amount':'0.000001'}:
            raise ValueError('Replacement has wrong threshold or is not exhausted')
        manifest=args.trades_root/condition/'manifest.json'
        manifest_hashes[condition]=hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest_hashes[condition]!=before_manifests[condition]:
            raise ValueError('Collection changed during raw provenance verification')
        rows.append({'condition_id':condition,'status':'exhausted','integrity_validated':True,
                     'row_count':state['row_count'],'page_count':state['page_count'],
                     'earliest_block_timestamp':state['earliest_block_timestamp'],
                     'latest_block_timestamp':state['latest_block_timestamp']})
        for page in state['pages']:
            path=args.trades_root/condition/page['file']
            pages.append(path);hashes[path]=page['normalized_sha256'];counts[path]=page['row_count']
            verified_pages.append({'path':str(path.resolve()),'uncompressed_sha256':page['normalized_sha256'],'row_count':page['row_count'],'condition_id':condition})
    provenance['condition_manifest_sha256']=manifest_hashes
    provenance['verified_condition_ids']=conditions
    provenance['verified_pages']=verified_pages
    collection={'status':'api_exhausted','conditions':rows,'condition_count':len(rows),
        'status_counts':{'exhausted':len(rows),'paused':0,'failed':0,'queued':0,'running':0},
        'requested_minimum_size_tokens':'0.000001','taker_only':False,'validated_observation_count':sum(r['row_count'] for r in rows),
        'committed_observation_count':sum(r['row_count'] for r in rows),
        'committed_page_count':len(pages),'condition_manifest_sha256':manifest_hashes,
        'finished_at':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),
        'training_coverage_certified':False,'canonical_fill_identity_available':False}
    with sqlite3.connect(args.source.resolve().as_uri()+'?mode=ro',uri=True) as db:
        news=[json.loads(row[0]) for row in db.execute('SELECT record_json FROM news ORDER BY news_id')]
    write_json(output/'fresh_provenance.json',provenance)
    write_json(output/'fresh_collection.json',collection)
    print(json.dumps({'phase':'fresh_attribution','source_observations':collection['validated_observation_count']}),flush=True)
    attributed=build_attribution_index(registry=registry,trade_pages=pages,news_records=news,
        output_path=output/'fresh_attribution.sqlite',expected_page_hashes=hashes,expected_page_row_counts=counts)
    write_json(output/'fresh_attribution_report.json',attributed)
    print(json.dumps({'phase':'merge_cohort','fresh_observations':attributed['observation_count']}),flush=True)
    report=complete_tournament_database(args.source,output/'fresh_attribution.sqlite',output/'completed_tournament.sqlite',
        original_checkpoint=checkpoint,original_provenance=original_provenance,fresh_collection=collection,fresh_provenance=provenance,
        require_tournament_coverage=args.require_tournament_coverage)
    write_json(output/'completion_report.json',report)
    print(json.dumps({'phase':'completed',**report},sort_keys=True),flush=True)

if __name__=='__main__':
    main()
