#!/usr/bin/env python3
"""Bounded, read-only checks of the actual data endpoints used by collection.

No DNS settings, TLS verification, proxy settings or installed packages change.
Successful probes confirm connectivity now, not reliability of a long capture.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlencode


def endpoints(market_id='1897035', wallet=None, include_espn=True):
    scripts = str(Path(__file__).resolve().parents[1] / 'scripts')
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import build_actor_dataset as builder
    contract = next((c for c in builder.BUNDLED_REGISTRY['contracts']
                     if str(c['market_id']) == str(market_id)), None)
    if contract is None:
        raise ValueError('Network check needs a registered World Cup market ID')
    stamp = int(datetime.fromisoformat(contract['game_start_time'].replace('Z', '+00:00')).timestamp())
    result = [
        ('market_trades', 'https://data-api.polymarket.com/v2/trades?' + urlencode({
            'condition': contract['condition_id'], 'limit': 1, 'taker_only': 'false',
            'filter_type': 'TOKENS', 'filter_amount': '0.000001'}), 'data'),
        ('price_history', 'https://clob.polymarket.com/prices-history?' + urlencode({
            'market': contract['tokens'][0]['token_id'], 'startTs': stamp - 3600,
            'endTs': stamp + 3600, 'fidelity': 1}), 'history'),
    ]
    if include_espn:
        event = contract['fixture_id'].split(':')[-1]
        result.append(('espn', 'https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?'
                       + urlencode({'event': event}), 'header'))
    if wallet:
        import re
        if not re.fullmatch(r'0x[0-9a-fA-F]{40}', wallet):
            raise ValueError('Invalid wallet address')
        result.append(('wallet_trades', 'https://data-api.polymarket.com/v2/trades?' + urlencode({
            'user': wallet, 'start': 1, 'end': stamp, 'limit': 1,
            'taker_only': 'false', 'filter_type': 'TOKENS', 'filter_amount': '0.000001'}), 'data'))
    return result


def probe(endpoint, *, timeout=15):
    name, url, expected = endpoint
    result = {'endpoint': name, 'url': url, 'ok': False}
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='collection-probe-') as directory:
        body = Path(directory) / 'body.json'
        command = ['curl', '--disable', '--silent', '--show-error',
                   '--connect-timeout', str(min(8, timeout)), '--max-time', str(timeout),
                   '--max-filesize', '8388608', '--output', str(body),
                   '--write-out', '%{http_code}\t%{remote_ip}\t%{time_namelookup}\t%{time_connect}\t%{time_appconnect}', url]
        try:
            response = subprocess.run(command, capture_output=True, text=True, timeout=timeout + 2)
        except subprocess.TimeoutExpired:
            result.update(failure='process_timeout', detail='curl exceeded the probe time limit')
        else:
            fields = response.stdout.strip().split('\t')
            result.update(curl_exit_code=response.returncode, http_status=fields[0] if fields else '000',
                          detail=response.stderr.strip()[-1000:])
            if len(fields) == 5:
                result.update(remote_ip=fields[1], dns_seconds=fields[2],
                              tcp_seconds=fields[3], tls_seconds=fields[4])
            if response.returncode:
                result['failure'] = {5: 'proxy_dns', 6: 'dns', 7: 'tcp_connect', 28: 'timeout',
                                     35: 'tls_connect', 56: 'connection_receive', 60: 'tls_certificate'}.get(
                                         response.returncode, 'curl_transport')
            elif result['http_status'] != '200':
                result['failure'] = 'http_' + result['http_status']
            else:
                try:
                    payload = json.loads(body.read_text())
                    value = payload.get(expected) if isinstance(payload, dict) else None
                    valid = isinstance(value, dict if expected == 'header' else list)
                    if not valid:
                        raise ValueError(f'Missing expected {expected} response field')
                    result['ok'] = True
                    result['response_bytes'] = body.stat().st_size
                except (ValueError, OSError) as error:
                    result.update(failure='unexpected_response', detail=str(error))
    result['elapsed_seconds'] = round(time.monotonic() - started, 3)
    return result


def check_network(market_id='1897035', *, wallet=None, repeats=2, timeout=15, include_espn=True):
    if not shutil.which('curl'):
        raise ValueError('curl is required for the collection network check')
    checks = endpoints(market_id, wallet, include_espn)
    results = []
    for attempt in range(1, repeats + 1):
        with ThreadPoolExecutor(max_workers=len(checks)) as pool:
            batch = list(pool.map(lambda endpoint: probe(endpoint, timeout=timeout), checks))
        results.extend(dict(item, attempt=attempt) for item in batch)
        if any(not item['ok'] for item in batch):
            break
    return {'ok': all(item['ok'] for item in results), 'checks': results,
            'note': 'No network settings changed. Passing probes do not guarantee subsequent requests.'}


def require_network(market_id, **kwargs):
    report = check_network(market_id, **kwargs)
    print(json.dumps({'network_preflight': report}, indent=2), flush=True)
    if not report['ok']:
        failures = ', '.join(item['endpoint'] + ': ' + item['failure']
                             for item in report['checks'] if not item['ok'])
        raise ValueError('Collection network check failed (' + failures + '). '
                         'Saved captures are unchanged. Check this host with RunPod support or '
                         'collect on a working machine and transfer the completed dataset.')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--market-id', default='1897035')
    parser.add_argument('--wallet')
    parser.add_argument('--timeout', type=float, default=15)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--skip-espn', action='store_true')
    args = parser.parse_args(argv)
    if not (0 < args.timeout <= 60 and 1 <= args.repeats <= 3):
        parser.error('Use timeout in (0, 60] and repeats between 1 and 3')
    report = check_network(args.market_id, wallet=args.wallet, timeout=args.timeout,
                           repeats=args.repeats, include_espn=not args.skip_espn)
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        raise SystemExit(f'Error: {error}') from error
