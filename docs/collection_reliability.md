# Collection reliability and capacity planning

The collection and training stages have different resource requirements. Collect and validate the three datasets once on a CPU machine with working access to the required APIs. Then transfer the prepared datasets to persistent storage and train on the GPU Pods. A GPU does not speed up these HTTP requests. Training should not start by downloading the dataset again.

## What failed on the RunPod collector

The observed failures were two different network problems:

| Evidence | Supported conclusion |
| --- | --- |
| `curl: (6) Could not resolve host: clob.polymarket.com` | The Pod could not resolve that hostname. The successful lookup after changing DNS configuration confirms that the initial failure was at the resolver layer. |
| DNS subsequently worked, but `curl: (35) Recv failure: Connection reset by peer` returned HTTP `000` | The TLS connection failed before an HTTP response was received. The log does not identify who reset it. |
| The same price-history URL returned HTTP 200 from the Mac | The endpoint and request were usable from that machine at that time. This supports investigating the Pod's outbound network path and remote handling of that connection. It does not prove that the Pod was rate-limited, geographically restricted, or blocked by a specific service. |

The [curl error documentation](https://curl.se/libcurl/c/libcurl-errors.html) distinguishes hostname resolution failure (6) from SSL/TLS connection failure (35). An HTTP status of `000` is not a server response code.

RunPod's [port exposure guide](https://docs.runpod.io/pods/configuration/expose-ports) describes **incoming** traffic through its HTTP proxy: a user accesses an application running inside a Pod. Its 100-second timeout and HTTP 524 errors do not explain a command inside a Pod failing to fetch an external Polymarket URL. Similarly, the [RunPod 502 troubleshooting guide](https://docs.runpod.io/pods/troubleshooting/troubleshooting-502-errors) concerns access to programs exposed by a Pod. Opening more inbound ports does not fix the observed outbound failure.

As checked on 2026-09-28, the [official Polymarket rate-limit documentation](https://docs.polymarket.com/api-reference/rate-limits) lists 300 requests per 10 seconds for `/v2/trades` and 1,000 per 10 seconds for CLOB `/prices-history`. These limits are IP-based; excess requests are described as delayed or queued. Shared-IP traffic can matter, but the provided reset logs do not establish that throttling was the cause. A conservative client rate limit remains appropriate.

If a bounded preflight fails, stop collection and retain its diagnostics. Report the Pod ID, hostname, UTC test time, failing endpoint, DNS result, curl error and timing information to [RunPod support](https://contact.runpod.io/hc/en-us/requests/new). Do not spend hours retrying the same broken path, silently substitute different price semantics, disable certificate verification, or rotate addresses to evade access restrictions.

## Why the old retry loop wasted time

The old command used eight retries after the first attempt, with a 45-second timeout per attempt. Its delays were 2, 4, 8, 16, 32, 60, 60 and 60 seconds: **242 seconds of sleep for one URL**. Nine full 45-second attempts plus those sleeps amount to about **647 seconds, or 10.8 minutes**, excluding process overhead or a longer server `Retry-After` header. Fast resets take less time, but still incur the retry delays. The allowance started over for every new page.

A reliable collector needs a total retry budget, a small retry count, informative error categories, and a run-wide stopping condition. Retrying transient failures is useful; unbounded cumulative waiting is not. A stopped collection must remain resumable and must not publish an incomplete dataset as complete.

## What the logs establish about the data volume

The completed Basic run selected 40,001 training targets, 2,013 validation targets and 2,000 test targets: **44,014 decision targets total**. These are not the number of API observations fetched. The eight market captures in that log required **162 trade pages containing 158,145 observations**, before actor filtering and target selection. ESPN, market metadata and YES/NO price histories required additional requests.

In-market derives its metrics from those existing market exports. It does not need another wallet-wide API download.

Global identified **17,170 unique wallets** and requested each wallet's market-spanning history up to its relevant historical cutoff. The 40,000 training-target setting does not cap the number of prior executions downloaded. A low trade count in the selected World Cup market does not imply a low lifetime trade count across other markets.

The supplied progress excerpt contains 177 completed wallet captures with 318 pages and 179,385 executions. The median wallet in that excerpt had 89 executions, while the largest had 28,243. These wallets form an alphabetically ordered slice, not a random sample. The excerpt has no wall-clock timestamps or disk-byte totals. It cannot justify an exact remaining-time or storage estimate after the overnight run.

## Obtain the current estimate without making API requests

Run this on the machine containing the existing collection:

```bash
python3 tools/estimate_actor_collection.py \
  --root "$HOME/world_cup_40k_transfer" \
  --elapsed-hours 7 \
  --out "$HOME/world_cup_40k_transfer/collection_estimate.json"
```

The elapsed-hours argument supplies the time already spent; it is not a target deadline. The report distinguishes observed cache size and completed work from assumptions about unfinished wallets. It reads the existing dataset and cache without fetching more data. The scenario parameters can be changed explicitly:

```bash
python3 tools/estimate_actor_collection.py \
  --root "$HOME/world_cup_40k_transfer" \
  --workers 4 \
  --min-interval 0.25 \
  --pages-per-wallet 1,5,20 \
  --seconds-per-page 1,3,10
```

The second command's page counts and service times are assumptions, not measured API performance. Prefer actual cache measurements when they are available. An estimate from a prefix of wallets cannot rule out a much larger history among the remaining wallets.

## Planning examples, not completion guarantees

These estimates exclude model weights, tokenization caches, training checkpoints and transfer time.

| Variant | Network work | Time and storage planning basis |
| --- | --- | --- |
| Basic, already completed | Zero new requests when reusing the prepared dataset | Read its actual size from the estimator. Do not collect it again to train another variant. |
| Basic, fresh run of the same eight markets | 162 observed trade pages, plus metadata, ESPN and price-history requests | Hundreds of successful requests rather than tens of thousands. At 1–3 seconds per successful request, 400 requests represent roughly 7–20 minutes of network service, before export and preparation. A 400-request total is a scenario, not the measured total. For 44,014 targets, 10–50 KB of retained source/export/prepared data per target would be approximately 0.41–2.05 GiB; actual storage depends heavily on repeated news and price histories. |
| Basic + In-market | Zero additional API requests for the selected execution metrics | An offline pass over the exports. At 100–1,000 processed targets per second, 44,014 targets take about 7.3 minutes to 44 seconds, before extra validation and I/O. An incremental 2–10 KB per target would require roughly 84–420 MiB. Both throughput and bytes per target must be measured on the collector; they are sizing assumptions. |
| Basic + Global, full history | At least one request per uncached wallet; additional pages depend on each wallet's history | Use the wallet and request scenarios below. The completed Basic and In-market datasets do not make this download small. |

For 17,170 wallets, a fresh serial Global download has the following **network-service scenarios**, without retries or feature computation:

| Mean pages per wallet | Total requests | At 1 second per page | At 3 seconds per page |
| --- | ---: | ---: | ---: |
| 1 | 17,170 | 4.8 hours | 14.3 hours |
| 5 | 85,850 | 23.8 hours | 71.5 hours |
| 20 | 343,400 | 95.4 hours | 286.2 hours |

If the supplied 177-wallet excerpt happened to represent the whole cohort, it would imply about **30,848 requests and 17.4 million executions**. At 1–3 seconds per page, that is roughly **8.6–25.7 hours serially**. Four independent wallet workers could reduce the network-service component toward **2.1–6.4 hours**, subject to shared rate limits, long individual wallets, disk I/O and retries. This is a conditional scenario, not an ETA or a statistically representative estimate. Cached work reduces what remains.

The previous normalized wallet format repeated source provenance in every execution and stored those pages uncompressed. Serializing the previously rejected example with representative request provenance produces about 1.1 KB on a first page and 1.3 KB on a page with a cursor. Those are example record sizes, not a measurement of the user's cache. At 1.3 KB per execution, 17.4 million executions alone would occupy about **21.1 GiB**, before compressed raw HTTP bodies, metadata and derived outputs.

New normalized pages are compressed, while old valid pages are preserved for resume. Compression ratios vary. If the **combined** compressed raw and normalized records averaged 200–500 bytes per execution, the same scenario would consume about **3.2–8.1 GiB**, plus manifests, metrics, final datasets and filesystem overhead. Existing uncompressed pages still occupy their original space. The estimator measures existing files instead of assuming this compression ratio. Do not allocate storage based only on the small final SFT files.

## Bounded Global runs and honest comparisons

The Global collector supports four wallet workers by default, a shared request pacing policy, compressed new wallet pages and reusable per-wallet metric checkpoints. It reports progress during collection rather than only every 100 wallets.

Its default runtime budget is two hours (`--max-runtime-seconds 7200`) and its default cache-size budget is 20 GiB (`--max-cache-gib 20`). `--max-new-pages` can set an additional run limit. These are stop-and-resume budgets, not promises that full history will finish within them. A budget stop leaves the cached work in place and does not publish a complete Global SFT dataset. Use `--wallet-workers` to set 1–8 workers. Setting runtime or cache limits to zero explicitly disables that limit; doing so removes that protection.

The collection clock starts after initial source validation and staging. In-flight requests or CPU work can finish after a stop is requested. Cache-size checks run periodically, so all writes between checks and in-flight responses can exceed the configured threshold. The cache budget excludes temporary/final datasets, model files, token caches and training outputs. Keep free space beyond the configured cache allowance. Curl enforces total transfer deadlines; urllib's socket timeout does not provide the same bound on every platform's DNS and reads.

Budget changes must not silently change the scientific comparison. Keep the same actors, target decisions and train/validation/test splits in all variants. Do not skip slow wallets or truncate their history and still label the result full-history Global. A deliberate trailing-window experiment is a different feature definition and needs a separate manifest and explicit comparison plan. Features at decision time must still exclude the current and future executions.

## RunPod workflow

1. Reuse the completed Basic and In-market outputs and the existing wallet cache. Keep the dataset and cache out of GitHub.
2. Estimate the remaining collection on its current machine. Use bounded runs and resume instead of launching an open-ended fetch.
3. Collect once on a suitable CPU machine whose endpoint preflight succeeds. Persistent DNS or TLS failures require investigation, not a bigger retry count.
4. Validate matching targets and splits with `tools/compare_actor_variants.py` when Global is complete.
5. Transfer the prepared datasets once to persistent storage. The raw wallet cache is needed for collection/audit, not for the trainer to read each example. RunPod documents [rsync and SCP transfer](https://docs.runpod.io/pods/storage/transfer-files); use the actual Pod's SSH endpoint, not an invented host or port.
6. Train each variant with `scripts/train_world_cup_multigpu.py` and a separate output directory. The training jobs should not fetch market data or overwrite each other's outputs.

[RunPod network volumes](https://docs.runpod.io/storage/network-volumes) persist independently of compute and normally mount at `/workspace` for Pods. Distinct volumes do not synchronize automatically. Confirm that the intended Pods use the same volume before relying on one upload. RunPod also supports [uploading to a network volume through its S3-compatible API](https://docs.runpod.io/storage/s3-api) without starting GPU compute. Use `df -h` and the estimator before collection; see the [storage-full troubleshooting guide](https://docs.runpod.io/pods/troubleshooting/storage-full).
