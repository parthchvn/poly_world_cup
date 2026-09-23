# Receipt sample checks and the remaining coverage gap

`poly_world_cup.reconcile` checks a bounded set of observed API rows against
Polygon transaction receipts. Its default selection is one observation from the
newest committed page for each fixture, using deterministic contract ordering.
It is a coverage sample, not a random sample or an estimate of dataset error rates.

The sample reports known-contract log IDs, order hashes, raw integer amounts,
the wallet's field in the log, and compatible economic directions. It preserves
both the API observation and the receipt evidence. It never replaces API row
identifiers or certifies missing-trade intervals.

```python
import json
from pathlib import Path
from poly_world_cup.reconcile import run_sample

registry = json.loads(Path("data/registry/registry.json").read_text())
report = run_sample(registry, Path("data/full/trades"),
                    Path("data/full/reconciliation"), max_transactions=104)
print(report["status_counts"])
```

The public RPC endpoint is read-only and requires no credentials. Its historical
coverage and availability are not guaranteed. Missing receipts and RPC failures
remain unknown. The collector checks chain ID 137 and caches returned receipts
with capture time, provider, and checksum.

Four sample workers share a request-start pacer, while one writer updates the
report. `sample_collection_status` distinguishes a running report from a completed
sample. A completed sample may still have missing fixtures or unknown receipts;
inspect `selected_fixture_count`, `missing_fixture_ids`, and every status count.
Sampling the newest page mainly checks late market activity; it does not establish
accuracy for early premarket activity or other wallets. The initial separate
V1 smoke check found a quantity discrepancy and preserved it as a mismatch.
Reruns archive the previous report under `runs/` before replacing the current
report, so prior mismatches remain inspectable if sample selection changes.

The separate April 9 diagnostic transaction
`0x96858f8b2dff012bb872ac54f3d40cad1dbade0f09ed02835d95c985bee2a5c8`
has an API size of `1.029060` shares. Its aggregate V1 event records `1.030900`
gross shares and `0.003188` in its fee field. Two observed ERC-1155 transfers to
the same wallet for the same token, at log indices 1615 and 1620, contain
`1.027712` and `0.002268` shares, respectively: `1.029980` shares in total.
The role of the additional transfer and the remaining `0.000920` difference
from the API quantity are unresolved. The sample does not impose a correction
formula or classify that additional transfer's purpose. This early diagnostic
is separate from the 104-fixture late-market sample and supports no estimate of
the overall dataset's error rate.

The separate trade batch collector permits up to 64 workers to overlap network
latency. Its global logical-request pacing remains independent of the worker
count; increasing workers does not raise that configured request rate. HTTP
retries have their own bounded backoff. Receipt sampling stays capped at eight
workers and defaults to four.

## Why a log is not automatically a distinct trade

Polymarket's matching contracts emit maker-order fills and an aggregate event
for the active order. In the aggregate event, the `maker` field holds the active
order owner and the `taker` field holds the exchange address. Counting both paths
as separate user-to-user executions would double count. One API row may match
both paths; the report retains both candidates and labels the mapping ambiguous.
It does not choose whichever log happens to appear first.

V1 identifies collateral with asset ID zero and exposes two asset IDs. V2 has an
explicit maker-order side and outcome token ID. Their ABI layouts differ. The
implementation recognizes four exchange addresses and their corresponding event
signatures; unknown addresses and malformed or removed logs fail closed.

Gross token sizes and cash/token prices are calculated from integer fields using
decimal arithmetic. A V1 BUY order owner may receive shares after the logged token
fee, so that comparison is a separately labelled variant. Defaults permit one
microshare of size difference and 0.000001 of price difference. Larger differences
remain mismatches. This is compatibility testing, not a general reconstruction of
all mint, merge, fee, transfer, or partially filled-order semantics.

Pinned official sources:

- [V1 ABI](https://github.com/Polymarket/ctf-exchange/blob/ed5c7708b7be3aa98bf5f0c6602b57cc498e2ef4/src/exchange/interfaces/ITrading.sol)
- [V1 matching and event emissions](https://github.com/Polymarket/ctf-exchange/blob/ed5c7708b7be3aa98bf5f0c6602b57cc498e2ef4/src/exchange/mixins/Trading.sol)
- [V2 ABI](https://github.com/Polymarket/ctf-exchange-v2/blob/ccc0596074f4dfd62c944fbca4de252893b82b4b/src/exchange/interfaces/ITrading.sol)
- [V2 matching and event emissions](https://github.com/Polymarket/ctf-exchange-v2/blob/ccc0596074f4dfd62c944fbca4de252893b82b4b/src/exchange/mixins/Trading.sol)
- [Official migration guide](https://docs.polymarket.com/v2-migration)

## Full reconciliation is still separate work

The [Polymarket Institute guide](https://institute.polymarket.com/data) documents
APIs; it does not supply a bulk historical archive. A public independent archive,
[wzsg/polymarket-orderfilled-v2](https://huggingface.co/datasets/wzsg/polymarket-orderfilled-v2),
has canonical log fields and an accompanying V1 dataset. At inspected revision
`f8782fc0c18a4e32b7bdc348ada114e5007a4367`, V2 covers April 28 through August 3,
2026. April's V1 data plus April–July V2 data total roughly 46 GB compressed before
projection/filtering. It was inspected as a candidate source, not imported.

Hugging Face's [filter API](https://huggingface.co/docs/dataset-viewer/en/filter)
can index only the first 5 GB of large datasets and returns `partial: true` in
that case. It cannot establish complete tournament coverage. A future full
reconciliation must pin archive files, validate block-range coverage and token
mapping, normalize V1/V2 separately, and compare economic executions without
double counting aggregate emissions. API exhaustion and a successful receipt
sample remain insufficient for complete no-trade labels.
