# Historical headline evidence

Present-day article metadata is useful for finding candidate news, but its
publication timestamp does not prove that its present text existed at that time.
`poly_world_cup.news_archive.ArchiveCollector` retrieves a separate, historical
headline version from Internet Archive captures.

A version is accepted only when all of these agree:

- The CDX index reports a successful HTML capture for the exact article URL.
- The replay returns the requested capture without redirecting to a later one.
- `Memento-Datetime` equals the CDX capture timestamp.
- The replay's original-URL link identifies the requested article.
- The replay payload's SHA-1 digest matches the CDX digest. Both encoded and
  decoded payloads are checked because archives can preserve gzip transport.
- An archived `og:title` or page title is present and is not a recognized error
  page. Conflicting `og:title` values fail validation.

The conservative availability upper bound is the **capture time**, not the
publisher's earlier publication claim. The collector never applies the archive
proof to a present-day headline or article body. Output contains the archived
headline only, with its content hash, replay URL and source evidence. Full HTML
is retained exclusively in the ignored local cache for reproducibility.

## Running the collector

```python
import json
from pathlib import Path
from poly_world_cup.news_archive import ArchiveCollector

articles = [json.loads(line) for line in Path("data/news/news.jsonl").read_text().splitlines()]
report = ArchiveCollector(Path("data/news_archive"), workers=16).collect(articles)
print(report)
```

Requests start at most once per second across all workers. Sixteen workers
can overlap archive latency without raising that request rate. Per-article
checkpoints allow resuming, and cached headline content is revalidated against
its saved HTML hash on resume. Encoded replay payloads are also retained locally
and rechecked against CDX digests and Memento metadata; saved timing flags are
rebuilt from that evidence, rather than trusted from an old checkpoint. The
collector batches ESPN path-prefix index queries and retrieves each URL's first
capture. Exact-URL fallback queries check up to three capture rows, including repeat
captures of the same content digest.
It exports the earliest retrieved replay that passes validation. Index response
limits and errors are recorded; capped prefix responses never certify absence. This is a conservative sample of historical
versions, not a complete article-revision archive. Lookup errors, missing captures
and failed replay verification remain explicit statuses. `historical_coverage_complete`
is always false. Changing the lookup cutoff creates distinct lookup checkpoints.

The final archive pass queries every URL in the current candidate catalog,
including metadata with publication claims outside the study window. An article
updated after the tournament might still have a valid earlier capture. Historical
eligibility therefore follows capture evidence rather than today's publication
date. This does not remove retrospective catalog selection bias: deleted or
unlisted articles may still be missing.

## Scope of attribution

An archived report/preview URL containing the exact ESPN `gameId` can establish
that fixture link. Other links inferred only from current metadata remain
unverified candidates; final knockout opponents must not be backdated.

An archived headline explicitly mentioning `World Cup` can enter the separate
tournament-wide public-context stream only when it passes the shared 2026
scope filter. Explicit other editions (for example 2030), Club, Women, youth
and other-sport World Cups remain candidates. Generic headlines remain a
broad relevance heuristic, not proof of direct match relevance. That does not establish an individual
wallet's attention, beliefs, or reason for trading. A broad tournament context
stream also avoids treating present-day fixture assignments as historical facts.

## Verified examples

- ESPN article `48919840`, about Billy Gilmour's World Cup injury: publisher
  claims May 30, 2026; verified Wayback headline capture is June 1 at 02:47:26 UTC.
- ESPN game report `760415`, Mexico–South Africa: verified headline capture is
  June 18 at 04:01:44 UTC. This captured version cannot be used in the June 11
  match's earlier trading contexts.

Common Crawl's June and July indices returned no exact-URL captures for the
second example in this run. Historical GDELT GKG archives are reachable and can
supply archived `PAGE_TITLE` fields, but a full tournament GKG extraction has
not been performed. The rolling GDELT DOC API is not a substitute for older
April news coverage.

Sources: [Wayback CDX API](https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server),
[Common Crawl CDXJ index](https://commoncrawl.org/cdxj-index),
[GDELT archived page titles](https://blog.gdeltproject.org/gkg-2-0-now-includes-page-titles/).
