# News collection and attribution

The collector builds a source-attributed **metadata catalog**, not a certified
historical news corpus and not a record of what a wallet read. An old publication
date on a page fetched today does not prove that today's headline or article
version was visible at that time. All records produced by `collect_news` therefore
start with `historical_availability_verified=false`, `availability_upper_utc=null`,
and `feature_eligible=false`.

## Implemented sources

1. Every one of the 104 ESPN fixture summary endpoints:
   `https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/summary?event=EVENT_ID`.
   Only the `article` record with a matching `gameId` establishes a direct match
   association. The `news.articles` widget contains **current league headlines**,
   including stories months after the fixture, and is deliberately ignored.
2. The historical ESPN World Cup league feed:
   `https://content.core.api.espn.com/v1/sports/soccer/fifa.world/news?limit=50&offset=OFFSET`.
   Live verification on September 23, 2026 found `offset=2000` returning May 2026
   stories and `offset=3000` returning 2023 stories. Pagination checks the echoed
   offset and rejects repeated pages. The collection starts at offset zero and
   advances by the actual row count. It stops at an empty page, the configured
   page limit, an error, or two successive pages whose publication claims are
   entirely before the study start.

This undocumented public endpoint can change. Two old pages are a query stopping
rule, not a completeness certificate. Offsets can shift while a provider changes
its archive. Deduplication retains all observed metadata versions and response
provenance, but it cannot prove that shifting pagination did not omit a story.
A rerun with a refreshed client can compare article IDs and coverage.

The superficially similar `site.api.espn.com/.../news` endpoint ignored `dates`,
`before`, `page`, and `offset` during live probes, and capped the current feed at
50 items. It must not be represented as a historical archive. Guardian's public
`api-key=test` probe returned HTTP 401, and GDELT DOC probes returned HTTP 429 or
timed out; neither supplied this collection. No access controls were bypassed.

## Scope and fields

The default requested interval starts 90 days before the earliest observed market
opening and ends one day after the final kickoff. The initial live collection
explicitly requests January 1, 2026 onward to include the pre-market background
for the initial public prior. This is an explicit context collection
window; it is not a claim that all trading ended by that upper bound. Background earlier than the configured study start is outside this collection.
All returned metadata is preserved, including out-of-window records used to
check archive traversal. `within_study_window` separates them without discarding
provenance.

`news.jsonl` has one row per observed article metadata version. It includes:

- `news_id`, `news_item_id`, `source_article_id`, and `metadata_sha256`;
- headline, source URL, content type, publisher, and source team/game identifiers;
- raw and normalized publication claims, last modification, original posting,
  capture time, and response SHA256/URL provenance;
- candidate fixture links with their relationship, evidence, and confidence;
- explicit false flags for historical availability, wallet exposure, and causal
  attribution.

The metadata version hash covers article ID, title, URL, publication, and modified
timestamp. `version_rank` orders the observed metadata deterministically using
claimed modification time then capture time. It is **not a verified chronology
of historical article versions** (`version_order_historically_verified=false`).

The export omits article body, description, image captions, and embedded videos.
The immutable private/local HTTP response cache may contain full provider
responses, so it is ignored by git and must not be published as a news corpus.
No redistribution or model-training rights in article text are established by
public API access. Licensing must be assessed before adding full text to SFT.

## Relevance is distinct from historical availability

The linkage rules, in priority order, are:

1. An explicit ESPN `article.gameId` matching the fixture: high-confidence subject
   association. It may be a post-match recap and remains ineligible before its
   publication even if a historical version is later verified.
2. A national-team category matching one of the fixture teams, with the publication
   claim in `[market_open - 90 days, kickoff + 6 hours)`: medium-confidence team
   context, with a separate `team_category_background` relation before opening.
3. An exact, word-bounded team phrase from an explicit alias list in the headline,
   in the same window: low-confidence review candidate. Ambiguous geographic
   phrases can still produce false positives; these are never asserted as proven
   match news.

A tournament-only article remains `context_scope=tournament` with no fabricated
fixture assignments. It can form a separate global news timeline. Unmatched and
unknown-time articles remain in the export. Current final knockout opponents
must not be assumed known at earlier trading times; every fixture link is marked
`historical_link_verified=false` until independently verified.

`coverage.json` includes **all fixtures**, per-source failure status, candidate
counts, claimed pregame counts, historically verified counts, and errors.
A positive candidate count does not establish complete news coverage. A zero
candidate count does not establish that no news existed.

## Reproduce

The public Python API is dependency free:

```python
import json
from pathlib import Path
from poly_world_cup.http import HttpClient
from poly_world_cup.news import collect_news

registry = json.loads(Path("data/registry/registry.json").read_text())
report = collect_news(
    HttpClient(Path("data/news/cache"), timeout=25, retries=1),
    registry,
    Path("data/news"),
    workers=4,
    max_archive_pages=100,
    window_start="2026-01-01T00:00:00Z",
)
```

Reruns reuse the immutable cached captures. To collect a new current version,
use a client with `refresh=True`; existing cached raw bodies remain available.
The collector writes `news.jsonl`, `coverage.json`, `sources.json`, and a progress
checkpoint. Each run rewrites derived outputs from its collected source set; it
does not silently merge an unrelated archived-version corpus.

Before historical news becomes a training feature, retrieve the actual earlier
content version, record independently verifiable archive capture evidence and
content hashes, prove any fixture association was known by that time, and enforce
`availability_upper_utc < query_time`. Merely finding an archive URL must never
promote a current headline or current body to a historical feature.

## Recorded collection, September 23, 2026

The initial January 1–July 20, 2026 sweep captured 55 archive pages and queried all
104 fixture summary endpoints without a source error. It produced 2,751 unique
article metadata versions, including 2,291 with publication claims inside the
requested interval. There were 103 explicit game-ID recaps; fixture
`espn:760437` had no summary article. Team and background news nevertheless
provided candidate context for all 104 fixtures (12–532 candidate versions per
fixture). These counts describe retrieved metadata, not an exhaustive news set.

The base export contains 11,197 candidate fixture links: 5,770 category-based
background links, 397 headline-based background links, 4,611 category-based
post-opening links, 316 headline-based post-opening links, and 103 explicit game
links. There are 806 tournament-scope metadata records, preserved separately from
fixture attribution. All base-export records remain historically unverified;
independently collected archived versions, if available, must be assessed as
separate observations.
