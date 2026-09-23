# Historical GDELT headline evidence

Source: The GDELT Project, https://www.gdeltproject.org/ .
These records are selected from GDELT's public GKG 2.1 archive. Credit and terms:
https://www.gdeltproject.org/about.html#termsofuse .

`candidate_news.jsonl.gz` preserves the collected headline candidates. Fixture
relevance is established separately. `records/` contains the exact selected GKG
metadata records, with source ZIP SHA256, one-based row number and record SHA256.
The record hash covers its exact UTF-8 bytes without the line ending. These raw
records are audit evidence and must not be used as SFT prompts. Article bodies
and full source ZIP files are not included. To verify against the upstream
archive, download the listed ZIP, check its checksum, and compare the indexed row.
`batches.jsonl.gz` records every requested batch and error. `invalid_records`
records rejected, invalid-UTF8 rows by archive identity and hash.

The search starts with three-hour sampling and adds denser windows around
fixtures with missing context. Its exact visited batch schedule is recorded.
This does not assert that all fifteen-minute batches or all relevant news were
collected. Existing headlines retain their archive version and never use today's
article text or a publisher's claimed publication date for historical timing.

Availability uses the GKG batch label plus fifteen minutes, a conservative
convention for GKG's documented fifteen-minute seen/processed resolution. The
raw batch label remains separate. This is not an exact page-capture timestamp.
Use only news whose availability upper bound is strictly before the query time.

Official format and timestamp sources:
- https://data.gdeltproject.org/documentation/GDELT-Global_Knowledge_Graph_Codebook-V2.1.pdf
- https://blog.gdeltproject.org/gkg-2-0-now-includes-page-titles/
- https://blog.gdeltproject.org/announcing-our-first-api-gkg-geojson/
- https://blog.gdeltproject.org/a-behind-the-scenes-look-at-how-we-think-about-master-file-formats-and-timestamping/
