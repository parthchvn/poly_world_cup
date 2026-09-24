# Fresh capture recovery inputs

The archive has 28 ordered parts, each at most 8 MiB, to fit GitHub API uploads.

This archive preserves the fresh 19 replacement-contract API captures and 1,700 exactly-20 actor–contract queries used to rebuild the actor-sequence dataset. It contains immutable raw response caches, normalized pages, cursor manifests, capture reports, and recovery audits. It does not contain `source.sqlite`, the original 3.35 GB filtered archive, or a complete set of histories for excluded actor–contract pairs across all 312 contracts. The 19 full-contract captures do include their excluded high-activity pairs.

The original `world_cup_lt20.sqlite` and its `MANIFEST.json` are required separately. Their verified SHA256 is recorded in this package manifest. The complete published count ledger and registry remain in `datasets/world_cup_2026_tournament_lt20_v2_evidence`.

From the repository root, first verify the parts inside this directory using `sha256sum -c CHECKSUMS.sha256`. Then concatenate parts in numeric order, extract into a new recovery directory, and rebuild offline:

```sh
cat datasets/world_cup_2026_actor_sequences_v3_recovery/capture_inputs.tar.gz.part* > /tmp/worldcup_capture_inputs.tar.gz
mkdir -p data/sequence_v3_recovery
tar -xzf /tmp/worldcup_capture_inputs.tar.gz -C data/sequence_v3_recovery
python scripts/recover_sequence_trades.py --archive /path/to/world_cup_lt20.sqlite
```

The recovery directory must not already contain a published `source.sqlite`. For a new online collection instead, add `--collect` to the recovery command. Every selected actor–contract pair must exactly match the full count ledger before publication. Inherited 293-contract normalized evidence is never described as newly raw-replayed. Coverage bounds are the narrower selected-cohort observed interval. API exhaustion and repeated observations are not proofs of canonical fill identity, deliberate inactivity, or chain completeness.
