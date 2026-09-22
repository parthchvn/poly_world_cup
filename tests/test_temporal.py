"""Tests of dataset validity invariants, using synthetic observations only."""

import unittest
from datetime import datetime, timedelta, timezone

from poly_world_cup.temporal import (
    CoverageInterval, available_before, build_historical_universe,
    occurrence_label, parse_utc, select_versions, validate_splits,
)


def stamp(minute):
    return (datetime(2026, 6, 11, 10, tzinfo=timezone.utc) + timedelta(minutes=minute)).isoformat()


def observation(minute=0, **extra):
    return {"event_time_utc": stamp(minute), "availability_upper_utc": stamp(minute),
            "availability_verified": True, **extra}


class AvailabilityTests(unittest.TestCase):
    def test_naive_timestamp_rejected_and_offsets_normalized(self):
        with self.assertRaises(ValueError):
            parse_utc("2026-06-11T10:00:00")
        self.assertEqual(parse_utc("2026-06-11T12:00:00+02:00"), parse_utc(stamp(0)))

    def test_event_time_does_not_establish_availability(self):
        row = observation(0, availability_upper_utc=stamp(20))
        self.assertFalse(available_before(row, stamp(10)))
        self.assertTrue(available_before(row, stamp(21)))

    def test_equality_unknown_and_unverified_fail_closed(self):
        for row in [observation(10), observation(0, availability_upper_utc=None),
                    observation(0, availability_verified=False), {"event_time_utc": stamp(0)}]:
            self.assertFalse(available_before(row, stamp(10)))

    def test_uncertainty_upper_bound_controls_inclusion(self):
        row = observation(0, availability_lower_utc=stamp(5), availability_upper_utc=stamp(15))
        self.assertFalse(available_before(row, stamp(10)))
        self.assertTrue(available_before(row, stamp(16)))
        with self.assertRaises(ValueError):
            available_before(observation(0, availability_lower_utc=stamp(1)), stamp(10))

    def test_entire_atomic_target_excluded(self):
        row = observation(0, atomic_id="chain:transaction")
        self.assertFalse(available_before(row, stamp(10), excluded_atomic_ids={"chain:transaction"}))
        self.assertTrue(available_before(row, stamp(10), excluded_atomic_ids={"other"}))

    def test_future_extension_and_input_order_do_not_change_versions(self):
        original = [observation(0, item_id="news", version_rank=0, text="Original report")]
        future = observation(20, item_id="news", version_rank=1, text="Later result")
        expected = select_versions(original, stamp(10))
        self.assertEqual(expected, select_versions([future] + original, stamp(10)))
        self.assertEqual(expected, select_versions(original + [future], stamp(10)))
        self.assertEqual("Later result", select_versions(original + [future], stamp(21))[0]["text"])

    def test_source_version_order_is_distinct_from_capture_order(self):
        rows = [observation(9, item_id="news", version_rank=0),
                observation(8, item_id="news", version_rank=1)]
        self.assertEqual(1, select_versions(rows, stamp(10))[0]["version_rank"])

    def test_conflicting_versions_rejected(self):
        a = observation(0, item_id="news", version_rank=0, text="one")
        b = {**a, "text": "two"}
        with self.assertRaises(ValueError):
            select_versions([a, b], stamp(10))

    def test_future_knockout_identity_cannot_replace_placeholder(self):
        rows = [observation(0, item_id="fixture", version_rank=0, teams=["Winner A", "Runner-up B"]),
                observation(20, item_id="fixture", version_rank=1, teams=["Actual A", "Actual B"])]
        self.assertEqual(["Winner A", "Runner-up B"], select_versions(rows, stamp(10))[0]["teams"])


class UniverseTests(unittest.TestCase):
    def test_future_wallet_and_late_publication_do_not_change_membership(self):
        earlier = observation(0, wallet="known")
        later = observation(15, wallet="future")
        late_publication = observation(0, wallet="not_yet_observed", availability_upper_utc=stamp(15))
        expected = build_historical_universe([earlier], stamp(10), source_scope="global_prefix")
        self.assertEqual(expected, build_historical_universe([earlier, later, late_publication], stamp(10), source_scope="global_prefix"))

    def test_eventual_participants_scope_rejected(self):
        with self.assertRaises(ValueError):
            build_historical_universe([], stamp(10), source_scope="eventual_world_cup_wallets")

    def test_lookback_is_historical_and_left_inclusive(self):
        rows = [observation(0, wallet="at_boundary"), observation(-1, wallet="too_old")]
        result = build_historical_universe(rows, stamp(10), lookback=timedelta(minutes=10), source_scope="global_prefix")
        self.assertEqual(frozenset({"at_boundary"}), result.wallets)

    def test_impossible_execution_availability_rejected(self):
        with self.assertRaises(ValueError):
            build_historical_universe([observation(5, wallet="w", availability_upper_utc=stamp(0))], stamp(10), source_scope="global_prefix")


class OccurrenceTests(unittest.TestCase):
    def setUp(self):
        self.universe = build_historical_universe([observation(0, wallet="w")], stamp(10), source_scope="global_prefix")
        self.full = [CoverageInterval(stamp(10), stamp(15), "w", "f", complete=True)]

    def label(self, actions=(), coverage=None, **kwargs):
        arguments = dict(wallet="w", fixture_id="f", query_time=stamp(10), horizon=timedelta(minutes=5),
                         coverage=self.full if coverage is None else coverage, universe=self.universe)
        arguments.update(kwargs)
        return occurrence_label(actions, **arguments)

    def action(self, minute, **kwargs):
        return {"action_id": str(minute), "wallet": "w", "fixture_id": "f", "role": "taker",
                "event_time_utc": stamp(minute), **kwargs}

    def test_no_trade_requires_complete_full_horizon(self):
        self.assertEqual("NO_OBSERVED_TRADE", self.label()["status"])
        self.assertEqual("CENSORED", self.label(coverage=[])["status"])
        gap = [CoverageInterval(stamp(10), stamp(12), "w", "f", complete=True),
               CoverageInterval(stamp(13), stamp(15), "w", "f", complete=True)]
        self.assertEqual("CENSORED", self.label(coverage=gap)["status"])

    def test_contiguous_intervals_merge(self):
        contiguous = [CoverageInterval(stamp(12), stamp(15), "w", "f", complete=True),
                      CoverageInterval(stamp(9), stamp(12), "w", "f", complete=True)]
        self.assertEqual("NO_OBSERVED_TRADE", self.label(coverage=contiguous)["status"])

    def test_scope_and_target_coverage_must_match(self):
        for coverage in [[CoverageInterval(stamp(10), stamp(15), "other", "f", complete=True)],
                         [CoverageInterval(stamp(10), stamp(15), "w", "other", complete=True)],
                         [CoverageInterval(stamp(10), stamp(15), "w", "f", complete=True, target_kind="maker")],
                         [CoverageInterval(stamp(10), stamp(15), "w", "f", complete=False)]]:
            self.assertEqual("CENSORED", self.label(coverage=coverage)["status"])
        all_coverage = [CoverageInterval(stamp(10), stamp(15), "*", "*", complete=True, target_kind="all")]
        self.assertEqual("NO_OBSERVED_TRADE", self.label(coverage=all_coverage)["status"])

    def test_horizon_is_half_open(self):
        result = self.label([self.action(10), self.action(15), self.action(9)])
        self.assertEqual("TRADE", result["status"])
        self.assertEqual(["10"], result["first_action_ids"])
        self.assertEqual(1, result["action_count"])
        self.assertEqual("NO_OBSERVED_TRADE", self.label([self.action(15)])["status"])

    def test_positive_with_gap_is_censored_as_first_action_is_unknown(self):
        gap = [CoverageInterval(stamp(12), stamp(15), "w", "f", complete=True)]
        self.assertEqual("CENSORED", self.label([self.action(13)], coverage=gap)["status"])

    def test_same_timestamp_order_is_explicitly_ambiguous(self):
        result = self.label([self.action(11, action_id="b"), self.action(11, action_id="a")])
        self.assertEqual(["a", "b"], result["first_action_ids"])
        self.assertTrue(result["ordering_ambiguous"])

    def test_duplicates_do_not_increase_action_count(self):
        action = self.action(11)
        self.assertEqual(1, self.label([action, action])["action_count"])
        with self.assertRaises(ValueError):
            self.label([action, {**action, "extra": "conflict"}])

    def test_unknown_roles_cannot_become_taker_negatives(self):
        result = self.label([self.action(11, role="unknown")])
        self.assertEqual("CENSORED", result["status"])
        self.assertEqual("unclassified_execution_role", result["censor_reason"])

    def test_universe_cannot_be_reused_at_another_query(self):
        with self.assertRaises(ValueError):
            self.label(query_time=stamp(11))
        with self.assertRaises(ValueError):
            self.label(wallet="future")


class SplitTests(unittest.TestCase):
    def row(self, split, start, end, fixture=None, identity=None):
        return {"example_id": identity or split, "fixture_id": fixture or split,
                "split": split, "query_time_utc": stamp(start), "label_end_utc": stamp(end)}

    def validate(self, rows):
        return validate_splits(rows, train_cutoff=stamp(10), validation_cutoff=stamp(20))

    def test_disjoint_fixtures_and_global_times_pass(self):
        self.assertEqual([], self.validate([self.row("train", 5, 10), self.row("validation", 10, 15),
                                           self.row("test", 20, 25)]))

    def test_fixture_split_alone_does_not_allow_future_training(self):
        errors = self.validate([self.row("train", 11, 16), self.row("validation", 10, 15)])
        self.assertTrue(any("training overlaps" in error for error in errors))

    def test_horizon_crossing_boundary_is_rejected(self):
        self.assertTrue(self.validate([self.row("train", 9, 14)]))
        self.assertTrue(self.validate([self.row("validation", 19, 24)]))

    def test_same_fixture_across_splits_is_rejected(self):
        errors = self.validate([self.row("train", 0, 5, fixture="f"),
                                self.row("test", 20, 25, fixture="f")])
        self.assertTrue(any("fixture appears" in error for error in errors))

    def test_invalid_ids_and_point_target_precision_are_rejected(self):
        self.assertTrue(self.validate([self.row("train", 0, 0)]))
        self.assertTrue(self.validate([self.row("train", 0, 5), self.row("train", 0, 5)]))
        self.assertTrue(self.validate([self.row("test", 19, 24)]))


if __name__ == "__main__":
    unittest.main()
