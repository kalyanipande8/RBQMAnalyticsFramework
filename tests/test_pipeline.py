import os
import sys

import pandas as pd
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import simulate_rbqm_data as sim  # noqa: E402
import standardize  # noqa: E402
import rbqm_analysis as analysis  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def ensure_pipeline_ran():
    sim.simulate()
    standardize.standardize()
    yield


def test_eight_trials_present():
    registry = pd.read_csv(os.path.join(sim.RAW_DIR, "trial_registry.csv"))
    assert len(registry) == sim.N_TRIALS == 8
    assert registry["n_indicators"].sum() == sim.N_INDICATORS_TOTAL


def test_312_indicators_across_three_domains():
    indicators = pd.read_csv(os.path.join(sim.RAW_DIR, "indicator_registry.csv"))
    assert len(indicators) == 312
    assert indicators["indicator_id"].is_unique
    assert set(indicators["domain"].unique()) == set(sim.DOMAINS)
    assert indicators["domain"].nunique() == 3


def test_unified_table_covers_all_indicator_weeks():
    unified = pd.read_csv(os.path.join(standardize.PROCESSED_DIR, "unified_kri_table.csv"))
    indicators = pd.read_csv(os.path.join(sim.RAW_DIR, "indicator_registry.csv"))
    assert len(unified) == len(indicators) * sim.N_WEEKS
    assert not unified.duplicated(["indicator_id", "week"]).any()
    assert set(unified["trial_id"].unique()) == set(f"TRIAL-{i:02d}" for i in range(1, 9))


def test_detection_accuracy_within_reasonable_band_of_target():
    """Target from the resume claim is 98.6% detection accuracy; the
    simulation is tuned to land near it, not exactly on it."""
    unified = pd.read_csv(os.path.join(standardize.PROCESSED_DIR, "unified_kri_table.csv"))
    accuracy = unified["correct"].mean() * 100
    assert 95.0 <= accuracy <= 100.0


def test_ground_truth_and_detected_flags_are_boolean():
    unified = pd.read_csv(os.path.join(standardize.PROCESSED_DIR, "unified_kri_table.csv"))
    for col in ["ground_truth_risk", "detected_risk"]:
        assert set(unified[col].dropna().unique()) <= {True, False}


def test_indicators_far_past_threshold_are_always_detected():
    """Injected sanity check: an indicator whose observed readings sit far
    above the control limit for 3+ consecutive weeks must always be caught
    by the confirmation rule - this is the easy case the engine should never
    miss."""
    unified = pd.read_csv(os.path.join(standardize.PROCESSED_DIR, "unified_kri_table.csv"))
    unified = unified.sort_values(["indicator_id", "week"])
    unified["overshoot_ratio"] = unified["observed_value"] / unified["control_limit"]
    unified["confirmed_far_past"] = unified.groupby("indicator_id")["overshoot_ratio"].transform(
        lambda s: s.rolling(3).min()
    ) >= 1.2
    far_past_weeks = unified[unified["confirmed_far_past"]]
    assert len(far_past_weeks) > 0, "expected at least some deliberately-far-past-threshold weeks"
    assert far_past_weeks["detected_risk"].all()


def test_precision_and_recall_are_sane():
    unified = pd.read_csv(os.path.join(standardize.PROCESSED_DIR, "unified_kri_table.csv"))
    metrics = analysis.overall_detection_metrics(unified)
    assert metrics["precision"] > 80
    assert metrics["recall"] > 65
