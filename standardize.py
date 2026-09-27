"""
standardize.py

Applies the RBQM engine's detection logic to the raw weekly KRI readings and
builds one unified indicator-period table across all 8 trials, 312
indicators, and 3 domains, with both the ground-truth label and the engine's
detected-flag column.

Detection rule (control-limit / trend-based rule engine, not ML - this is
how real RBQM platforms such as risk-based monitoring dashboards work):
an indicator-week is flagged as a detected risk signal if at least 2 of the
last 3 observed weekly readings (this week and the two before it) are past
the control limit. This is a standard "don't alarm on one noisy point"
confirmation rule, applied only to the observed (noisy) readings - the
engine never sees the underlying ground truth.

Also computes per-trial and per-domain risk summaries (count of ground-truth
risk indicator-weeks, count of detected flags, by trial and by domain).

Writes data/processed/unified_kri_table.csv,
data/processed/trial_risk_summary.csv, data/processed/domain_risk_summary.csv.
"""

import os

import numpy as np
import pandas as pd

from simulate_rbqm_data import RAW_DIR

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")

CONFIRMATION_WINDOW = 3
CONFIRMATION_MIN_BREACHES = 2


def apply_detection_rule(df):
    """For each indicator, flag week w if >= CONFIRMATION_MIN_BREACHES of the
    last CONFIRMATION_WINDOW observed readings (weeks w-2..w) are past the
    control limit."""
    df = df.sort_values(["indicator_id", "week"]).copy()
    df["breach"] = (df["observed_value"] > df["control_limit"]).astype(int)

    breach_roll = df.groupby("indicator_id")["breach"].transform(
        lambda s: s.rolling(window=CONFIRMATION_WINDOW, min_periods=1).sum()
    )
    df["detected_risk"] = breach_roll >= CONFIRMATION_MIN_BREACHES
    return df


def standardize():
    ts = pd.read_csv(os.path.join(RAW_DIR, "kri_timeseries.csv"))
    indicators = pd.read_csv(os.path.join(RAW_DIR, "indicator_registry.csv"))

    unified = apply_detection_rule(ts)
    unified["ground_truth_risk"] = unified["ground_truth_risk"].astype(bool)
    unified["detected_risk"] = unified["detected_risk"].astype(bool)
    unified["true_positive"] = unified["ground_truth_risk"] & unified["detected_risk"]
    unified["false_positive"] = (~unified["ground_truth_risk"]) & unified["detected_risk"]
    unified["false_negative"] = unified["ground_truth_risk"] & (~unified["detected_risk"])
    unified["true_negative"] = (~unified["ground_truth_risk"]) & (~unified["detected_risk"])
    unified["correct"] = unified["ground_truth_risk"] == unified["detected_risk"]

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    unified.round(3).to_csv(os.path.join(PROCESSED_DIR, "unified_kri_table.csv"), index=False)

    # Per-trial summary
    trial_summary = unified.groupby("trial_id").agg(
        n_indicator_weeks=("indicator_id", "count"),
        n_indicators=("indicator_id", "nunique"),
        ground_truth_risk_weeks=("ground_truth_risk", "sum"),
        detected_risk_weeks=("detected_risk", "sum"),
        accuracy=("correct", "mean"),
    ).reset_index()
    trial_summary["accuracy_pct"] = (trial_summary["accuracy"] * 100).round(2)
    trial_summary.to_csv(os.path.join(PROCESSED_DIR, "trial_risk_summary.csv"), index=False)

    # Per-domain summary
    domain_summary = unified.groupby("domain").agg(
        n_indicator_weeks=("indicator_id", "count"),
        n_indicators=("indicator_id", "nunique"),
        ground_truth_risk_weeks=("ground_truth_risk", "sum"),
        detected_risk_weeks=("detected_risk", "sum"),
        accuracy=("correct", "mean"),
    ).reset_index()
    domain_summary["accuracy_pct"] = (domain_summary["accuracy"] * 100).round(2)
    domain_summary.to_csv(os.path.join(PROCESSED_DIR, "domain_risk_summary.csv"), index=False)

    return unified, trial_summary, domain_summary


def main():
    unified, trial_summary, domain_summary = standardize()
    n_indicators = unified["indicator_id"].nunique()
    n_trials = unified["trial_id"].nunique()
    n_domains = unified["domain"].nunique()
    overall_acc = unified["correct"].mean() * 100
    print(f"Standardized {len(unified)} indicator-week records across "
          f"{n_indicators} KRIs, {n_trials} trials, {n_domains} domains")
    print(f"  Overall detection accuracy: {overall_acc:.2f}%")
    print(f"  Wrote data/processed/unified_kri_table.csv")
    print(f"  Wrote data/processed/trial_risk_summary.csv")
    print(f"  Wrote data/processed/domain_risk_summary.csv")


if __name__ == "__main__":
    main()
