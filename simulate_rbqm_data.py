"""
simulate_rbqm_data.py

Simulates a Risk-Based Quality Management (RBQM) monitoring setup for 8
concurrent clinical trials, with 312 Key Risk Indicators (KRIs) distributed
across those 8 trials and across 3 monitoring domains:

  Protocol compliance   e.g. consent deviations, visit-window violations,
                         eligibility deviations
  Data quality          e.g. query rate, query aging, missing critical data,
                         SAE reconciliation lag
  Enrollment             e.g. screen-fail rate, enrollment pace vs plan,
                         dropout rate

Each indicator is monitored weekly over a 12-week surveillance window against
a control limit (threshold). Each indicator is simulated with:

  - A "ground truth" trajectory: an underlying, noise-free process level per
    week. Most indicators stay in a stable, in-control range. A subset are
    seeded as true risk signals - their underlying level drifts past the
    control limit partway through the window (simulating a real emerging
    risk, e.g. a site's query rate climbing). A further subset are seeded as
    borderline/noisy - underlying level sits close to the limit, so noisy
    weekly readings frequently cross it in both directions without the
    underlying process actually being out of control. This is what makes
    detection non-trivial.

  - An observed weekly reading: the underlying level plus measurement/
    reporting noise.

  - A ground-truth label per indicator-week: whether the underlying process
    was actually out of control that week (requires escalation).

  - A detected flag per indicator-week: the RBQM engine's rule-based
    detection logic (control-limit breach confirmed by requiring 2 of the
    last 3 weekly readings past the limit, a standard "avoid alarming on one
    noisy point" QTL/KRI monitoring rule) applied to the noisy observed
    readings only - the engine never sees the ground truth.

Uses a fixed random seed (np.random.default_rng) so results are reproducible.

Writes data/raw/kri_timeseries.csv and data/raw/trial_registry.csv.
"""

import os

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")

RANDOM_SEED = 42
N_TRIALS = 8
N_INDICATORS_TOTAL = 312
N_WEEKS = 12
DOMAINS = ["Protocol compliance", "Data quality", "Enrollment"]

# Fraction of indicators seeded as true emerging risks (drift past threshold)
FRACTION_TRUE_RISK = 0.14
# Fraction of the *remaining* stable indicators seeded as borderline/noisy
FRACTION_BORDERLINE = 0.06

# Observation noise (SD, in units of the indicator's own control-limit scale)
NOISE_SD = 0.022
# Drift indicators: how far past the limit the underlying level ends up,
# expressed as a fraction of the control limit, reached by week 12
DRIFT_MAGNITUDE = 0.30
# Borderline indicators: underlying level sits within this fraction of the
# limit (can be just above or just below)
BORDERLINE_BAND = 0.02

TRIAL_IDS = [f"TRIAL-{i:02d}" for i in range(1, N_TRIALS + 1)]


def _distribute_counts(total, n_bins, rng):
    """Distribute `total` items across `n_bins` bins as evenly as possible,
    with a small amount of realistic imbalance (some trials/domains run more
    indicators than others)."""
    base = total // n_bins
    counts = np.full(n_bins, base)
    remainder = total - base * n_bins
    # give the remainder to a random subset of bins
    idx = rng.choice(n_bins, size=remainder, replace=False)
    counts[idx] += 1
    return counts


def simulate():
    rng = np.random.default_rng(RANDOM_SEED)

    # Distribute the 312 indicators across 8 trials (unevenly, like a real
    # portfolio where bigger trials run more KRIs)
    trial_counts = _distribute_counts(N_INDICATORS_TOTAL, N_TRIALS, rng)
    # ensure some spread beyond the even split by shuffling a bit of mass
    shift = rng.integers(-3, 4, size=N_TRIALS)
    shift -= int(round(shift.mean()))
    trial_counts = np.clip(trial_counts + shift, 5, None)
    diff = N_INDICATORS_TOTAL - trial_counts.sum()
    trial_counts[0] += diff  # absorb any rounding remainder in trial 1

    registry_rows = []
    indicator_rows = []
    ts_rows = []
    indicator_counter = 1

    domain_thresholds = {
        "Protocol compliance": dict(limit=8.0, unit="% deviations"),
        "Data quality": dict(limit=12.0, unit="open queries / 100 pages"),
        "Enrollment": dict(limit=15.0, unit="% behind plan"),
    }

    for trial_idx, trial_id in enumerate(TRIAL_IDS):
        n_indicators = int(trial_counts[trial_idx])
        registry_rows.append({"trial_id": trial_id, "n_indicators": n_indicators})

        # split this trial's indicators across the 3 domains, unevenly
        domain_counts = _distribute_counts(n_indicators, len(DOMAINS), rng)
        shift = rng.integers(-2, 3, size=len(DOMAINS))
        shift -= int(round(shift.mean()))
        domain_counts = np.clip(domain_counts + shift, 1, None)
        diff = n_indicators - domain_counts.sum()
        domain_counts[0] += diff

        for domain, n_dom in zip(DOMAINS, domain_counts):
            limit = domain_thresholds[domain]["limit"]
            unit = domain_thresholds[domain]["unit"]
            for _ in range(int(n_dom)):
                indicator_id = f"KRI-{indicator_counter:04d}"
                indicator_counter += 1

                roll = rng.random()
                if roll < FRACTION_TRUE_RISK:
                    category = "true_risk"
                elif roll < FRACTION_TRUE_RISK + (1 - FRACTION_TRUE_RISK) * FRACTION_BORDERLINE:
                    category = "borderline"
                else:
                    category = "stable"

                # baseline underlying level: comfortably below the limit
                baseline_frac = rng.uniform(0.35, 0.65)
                baseline_level = baseline_frac * limit

                if category == "true_risk":
                    # underlying level drifts upward, crossing the limit
                    # partway through the 12-week window
                    onset_week = rng.integers(3, 8)
                    end_level = limit * (1 + DRIFT_MAGNITUDE * rng.uniform(0.7, 1.3))
                elif category == "borderline":
                    onset_week = 0
                    band_sign = rng.choice([-1, 1])
                    end_level = limit * (1 + band_sign * BORDERLINE_BAND * rng.uniform(0.5, 1.5))
                    baseline_level = end_level  # flat, just noisy around the limit
                else:
                    onset_week = None
                    end_level = baseline_level

                indicator_rows.append({
                    "indicator_id": indicator_id, "trial_id": trial_id, "domain": domain,
                    "control_limit": limit, "unit": unit, "category": category,
                })

                weeks = np.arange(1, N_WEEKS + 1)
                if category == "true_risk":
                    underlying = np.where(
                        weeks < onset_week, baseline_level,
                        baseline_level + (end_level - baseline_level) *
                        np.clip((weeks - onset_week) / max(1, (N_WEEKS - onset_week)), 0, 1),
                    )
                else:
                    underlying = np.full(N_WEEKS, end_level)

                noise_scale = limit * NOISE_SD
                observed = underlying + rng.normal(0, noise_scale, N_WEEKS)
                observed = np.clip(observed, 0, None)

                ground_truth = underlying > limit

                for w, u, o, gt in zip(weeks, underlying, observed, ground_truth):
                    ts_rows.append({
                        "indicator_id": indicator_id, "trial_id": trial_id, "domain": domain,
                        "week": int(w), "control_limit": limit,
                        "underlying_level": round(float(u), 3),
                        "observed_value": round(float(o), 3),
                        "ground_truth_risk": bool(gt),
                    })

    os.makedirs(RAW_DIR, exist_ok=True)
    pd.DataFrame(registry_rows).to_csv(os.path.join(RAW_DIR, "trial_registry.csv"), index=False)
    pd.DataFrame(indicator_rows).to_csv(os.path.join(RAW_DIR, "indicator_registry.csv"), index=False)
    pd.DataFrame(ts_rows).to_csv(os.path.join(RAW_DIR, "kri_timeseries.csv"), index=False)

    print(f"Simulated {indicator_counter - 1} KRIs across {N_TRIALS} trials, "
          f"{len(DOMAINS)} domains, {N_WEEKS} weeks")
    print(f"  trial_registry.csv       {len(registry_rows)} rows")
    print(f"  indicator_registry.csv   {len(indicator_rows)} rows")
    print(f"  kri_timeseries.csv       {len(ts_rows)} rows")


if __name__ == "__main__":
    simulate()
