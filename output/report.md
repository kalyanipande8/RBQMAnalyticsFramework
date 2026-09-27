# RBQM Analytics Framework Report

> All figures and numbers below come directly from running `python run_all.py` on simulated risk-monitoring data. Nothing is hardcoded.

## Portfolio overview

- **8 concurrent trials**, **312 risk indicators (KRIs)** total, across **3 domains** (protocol compliance, data quality, enrollment).
- **12 weekly monitoring periods** per indicator -> **3,744 indicator-week observations** analyzed.

## Overall detection performance

- **Detection accuracy: 98.48%** (3,687 of 3,744 indicator-weeks correctly classified against ground truth).
- **Precision: 95.36%** (of indicator-weeks flagged by the engine, this fraction were true risk signals).
- **Recall: 74.23%** (of true risk signals, this fraction were caught by the engine).
- **F1 score: 83.48%**

### Confusion matrix

| | Detected: no risk | Detected: risk |
|---|---|---|
| **Ground truth: no risk** | 3,543 (TN) | 7 (FP) |
| **Ground truth: risk** | 50 (FN) | 144 (TP) |

## Risk signals by domain

| Domain | Indicator-weeks | Ground-truth risk weeks | Detected risk weeks | Domain accuracy |
|---|---|---|---|---|
| Enrollment | 1,296 | 70 | 49 | 98.38% |
| Protocol compliance | 1,236 | 69 | 53 | 98.54% |
| Data quality | 1,212 | 55 | 49 | 98.51% |

## Risk signals by trial

| Trial | Indicators | Ground-truth risk weeks | Detected risk weeks | Trial accuracy |
|---|---|---|---|---|
| TRIAL-06 | 42 | 46 | 45 | 97.02% |
| TRIAL-02 | 41 | 31 | 20 | 97.76% |
| TRIAL-08 | 40 | 30 | 21 | 98.12% |
| TRIAL-01 | 35 | 26 | 21 | 98.81% |
| TRIAL-04 | 39 | 20 | 15 | 98.93% |
| TRIAL-03 | 40 | 15 | 12 | 99.38% |
| TRIAL-05 | 39 | 14 | 9 | 98.93% |
| TRIAL-07 | 36 | 12 | 8 | 99.07% |

## Current escalation summary (week 12, most recent monitoring period)

**48 of 312 indicators** are currently flagged by the RBQM engine for risk-review-committee escalation as of the most recent weekly snapshot.

By domain: Data quality = 13, Enrollment = 19, Protocol compliance = 16

By trial: TRIAL-01 = 5, TRIAL-02 = 8, TRIAL-03 = 4, TRIAL-04 = 3, TRIAL-05 = 5, TRIAL-06 = 12, TRIAL-07 = 4, TRIAL-08 = 7

## Figures

- `output/figures/confusion_matrix.png`
- `output/figures/risk_signals_by_domain.png`
- `output/figures/risk_signals_by_trial_heatmap.png`
