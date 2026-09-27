"""
rbqm_analysis.py

Analyzes the standardized indicator-period table (standardize.py) to
characterize how well the RBQM engine's rule-based detection logic performs
against the simulated ground truth, and where the real risk signals in the
8-trial portfolio concentrate.

Computes:
  1. Overall detection accuracy, precision, recall, and a confusion matrix
     (flag vs. ground truth), across all 312 x 12 = 3,744 indicator-weeks.
  2. Risk-signal breakdown by domain and by trial - which domains/trials
     carry the most true ground-truth risk signals, and how many the engine
     currently has flagged.
  3. Escalation summary: indicators currently flagged for action in the most
     recent monitoring week (week 12), the point-in-time view a risk review
     committee would actually look at.

Produces:
  output/figures/confusion_matrix.png
  output/figures/risk_signals_by_domain.png
  output/figures/risk_signals_by_trial_heatmap.png
  output/report.md
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load():
    return pd.read_csv(os.path.join(PROCESSED_DIR, "unified_kri_table.csv"))


def overall_detection_metrics(df):
    tp = int(df["true_positive"].sum())
    fp = int(df["false_positive"].sum())
    fn = int(df["false_negative"].sum())
    tn = int(df["true_negative"].sum())
    n = len(df)
    accuracy = (tp + tn) / n * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) else float("nan")
    recall = tp / (tp + fn) * 100 if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, n=n, accuracy=accuracy,
                precision=precision, recall=recall, f1=f1)


def risk_breakdown_by_domain(df):
    g = df.groupby("domain").agg(
        n_indicator_weeks=("indicator_id", "count"),
        ground_truth_risk_weeks=("ground_truth_risk", "sum"),
        detected_risk_weeks=("detected_risk", "sum"),
        true_positives=("true_positive", "sum"),
        false_positives=("false_positive", "sum"),
        false_negatives=("false_negative", "sum"),
        accuracy=("correct", "mean"),
    ).reset_index()
    g["accuracy_pct"] = (g["accuracy"] * 100).round(2)
    return g.sort_values("ground_truth_risk_weeks", ascending=False)


def risk_breakdown_by_trial(df):
    g = df.groupby("trial_id").agg(
        n_indicators=("indicator_id", "nunique"),
        n_indicator_weeks=("indicator_id", "count"),
        ground_truth_risk_weeks=("ground_truth_risk", "sum"),
        detected_risk_weeks=("detected_risk", "sum"),
        accuracy=("correct", "mean"),
    ).reset_index()
    g["accuracy_pct"] = (g["accuracy"] * 100).round(2)
    return g.sort_values("ground_truth_risk_weeks", ascending=False)


def trial_domain_heatmap_data(df):
    pivot = df.groupby(["trial_id", "domain"])["detected_risk"].sum().unstack("domain").fillna(0)
    return pivot.reindex(sorted(pivot.index), axis=0)[["Protocol compliance", "Data quality", "Enrollment"]]


def escalation_summary(df, week=12):
    current = df[df["week"] == week]
    n_flagged = int(current["detected_risk"].sum())
    n_total = current["indicator_id"].nunique()
    by_domain = current.groupby("domain")["detected_risk"].sum().astype(int)
    by_trial = current.groupby("trial_id")["detected_risk"].sum().astype(int)
    return n_flagged, n_total, by_domain, by_trial


def make_figures(metrics, domain_summary, heatmap_data):
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Figure 1: confusion matrix
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No risk", "Risk"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No risk", "Risk"])
    ax.set_xlabel("Detected (engine flag)")
    ax.set_ylabel("Ground truth")
    ax.set_title(f"Detection confusion matrix (accuracy = {metrics['accuracy']:.1f}%)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8, label="indicator-weeks")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Figure 2: risk signals by domain (grouped bar: ground truth vs detected)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(domain_summary))
    width = 0.35
    ax.bar(x - width / 2, domain_summary["ground_truth_risk_weeks"], width,
          label="Ground-truth risk weeks", color=PALETTE[0])
    ax.bar(x + width / 2, domain_summary["detected_risk_weeks"], width,
          label="Detected risk weeks", color=PALETTE[1])
    ax.set_xticks(x); ax.set_xticklabels(domain_summary["domain"])
    ax.set_ylabel("Indicator-weeks")
    ax.set_title("Risk signals by domain: ground truth vs. detected")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "risk_signals_by_domain.png"), dpi=150)
    plt.close(fig)

    # Figure 3: heatmap of detected risk signals by trial x domain
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(heatmap_data.values, cmap="OrRd", aspect="auto")
    ax.set_xticks(np.arange(len(heatmap_data.columns))); ax.set_xticklabels(heatmap_data.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(heatmap_data.index))); ax.set_yticklabels(heatmap_data.index)
    ax.set_title("Detected risk signals by trial x domain")
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            val = heatmap_data.values[i, j]
            ax.text(j, i, f"{int(val)}", ha="center", va="center",
                   color="white" if val > heatmap_data.values.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="detected risk indicator-weeks")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "risk_signals_by_trial_heatmap.png"), dpi=150)
    plt.close(fig)


def write_report(metrics, domain_summary, trial_summary, n_flagged, n_total, esc_by_domain, esc_by_trial,
                 n_trials, n_indicators, n_domains):
    lines = [
        "# RBQM Analytics Framework Report",
        "",
        "> All figures and numbers below come directly from running `python run_all.py` "
        "on simulated risk-monitoring data. Nothing is hardcoded.",
        "",
        f"## Portfolio overview",
        "",
        f"- **{n_trials} concurrent trials**, **{n_indicators} risk indicators (KRIs)** total, "
        f"across **{n_domains} domains** (protocol compliance, data quality, enrollment).",
        f"- **12 weekly monitoring periods** per indicator -> "
        f"**{metrics['n']:,} indicator-week observations** analyzed.",
        "",
        "## Overall detection performance",
        "",
        f"- **Detection accuracy: {metrics['accuracy']:.2f}%** "
        f"({metrics['tp'] + metrics['tn']:,} of {metrics['n']:,} indicator-weeks correctly "
        f"classified against ground truth).",
        f"- **Precision: {metrics['precision']:.2f}%** (of indicator-weeks flagged by the engine, "
        f"this fraction were true risk signals).",
        f"- **Recall: {metrics['recall']:.2f}%** (of true risk signals, this fraction were caught "
        f"by the engine).",
        f"- **F1 score: {metrics['f1']:.2f}%**",
        "",
        "### Confusion matrix",
        "",
        "| | Detected: no risk | Detected: risk |",
        "|---|---|---|",
        f"| **Ground truth: no risk** | {metrics['tn']:,} (TN) | {metrics['fp']:,} (FP) |",
        f"| **Ground truth: risk** | {metrics['fn']:,} (FN) | {metrics['tp']:,} (TP) |",
        "",
        "## Risk signals by domain",
        "",
        "| Domain | Indicator-weeks | Ground-truth risk weeks | Detected risk weeks | Domain accuracy |",
        "|---|---|---|---|---|",
    ]
    for row in domain_summary.itertuples():
        lines.append(f"| {row.domain} | {row.n_indicator_weeks:,} | {row.ground_truth_risk_weeks:,} | "
                     f"{row.detected_risk_weeks:,} | {row.accuracy_pct}% |")
    lines += [
        "",
        "## Risk signals by trial",
        "",
        "| Trial | Indicators | Ground-truth risk weeks | Detected risk weeks | Trial accuracy |",
        "|---|---|---|---|---|",
    ]
    for row in trial_summary.itertuples():
        lines.append(f"| {row.trial_id} | {row.n_indicators} | {row.ground_truth_risk_weeks:,} | "
                     f"{row.detected_risk_weeks:,} | {row.accuracy_pct}% |")
    lines += [
        "",
        "## Current escalation summary (week 12, most recent monitoring period)",
        "",
        f"**{n_flagged} of {n_total} indicators** are currently flagged by the RBQM engine for "
        f"risk-review-committee escalation as of the most recent weekly snapshot.",
        "",
        "By domain: " + ", ".join(f"{d} = {int(c)}" for d, c in esc_by_domain.items()),
        "",
        "By trial: " + ", ".join(f"{t} = {int(c)}" for t, c in esc_by_trial.items()),
        "",
        "## Figures",
        "",
        "- `output/figures/confusion_matrix.png`",
        "- `output/figures/risk_signals_by_domain.png`",
        "- `output/figures/risk_signals_by_trial_heatmap.png`",
    ]
    with open(os.path.join(OUTPUT_DIR, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load()
    n_trials = df["trial_id"].nunique()
    n_indicators = df["indicator_id"].nunique()
    n_domains = df["domain"].nunique()

    metrics = overall_detection_metrics(df)
    domain_summary = risk_breakdown_by_domain(df)
    trial_summary = risk_breakdown_by_trial(df)
    heatmap_data = trial_domain_heatmap_data(df)
    n_flagged, n_total, esc_by_domain, esc_by_trial = escalation_summary(df)

    make_figures(metrics, domain_summary, heatmap_data)
    write_report(metrics, domain_summary, trial_summary, n_flagged, n_total, esc_by_domain, esc_by_trial,
                n_trials, n_indicators, n_domains)

    print("=" * 60)
    print("RBQM ANALYTICS FRAMEWORK ANALYSIS")
    print("=" * 60)
    print(f"Trials: {n_trials} | Indicators: {n_indicators} | Domains: {n_domains}")
    print(f"Indicator-weeks analyzed: {metrics['n']:,}")
    print(f"Detection accuracy: {metrics['accuracy']:.2f}%")
    print(f"Precision: {metrics['precision']:.2f}% | Recall: {metrics['recall']:.2f}% | "
         f"F1: {metrics['f1']:.2f}%")
    print(f"Confusion matrix: TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']} TN={metrics['tn']}")
    print(f"Currently escalated (week 12): {n_flagged} of {n_total} indicators")
    print("\nWrote output/report.md, output/figures/*.png")


if __name__ == "__main__":
    main()
