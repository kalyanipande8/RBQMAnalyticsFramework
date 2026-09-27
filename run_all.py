"""
run_all.py

Runs the full RBQM analytics pipeline end-to-end:
  1. Simulate weekly KRI readings for 8 trials / 312 indicators / 3 domains
     -> data/raw/
  2. Standardize into one unified indicator-week table and apply the
     detection rule engine -> data/processed/
  3. Detection performance, risk breakdowns, escalation summary, figures
     -> output/

Run with:  python run_all.py
"""

import simulate_rbqm_data
import standardize
import rbqm_analysis


def main():
    print("--- Simulating RBQM monitoring data ---")
    simulate_rbqm_data.simulate()

    print("\n--- Standardizing + applying detection rule engine ---")
    standardize.main()

    print("\n--- RBQM analysis ---")
    rbqm_analysis.main()


if __name__ == "__main__":
    main()
