#!/usr/bin/env python3
"""
generate_synthetic_data.py

Generates synthetic user-level data for the ab_test pipeline.
Columns: user_id, baseline, spend_amount, cohort, group.

Usage:
    python generate_synthetic_data.py --n_users 2000 --effect_size 0.08 --out synthetic_data.csv
"""

import argparse
import numpy as np
import pandas as pd
import os
from scipy import stats

parser = argparse.ArgumentParser(description="Generate synthetic A/B test data")
parser.add_argument("--n_users", type=int, default=2000, help="Number of users to generate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--baseline_mean", type=float, default=100.0, help="Mean baseline")
parser.add_argument("--baseline_sd", type=float, default=20.0, help="SD baseline")
parser.add_argument("--effect_size", type=float, default=0.08, help="Relative effect size for treatment")
parser.add_argument("--outcome_noise_sd", type=float, default=10.0, help="Noise SD")
parser.add_argument("--cohort_prop", type=float, default=0.5, help="Fraction in cohort A")
parser.add_argument("--out", type=str, default="synthetic_data.csv", help="Output CSV path")
parser.add_argument("--n_strata", type=int, default=4, help="Number of strata for baseline")

args = parser.parse_args()
np.random.seed(args.seed)
n = args.n_users

# --- baseline ---
baseline = np.random.normal(loc=args.baseline_mean, scale=args.baseline_sd, size=n)
baseline = np.clip(baseline, 0, None)

# --- cohort / segment ---
cohort = np.random.choice(["A", "B"], size=n, p=[args.cohort_prop, 1 - args.cohort_prop])

# --- stratified assignment ---
data = pd.DataFrame({
    "user_id": [f"user_{i+1}" for i in range(n)],
    "baseline": baseline,
    "cohort": cohort
})

data["strata"] = pd.qcut(data["baseline"], q=args.n_strata, labels=False)
data["group"] = np.nan

# Ensure exact balance overall
total_treat = int(n * args.cohort_prop)
treat_assigned = 0
control_assigned = 0

for s in sorted(data["strata"].unique()):
    idx = data[data["strata"] == s].index
    n_treat_stratum = int(np.floor(len(idx) * args.cohort_prop))
    shuffled_idx = np.random.permutation(idx)

    data.loc[shuffled_idx[:n_treat_stratum], "group"] = "treatment"
    data.loc[shuffled_idx[n_treat_stratum:], "group"] = "control"

    treat_assigned += n_treat_stratum
    control_assigned += len(idx) - n_treat_stratum

# Correct for rounding drift
diff = total_treat - treat_assigned
if diff != 0:
    available_idx = data[data["group"] == ("control" if diff > 0 else "treatment")].sample(abs(diff), random_state=args.seed).index
    data.loc[available_idx, "group"] = "treatment" if diff > 0 else "control"

# Confirm balance
print(data["group"].value_counts())
# --- simulate spend_amount (metric) ---
noise = np.random.normal(0, args.outcome_noise_sd, n)
spend_amount = data["baseline"] + noise
spend_amount[data["group"] == "treatment"] *= 1 + args.effect_size
spend_amount = np.clip(spend_amount, 0, None)
data["spend_amount"] = spend_amount

# --- save ---
out_dir = os.path.dirname(args.out) or "."
os.makedirs(out_dir, exist_ok=True)
data.drop(columns="strata").to_csv(args.out, index=False)

print(f"Synthetic data saved to: {args.out}")
print(f"Rows: {len(data)}, columns: {list(data.columns)}")