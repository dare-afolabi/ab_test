#!/usr/bin/env bash
# arguments_example.sh
# Example invocations of ab_test.py

# Basic usage (CSV input)
python ab_test.py \
  --input synthetic_data.csv \
  --output_dir results

# Example: cap at 1000 users (random sample) with verbose output
python ab_test.py \
  --input synthetic_data.csv \
  --n_users 1000 \
  --output_dir results_sampled \
  -v

# Full master command with all arguments, explicit defaults
python ab_test.py \
  --input synthetic_data.csv \
  --table_name transactions \
  --metric spend_amount \
  --baseline_col baseline \
  --user_col user_id \
  --sample_by_q \
  --date_col event_date \
  --anchor_date 2025-09-01 \
  --period_length "1 month" \
  --num_periods 2 \
  --group_col group \
  --control_label control \
  --treatment_label treatment \
  --n_users None \
  --winsor_lower 0.01 \
  --winsor_upper 0.01 \
  --bins 30 \
  --output_dir results \
  --random_seed 42 \
  -v