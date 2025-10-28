# A/B Test Pipeline

This repository contains a self-contained pipeline to run A/B test simulations and analyses.

## Files

- `README.md` - A user manual explaining how to use everything.
- `generate_synthetic_data.py` - Generates synthetic user-level data (CSV) with `user_id`, `baseline`, `spend_amount`, and `group` columns.
- `ab_test.py` - Main analysis script (optional sampling, winsorization, Welch's t-test, ANCOVA, figures, Markdown report).
- `arguments_example.sh` - Example commands and a master command showing all CLI arguments.
- `requirements.txt` - Python packages required.

## Quick start

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows (PowerShell)
pip install -r requirements.txt
```

2. Generate synthetic data (default 2000 users):

```bash
python generate_synthetic_data.py --out synthetic_data.csv
```

3.	Run the analysis:

```bash
python ab_test.py --input synthetic_data.csv --output_dir results --metric spend_amount --baseline_col baseline
```

4.	Open the Markdown report:

```bash
less results/ab_test_report.md
# or preview in your editor to view images
```

## CLI overview (ab_test.py)

Key arguments (defaults shown):

```bash
--input <path or 'db'>       # (required) CSV file path or 'db' for database
--table_name transactions
--metric spend_amount
--baseline_col baseline
--user_col user_id
--date_col event_date
--anchor_date 2025-09-01
--period_length "1 month"
--num_periods 2
--group_col group            # optional: pre-assigned
--control_label control
--treatment_label treatment
--n_users None               # (optional sampling cap - default: use full population)
--sample_by_q                # sample by baseline quartiles - default: random sampling
--winsor_lower 0.01
--winsor_upper 0.01
--bins 30
--output_dir results
--random_seed 42
-v                           # increase verbosity (-v, -vv)
```

Run python `ab_test.py --help` to see full usage.

## DB usage

If you pass `--input db` the script will attempt to read from your Postgres database. The script expects environment variables:
- `DB_USER`
- `DB_PASS`
- `DB_HOST`
- `DB_NAME`
- (optional) `DB_PORT` (defaults to 5432)

Ensure your table has the necessary columns (or pass the right `--metric`, `--user_col`, `--date_col`, and `--group_col`).

## Notes & Best practices

- By default the script uses the full population. Use `--n_users` to sample a smaller random subset for exploratory work.
- If you want a PDF report, convert the generated Markdown with Pandoc:

```bash
pandoc results/ab_test_report.md -o results/ab_test_report.pdf
```

## Additional Recommendations for Use

- **Testing**: Users should test the pipeline with the provided `generate_synthetic_data.py` to ensure compatibility with their environment before using real data.
- **Environment Setup**: Follow the **Quick start** instructions to set up a virtual environment and install dependencies to avoid conflicts.
- **Database Configuration**: If using database input, ensure environment variables (`DB_USER`, `DB_PASS`, `DB_HOST`, `DB_NAME`, and optionally `DB_PORT`) are set correctly.
- **Report Conversion**: For PDF output, users will need Pandoc installed, as noted in **Notes & Best practices**.
- For reproducible runs, set `--random_seed`.

## Example workflow

```bash
python generate_synthetic_data.py --n_users 5000 --out data/synth_5k.csv
python ab_test.py --input data/synth_5k.csv --n_users 2000 --group_col group --metric spend_amount --baseline_col baseline --output_dir results/myrun -v
```

## Contact / Next steps

If you want additional features like additional metrics in the report (e.g., medians, MAD, skewness), a multi-metric batch runner that loops over several metrics and aggregates results, or CI tests or coverage scripts, open an issue, contact the author via [dare.afolabi@outlook.com](mailto:dare.afolabi@outlook.com), or modify the scripts to suit your environment.
