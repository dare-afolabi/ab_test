#!/usr/bin/env python3
"""
A/B Test Pipeline
=================
Processes A/B test data from a PostgreSQL database or CSV file, performing statistical analysis
(Welch's t-test and ANCOVA) and generating visualizations and a markdown report.

Expected Input:
- Database: Table with columns for user_id, metric, date, and optional group (via --user_col, --metric, --date_col, --group_col).
- CSV: Columns user_id, baseline, outcome, and optional group (mapped via --user_col, --metric, --group_col).
Assumes exactly two groups (control and treatment) and numeric baseline/outcome values.

Outputs:
- output/ab_test_data.csv: Processed data with group_normalized, baseline_win, outcome_win, and diff columns.
- output/*.png: Histogram and boxplot figures for baseline and outcome distributions.
- output/ab_test_report.md: Markdown report with summaries, statistical results, and figures.
"""

import os
import sys
import argparse
from datetime import datetime, date, timedelta
import logging
import re
from sqlalchemy import text, create_engine
from sqlalchemy.sql import quoted_name
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mstats
from scipy.stats import t as student_t
import statsmodels.formula.api as smf

def main():
    # ---------------------------
    # 0. Defaults / setup
    # ---------------------------
    DEFAULT_OUTPUT_DIR = "output"

    default_anchor_date = os.getenv("ANCHOR_DATE")
    if default_anchor_date:
        try:
            default_anchor_date = datetime.strptime(default_anchor_date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Environment variable ANCHOR_DATE must be in YYYY-MM-DD format")
    else:
        default_anchor_date = (date.today().replace(day=1) - timedelta(days=1)).replace(day=1)
    if default_anchor_date > date.today():
        raise ValueError(f"Default anchor date {default_anchor_date} is in the future")

    # ---------------------------
    # 1. CLI arguments
    # ---------------------------
    parser = argparse.ArgumentParser(description="A/B test pipeline (DB or CSV input)")

    # Data source
    parser.add_argument("--input", type=str, required=True, help="CSV file path or 'db' for database (reads environment DB_* variables)")
    parser.add_argument("--table_name", type=str, default="transactions", help="Database table name")
    parser.add_argument("--metric", type=str, default="spend_amount", help="Metric column name (for DB usage)")
    parser.add_argument("--baseline_col", type=str, default="baseline", help="Column name for baseline metric (CSV input)")
    parser.add_argument("--user_col", type=str, default="user_id", help="User ID column name (for DB usage)")
    parser.add_argument("--date_col", type=str, default="event_date", help="Date column name (for DB usage)")
    parser.add_argument("--anchor_date", type=str, default=default_anchor_date.strftime("%Y-%m-%d"), help="Anchor date for periods (YYYY-MM-DD)")
    parser.add_argument("--period_length", type=str, default="1 month", help="Period length for aggregation (SQL INTERVAL string)")
    parser.add_argument("--num_periods", type=int, default=2, help="Number of periods (baseline + outcome)")
    parser.add_argument("--group_col", type=str, default=None, help="Column name that contains pre-assigned group labels (e.g. 'group' or 'assignment'). If provided, the script will use it and NOT assign groups.")
    parser.add_argument("--control_label", type=str, default="control", help="Label to use for the control group (output normalization)")
    parser.add_argument("--treatment_label", type=str, default="treatment", help="Label to use for the treatment group (output normalization)")

    # Parameters
    parser.add_argument("--n_users", type=int, default=None, help="Optional: cap total users (random sample) - if omitted use full population")
    parser.add_argument("--sample_by_q", action="store_true", help="If set, sample n_users evenly across baseline quartiles. Only used when --n_users < dataset size.")
    parser.add_argument("--winsor_lower", type=float, default=0.01, help="Winsor lower limit (0-0.5)")
    parser.add_argument("--winsor_upper", type=float, default=0.01, help="Winsor upper limit (0-0.5)")
    parser.add_argument("--bins", type=int, default=30, help="Number of bins for histograms")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory to save plots and report")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity (repeatable)")

    args = parser.parse_args()

    # Validate arguments
    try:
        datetime.strptime(args.anchor_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("--anchor_date must be in YYYY-MM-DD format")

    def validate_period_length(period: str) -> str:
        """Validate PostgreSQL INTERVAL format."""
        valid_pattern = r"^\d+\s+(second|minute|hour|day|week|month|year)s?$"
        if not re.match(valid_pattern, period.strip()):
            raise ValueError(f"Invalid --period_length: '{period}'. Must be like '1 day', '2 months', etc.")
        return period

    args.period_length = validate_period_length(args.period_length)

    if not (0 <= args.winsor_lower <= 0.5 and 0 <= args.winsor_upper <= 0.5):
        raise ValueError("--winsor_lower and --winsor_upper must be between 0 and 0.5")

    if args.n_users is not None and args.n_users <= 0:
        raise ValueError("--n_users must be a positive integer")

    if args.num_periods < 2:
        raise ValueError("--num_periods must be at least 2 (baseline and outcome periods)")

    # ---------------------------
    # 2. Logging
    # ---------------------------
    log_level = logging.WARNING
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose == 1:
        log_level = logging.INFO

    logger = logging.getLogger("ab_test")
    logger.setLevel(log_level)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    logger.info("Starting A/B test pipeline")

    os.makedirs(args.output_dir, exist_ok=True)
    if not os.access(args.output_dir, os.W_OK):
        raise PermissionError(f"Output directory {args.output_dir} is not writable")

    # ---------------------------
    # 3. Load data (DB or CSV)
    # ---------------------------
    def read_from_db(args):
        """Reads paired baseline and outcome values per user from a PostgreSQL database.
        If args.group_col is provided and exists in the table, it will be included as group_col.
        """
        user = os.getenv("DB_USER")
        password = os.getenv("DB_PASS")
        host = os.getenv("DB_HOST")
        port_str = os.getenv("DB_PORT", "5432")
        try:
            port = int(port_str)
            if not (1 <= port <= 65535):
                raise ValueError(f"DB_PORT must be between 1 and 65535, got '{port}'")
        except ValueError:
            raise ValueError(f"DB_PORT must be an integer between 1 and 65535, got '{port_str}'")
        dbname = os.getenv("DB_NAME")
        missing = [k for k, v in (("DB_USER", user), ("DB_PASS", password), ("DB_HOST", host), ("DB_NAME", dbname)) if v is None]
        if missing:
            raise RuntimeError(f"Missing DB environment variables: {', '.join(missing)}")

        table_name_q = quoted_name(args.table_name, quote=True)
        metric_q = quoted_name(args.metric, quote=True)
        user_col_q = quoted_name(args.user_col, quote=True)
        date_col_q = quoted_name(args.date_col, quote=True)
        group_col_q = quoted_name(args.group_col, quote=True) if args.group_col else None

        engine_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        engine = create_engine(engine_url, connect_args={"connect_timeout": 10})
        logger.info(f"Connected to database {dbname} at {host}:{port}")

        group_select = f", MAX(t.{group_col_q}) AS group_col" if group_col_q else ", NULL::text AS group_col"

        query = text(f"""
        WITH params AS (
            SELECT
                CAST(:anchor_date AS DATE) AS anchor_date,
                CAST(:period_length AS INTERVAL) AS period_length,
                CAST(:num_periods AS INT) AS num_periods
        ),
        periods AS (
            SELECT generate_series(
                (SELECT anchor_date - period_length * num_periods FROM params),
                (SELECT anchor_date FROM params),
                (SELECT period_length FROM params)
            ) AS period_start
        ),
        aggregated AS (
            SELECT
                t.{user_col_q} AS user_id,
                p.period_start,
                SUM(t.{metric_q}) AS metric_value
                {group_select}
            FROM {table_name_q} t
            JOIN periods p
                ON t.{date_col_q} >= p.period_start
            AND t.{date_col_q} < p.period_start + (SELECT period_length FROM params)
            GROUP BY t.{user_col_q}, p.period_start
        ),
        paired AS (
            SELECT
                user_id,
                MAX(CASE WHEN period_start = (SELECT MIN(period_start) FROM periods) THEN metric_value END) AS baseline,
                MAX(CASE WHEN period_start = (SELECT MAX(period_start) FROM periods) THEN metric_value END) AS outcome,
                MAX(group_col) AS group_col
            FROM aggregated
            GROUP BY user_id
        )
        SELECT user_id, baseline, outcome, group_col
        FROM paired
        WHERE baseline IS NOT NULL AND outcome IS NOT NULL;
        """)

        params = {
            "anchor_date": args.anchor_date,
            "period_length": args.period_length,
            "num_periods": args.num_periods
        }

        logger.info("Executing query to extract paired baseline/outcome per user...")
        df = pd.read_sql(query, engine, params=params)
        if args.group_col:
            unique_groups = df["group_col"].dropna().unique()
            if len(unique_groups) < 2:
                raise RuntimeError(f"Database group_col '{args.group_col}' has {len(unique_groups)} unique non-null values; expected at least 2")
        logger.info(f"Retrieved {len(df):,} user records.")
        return df

    def read_from_csv(path, args):
        """Read CSV and map columns to expected names."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV input file not found: {path}")
        df = pd.read_csv(path)
        rename_dict = {}
        if args.user_col and args.user_col in df.columns:
            rename_dict[args.user_col] = "user_id"
        if args.metric and args.metric in df.columns:
            rename_dict[args.metric] = "outcome"
        if args.baseline_col and args.baseline_col in df.columns:
            rename_dict[args.baseline_col] = "baseline"
        if rename_dict:
            df = df.rename(columns=rename_dict)
        required = {"user_id", "baseline", "outcome"}
        if not required.issubset(set(df.columns)):
            missing = required - set(df.columns)
            raise RuntimeError(f"CSV input must contain columns: {', '.join(required)}. Missing: {', '.join(missing)}")
        if args.group_col and args.group_col not in df.columns:
            raise RuntimeError(f"CSV input does not contain specified --group_col: {args.group_col}")
        if args.group_col and args.group_col in df.columns:
            unique_groups = df[args.group_col].dropna().unique()
            if len(unique_groups) < 2:
                raise RuntimeError(f"CSV group_col '{args.group_col}' has {len(unique_groups)} unique non-null values; expected at least 2")
        return df

    try:
        if args.input.lower() == "db":
            data = read_from_db(args)
        else:
            data = read_from_csv(args.input, args)
    except Exception as e:
        logger.exception("Failed to load data")
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)

    # ---------------------------
    # 4. Optional sampling & data checks
    # ---------------------------
    np.random.seed(args.random_seed)

    # Ensure numeric columns
    for col in ("baseline", "outcome"):
        if not pd.api.types.is_numeric_dtype(data[col]):
            logger.info(f"Coercing column {col} to numeric (non-convertible values -> NaN)")
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Drop rows with NA in baseline or outcome
    initial_n = len(data)
    data = data.dropna(subset=["baseline", "outcome"])
    dropped_n = initial_n - len(data)
    if dropped_n > 0:
        logger.warning(f"Dropped {dropped_n} rows ({dropped_n/initial_n*100:.1f}%) with NA baseline/outcome")

    # Optional sampling
    if args.n_users is not None and len(data) > args.n_users:
        logger.info(f"Sampling {args.n_users} users out of {len(data)}")
        
        if args.sample_by_q:
            # Stratified sampling by baseline quartiles
            n_quantiles = 4
            data["__quantile__"] = pd.qcut(data["baseline"], q=n_quantiles, labels=False)
            sampled_idx = []
            
            for q in range(n_quantiles):
                for g in data["group"].unique():
                    idx = data[(data["__quantile__"] == q) & (data["group"] == g)].index
                    n_sample = round(len(idx) / len(data) * args.n_users)
                    sampled_idx.extend(np.random.choice(idx, size=n_sample, replace=False))

            data = data.loc[sampled_idx].reset_index(drop=True)
            data = data.drop(columns="__quantile__")
            logger.info("Stratified sampling by baseline completed")
        else:
            # Stratified random sampling: n_users/2 from each group
            n_per_group = args.n_users // 2
            data = (
                data.groupby("group", group_keys=False)
                .sample(n=n_per_group, random_state=args.random_seed)
                .reset_index(drop=True)
            )
            logger.info(f"Stratified random sampling completed: {n_per_group} per group")
            
    # ---------------------------
    # 5. Group handling
    # ---------------------------

    def validate_label(label: str) -> str:
        """Validate control/treatment labels for statsmodels formula."""
        if not re.match(r"^[a-zA-Z0-9_]+$", label):
            raise ValueError(f"Label '{label}' contains invalid characters; use alphanumeric and underscores only")
        return label

    args.control_label = validate_label(args.control_label)
    args.treatment_label = validate_label(args.treatment_label)

    if args.group_col and args.group_col in data.columns:
        data["group_raw"] = data[args.group_col].astype(str)
        unique_vals = sorted(data["group_raw"].dropna().unique().tolist())
        logger.info(f"Using user-supplied group_col '{args.group_col}' with values: {unique_vals}")
        if set(unique_vals) >= {args.control_label, args.treatment_label}:
            def _map_func(x):
                if pd.isna(x):
                    return np.nan
                s = str(x)
                if s == args.control_label:
                    return args.control_label
                if s == args.treatment_label:
                    return args.treatment_label
                return s
            data["group_normalized"] = data["group_raw"].map(_map_func)
        else:
            if len(unique_vals) == 2:
                map_dict = {unique_vals[0]: args.control_label, unique_vals[1]: args.treatment_label}
                logger.info(f"Mapping user group values {map_dict} -> ({args.control_label}, {args.treatment_label})")
                data["group_normalized"] = data["group_raw"].map(map_dict)
            else:
                raise RuntimeError(
                    f"--group_col provided but contains {len(unique_vals)} unique non-null values; expected exactly 2 (control/treatment)."
                )
        present = data["group_normalized"].dropna().unique().tolist()
        if len(present) != 2:
            raise RuntimeError(f"After normalizing --group_col, groups present: {present}. Need exactly two groups.")
        logger.info("Group column normalized and validated. No assignment performed.")
    elif "group" in data.columns:
        data["group_raw"] = data["group"].astype(str)
        unique_vals = sorted(data["group_raw"].dropna().unique().tolist())
        logger.info(f"Using existing 'group' column with values: {unique_vals}")
        if set(unique_vals) >= {args.control_label, args.treatment_label}:
            data["group_normalized"] = data["group_raw"]
        else:
            if len(unique_vals) == 2:
                map_dict = {unique_vals[0]: args.control_label, unique_vals[1]: args.treatment_label}
                logger.info(f"Mapping existing 'group' values {map_dict} -> ({args.control_label}, {args.treatment_label})")
                data["group_normalized"] = data["group_raw"].map(map_dict)
            else:
                raise RuntimeError(f"'group' column exists but has {len(unique_vals)} unique non-null values; expected exactly 2.")
    else:
        raise RuntimeError("No group information found. Provide --group_col or include a 'group' column in the input data. This script does not assign groups.")

    initial_n = len(data)
    data = data.dropna(subset=["group_normalized"])
    dropped_n = initial_n - len(data)
    if dropped_n > 0:
        logger.warning(f"Dropped {dropped_n} rows ({dropped_n/initial_n*100:.1f}%) due to missing or unmapped group values")

    # ---------------------------
    # 6. Winsorize and derived columns
    # ---------------------------
    # Winsorize both baseline and outcome to handle outliers consistently for ANCOVA
    data["baseline_win"] = mstats.winsorize(data["baseline"], limits=(args.winsor_lower, args.winsor_upper)).astype(float)
    data["outcome_win"] = mstats.winsorize(data["outcome"], limits=(args.winsor_lower, args.winsor_upper)).astype(float)
    data["diff"] = data["outcome_win"] - data["baseline_win"]

    # ---------------------------
    # 7. Helper functions (summaries, plots, tests)
    # ---------------------------
    def summarize(df, col):
        s = df.groupby("group_normalized")[col].agg(
            n="count", mean="mean", std="std", min="min", median="median", max="max"
        ).round(2).reset_index()
        return s

    def plot_distribution(metric, title, output_dir):
        plt.figure(figsize=(7, 4.5))
        sns.histplot(data=data, x=metric, hue="group_normalized", bins=args.bins, kde=True, stat="count")
        plt.title(title)
        plt.xlabel(metric)
        plt.ylabel("Count")
        fname_hist = f"{metric}_hist.png"
        plt.savefig(os.path.join(output_dir, fname_hist), bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.boxplot(x="group_normalized", y=metric, data=data)
        sns.pointplot(x="group_normalized", y=metric, data=data, estimator=np.mean, linestyle="none", errorbar=None, color="red")
        plt.title(title + " (Boxplot)")
        fname_box = f"{metric}_box.png"
        plt.savefig(os.path.join(output_dir, fname_box), bbox_inches="tight")
        plt.close()

        return fname_hist, fname_box

    def significance(p_value):
        if p_value < 0.001: return "< 0.001"
        elif p_value < 0.05: return f"= {p_value:.4f}"
        else: return f"= {p_value:.4f} (n.s.)"

    def run_ancova(df, outcome_var):
        args.control_label = validate_label(args.control_label)
        args.treatment_label = validate_label(args.treatment_label)
        model = smf.ols(f"{outcome_var} ~ C(group_normalized, Treatment(reference='{args.control_label}')) + baseline_win", data=df).fit(cov_type="HC3")
        logger.debug(f"ANCOVA model parameters: {model.params}")
        treatment_param = f"C(group_normalized, Treatment(reference='{args.control_label}'))[T.{args.treatment_label}]"
        if treatment_param not in model.params:
            raise RuntimeError(f"Treatment parameter '{treatment_param}' not found in model")
        coef = model.params[treatment_param]
        se = model.bse[treatment_param]
        tval = model.tvalues[treatment_param]
        pval = model.pvalues[treatment_param]
        try:
            df_resid = model.df_resid
            t_crit = student_t.ppf(0.975, df_resid)
            ci_lower = coef - t_crit * se
            ci_upper = coef + t_crit * se
        except Exception:
            logger.warning("Falling back to z-score for ANCOVA CI due to error")
            ci_lower, ci_upper = (coef - 1.96 * se, coef + 1.96 * se)
        return coef, se, tval, pval, model.rsquared_adj, (ci_lower, ci_upper), model

    # ---------------------------
    # 8. Generate figures
    # ---------------------------
    sns.set_style("ticks")
    hist_base, box_base = plot_distribution("baseline_win", "Baseline Distribution (winsorized)", args.output_dir)
    hist_out, box_out = plot_distribution("outcome_win", "Outcome Distribution (winsorized)", args.output_dir)

    # ---------------------------
    # 9. T-test & effect sizes (unified)
    # ---------------------------
    a = data.loc[data["group_normalized"] == args.control_label, "outcome_win"]
    b = data.loc[data["group_normalized"] == args.treatment_label, "outcome_win"]
    n_control, n_treat = a.count(), b.count()
    mean_control, mean_treat = a.mean(), b.mean()
    std_control, std_treat = a.std(ddof=1), b.std(ddof=1)
    diff_val = mean_treat - mean_control
    # Optional nonparametric fallback for very small samples
    use_mannwhitney = False
    if n_control < 30 or n_treat < 30:
        logger.warning(f"Small sample size detected (control: {n_control}, treatment: {n_treat}). "
                        "T-test assumes normality, which may not hold.")
        if min(n_control, n_treat) < 10:
            logger.warning("Using nonparametric Mann–Whitney U test instead of t-test due to small n (< 10 per group)")
            use_mannwhitney = True
    if use_mannwhitney:
        u_stat, p_mw = stats.mannwhitneyu(b, a, alternative='two-sided')
        test_used = "Mann–Whitney U"
    else:
        t_stat, p_ttest = stats.ttest_ind(b, a, equal_var=False)
        test_used = "Welch's t-test"

    if not use_mannwhitney:
        # Standard error and Welch–Satterthwaite df for CI
        se_diff = np.sqrt(std_control**2 / n_control + std_treat**2 / n_treat)
        df_welch = (std_control**2 / n_control + std_treat**2 / n_treat)**2 / (
            (std_control**2 / n_control)**2 / (n_control - 1) + (std_treat**2 / n_treat)**2 / (n_treat - 1)
        )
        # 95% CI
        t_crit = student_t.ppf(0.975, df_welch)
        ci_lower = diff_val - t_crit * se_diff
        ci_upper = diff_val + t_crit * se_diff
        # Cohen's d (pooled SD)
        cohen_d = diff_val / np.sqrt((std_control**2 + std_treat**2) / 2)
    else:
        ci_lower = ci_upper = np.nan
        cohen_d = np.nan

    # ---------------------------
    # 10. ANCOVA
    # ---------------------------
    try:
        coef_raw, se_raw, t_raw, p_raw, r2_raw, (anc_ci_l, anc_ci_u), model_obj = run_ancova(data, "outcome_win")
    except Exception as e:
        logger.exception("ANCOVA failed")
        coef_raw = se_raw = t_raw = p_raw = r2_raw = anc_ci_l = anc_ci_u = None
        model_obj = None

    if model_obj is not None:
        logger.debug(f"ANCOVA model summary:\n{model_obj.summary()}")
    # ---------------------------
    # 11. Console output + Markdown report
    # ---------------------------
    report_path = os.path.join(args.output_dir, "ab_test_report.md")

    # Prepare summaries
    baseline_summary = summarize(data, "baseline_win")
    outcome_summary = summarize(data, "outcome_win")

    def cohen_d_interpret(d):
        if abs(d) < 0.2: return "negligible"
        elif abs(d) < 0.5: return "small"
        elif abs(d) < 0.8: return "medium"
        else: return "large"

    cohen_d_desc = cohen_d_interpret(cohen_d) if not np.isnan(cohen_d) else "NA"

    # Prepare t-test or Mann–Whitney text
    # In section 11, replace ttest_text for markdown
    if use_mannwhitney:
        ttest_text = (
            f"- Control ({args.control_label}): n={n_control}, mean={mean_control:.2f}, sd={std_control:.2f}\n"
            f"- Treatment ({args.treatment_label}): n={n_treat}, mean={mean_treat:.2f}, sd={std_treat:.2f}\n"
            f"- {test_used}: U = {u_stat:.3f}, p {significance(p_mw)}\n"
            f"- Mean difference (treat - control) = {diff_val:.3f}\n"
            f"- 95% CI = [NA, NA]\n"
            f"- Cohen's d = NA\n"
            f"- Cohen's d interpretation: NA"
        )
    else:
        ttest_text = (
            f"- Control ({args.control_label}): n={n_control}, mean={mean_control:.2f}, sd={std_control:.2f}\n"
            f"- Treatment ({args.treatment_label}): n={n_treat}, mean={mean_treat:.2f}, sd={std_treat:.2f}\n"
            f"- {test_used}: t = {t_stat:.3f}, p {significance(p_ttest)}\n"
            f"- Mean difference (treat - control) = {diff_val:.3f}\n"
            f"- 95% CI = [{ci_lower:.3f}, {ci_upper:.3f}]\n"
            f"- Cohen's d = {cohen_d:.3f}\n"
            f"- Cohen's d interpretation: {cohen_d_desc}"
        )

    def fmt_num(x, ndigits=3):
        return f"{x:.{ndigits}f}" if x is not None else "NA"

    # Prepare ANCOVA text
    if coef_raw is not None:
        ancova_text = (
            f"Treatment effect = {fmt_num(coef_raw)} (robust SE = {fmt_num(se_raw)}, t = {fmt_num(t_raw)}, p {significance(p_raw)})\n"
            f"95% CI = [{fmt_num(anc_ci_l)}, {fmt_num(anc_ci_u)}], Adjusted R² = {fmt_num(r2_raw)}"
    )
    else:
        ancova_text = "ANCOVA failed. See logs for details."

    # Console output
    print(f"\n=== Analysis Context ===")
    print(f"Anchor date: {args.anchor_date}, Period length: {args.period_length}, Number of periods: {args.num_periods}")
    print("\n=== Baseline (winsorized) summaries by group ===")
    print(baseline_summary.to_string(index=False))
    print("\n=== Outcome (winsorized) summaries by group ===")
    print(outcome_summary.to_string(index=False))
    print(f"\n=== Unadjusted {test_used} t-test ===")
    print(ttest_text)
    print("\n=== ANCOVA (baseline-adjusted) ===")
    print(ancova_text)

    # Write Markdown report
    try:
        with open(report_path, "w") as f:
            f.write("# A/B Test Report\n\n")
            f.write(f"**Source:** `{args.input}`\n\n")
            f.write("## 1. Baseline summaries (winsorized)\n\n")
            f.write(baseline_summary.to_markdown(index=False))
            f.write("\n\n## 2. Outcome summaries (winsorized)\n\n")
            f.write(outcome_summary.to_markdown(index=False))
            f.write(f"\n\n## 3. Unadjusted {test_used} t-test\n\n")
            f.write(ttest_text + "\n\n")
            f.write("## 4. ANCOVA (baseline-adjusted)\n\n")
            f.write(ancova_text.replace("\n", "  \n") + "\n\n")
            if model_obj is not None:
                f.write("\n\n### ANCOVA model summary\n\n")
                f.write("```\n")
                f.write(model_obj.summary().as_text())
                f.write("\n```\n")
            f.write("## 5. Figures\n\n")
            f.write(f"![Baseline Histogram]({hist_base})  \n")
            f.write(f"![Baseline Boxplot]({box_base})  \n")
            f.write(f"![Outcome Histogram]({hist_out})  \n")
            f.write(f"![Outcome Boxplot]({box_out})  \n\n")
            f.write("## 6. Data\n\n")
            f.write(f"- Total rows processed: {len(data)}  \n")
            f.write(f"- Rows dropped due to NA baseline/outcome: {dropped_n}  \n")
    except IOError as e:
        logger.error(f"Failed to write report to {report_path}: {e}")
        raise

    # ---------------------------
    # 12. Save CSV
    # ---------------------------
    out_csv = os.path.join(args.output_dir, "ab_test_data.csv")
    try:
        data.to_csv(out_csv, index=False)
    except IOError as e:
        logger.error(f"Failed to write CSV to {out_csv}: {e}")
        raise

    print(f"\nReport and figures saved in: {args.output_dir}")
    print(f"- Markdown report: {report_path}")
    print(f"- Processed data CSV: {out_csv}")

if __name__ == "__main__":
    main()