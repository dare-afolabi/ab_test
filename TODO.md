# A/B Test Framework: Planned Fixes and Enhancements

---

## 0. Proposed final structure

```bash
ab_test/
│
├── pyproject.toml                  # Poetry/PEP-621: dependencies, scripts, entry-points
├── README.md                       # Project overview, installation, quick-start
├── CHANGELOG.md                    # Semantic versioning release notes
├── LICENSE                         # MIT (or Apache-2.0)
│
├── src/
│   └── ab_test/
│       │
│       ├── __init__.py             # Version, public API exports
│       ├── cli.py                  # Typer CLI entrypoint (ab-test command)
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   ├── settings.py         # Pydantic Settings: env, CLI, YAML/JSON config
│       │   └── schemas.py          # Pydantic models: ExperimentConfig, DataSource, RunSpec
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── session.py          # Context: seed, cache, temp dir, parallelism
│       │   └── registry.py         # Unified registry interface (SQLite + MLflow/DVC)
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── io.py               # Unified loader: CSV, Parquet, DB (SQLAlchemy), Delta Lake
│       │   ├── sampler.py          # Random, stratified (baseline quartiles), balanced
│       │   ├── validator.py        # Schema (Pandera/Great Expectations), SRM, covariate balance
│       │   └── transform.py        # Winsorize, CUPED, log, standardization, imputation
│       │
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── registry.py         # Dynamic KPI registry (name → callable)
│       │   ├── primary.py          # Mean, sum, ratio, conversion
│       │   ├── robust.py           # Trimmed mean, Huber, Winsorized variance, MAD
│       │   ├── composite.py        # Weighted indices, guardrail bundles
│       │   └── monitoring.py       # PSI, KS, CVM, Earth Mover’s Distance for drift
│       │
│       ├── stats/
│       │   ├── __init__.py
│       │   ├── inference.py        # Welch t-test, bootstrap (BCa), Wilcoxon, permutation
│       │   ├── adjustments.py      # FDR (BH), FWER (Bonferroni, Holm), closed testing
│       │   ├── power.py            # Analytical (normal), simulation-based, sequential MDE
│       │   ├── sequential.py       # Alpha-spending (OBF, Pocock), Bayesian (beta-binomial)
│       │   ├── diagnostics.py      # QQ, Shapiro, Levene, Durbin-Watson, variance ratio
│       │   ├── heterogeneity.py    # Subgroup analysis, causal forest, meta-learners
│       │   └── modeling.py         # ANCOVA, OLS, GLS, mixed-effects (lme4-style), GAM
│       │
│       ├── viz/
│       │   ├── __init__.py
│       │   ├── distributions.py    # Histograms, KDE, ECDF, overlapping density
│       │   ├── effects.py          # Forest plots, Cohen’s d, lift, CI ribbons
│       │   ├── diagnostics.py      # Love plots, balance tables, residual vs fitted
│       │   └── reporting.py        # Auto-figure grid, caption generator
│       │
│       ├── report/
│       │   ├── __init__.py
│       │   ├── builder.py          # Assemble sections: exec summary, methods, results
│       │   ├── renderers.py        # Markdown → HTML (Mistune), PDF (WeasyPrint), JSON
│       │   └── templates/          # Jinja2: report.md.j2, executive_summary.html.j2
│       │
│       ├── workflows/
│       │   ├── __init__.py
│       │   ├── pipeline.py         # Prefect/Luigi-style DAG: load → validate → analyze → report → register
│       │   └── presets.py          # Pre-built: revenue_test, engagement_suite, multi_arm
│       │
│       ├── engine/
│       │   ├── __init__.py
│       │   └── parallel.py         # Backend abstraction: joblib → multiprocessing → Dask → Ray
│       │
│       ├── api/
│       │   ├── __init__.py
│       │   ├── server.py           # FastAPI app: /experiments, /run, /results/{id}
│       │   ├── dashboard.py        # Streamlit app: explorer, comparison, audit log
│       │   └── security.py         # JWT, API keys, role-based (analyst, admin)
│       │
│       └── utils/
│           ├── __init__.py
│           ├── logging.py          # Loguru: structured, rotating, JSON output
│           ├── functional.py       # Helpers: compose, partial, safe_divide
│           └── decorators.py       # @timer, @retry(3), @cache(lru), @validate_args
│
├── scripts/
│   ├── generate_synthetic.py       # CLI: --n_users, --effect_size, --heterogeneity
│   └── migrate_registry.py         # Alembic-style schema evolution
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # pytest fixtures: temp paths, mock DB, synthetic data
│   ├── unit/
│   │   ├── test_data/
│   │   ├── test_metrics/
│   │   ├── test_stats/
│   │   ├── test_viz/
│   │   └── test_report/
│   ├── integration/
│   │   ├── test_workflow.py
│   │   └── test_registry.py
│   └── e2e/
│       ├── test_cli.py
│       └── test_api.py
│
├── docs/
│   ├── index.md
│   ├── user_guide.md
│   ├── api_reference.md
│   ├── contributing.md
│   └── mkdocs.yml                  # Material theme, mkdocstrings-python, navigation
│
└── .github/
    └── workflows/
        ├── ci.yml                  # ruff, black, mypy, pytest (cov>95%), security scan
        └── release.yml             # Semantic release, PyPI publish, GitHub release
```

---

## 1. Project Setup

| TODO | Layout Support |
|------|----------------|
| pyproject.toml with deps, CLI, build | `pyproject.toml` + `[project.scripts]` |
| README.md, CHANGELOG.md, LICENSE | Root-level files |
| .github/workflows/ci.yml | CI with ruff, mypy, pytest, coverage |
| .github/workflows/release.yml | Semantic release + PyPI |
| Dockerfile | (Add to root) — not in layout, but recommended |
| scripts/generate_synthetic.py, migrate_registry.py | `scripts/` directory |
| Action | Add Dockerfile (multi-stage with Poetry) to root |

---

## 2. `src/ab_test/core/`

| File | Status |
|------|--------|
| session.py | Global seed, cache, temp dir, parallelism context |
| registry.py | Unified SQLite + MLflow/DVC interface |
| Audit logging | Built into registry.py + utils/logging.py |

---

## 3. `src/ab_test/config/`

| File | Status |
|------|--------|
| settings.py | Pydantic Settings → env, CLI, YAML/JSON |
| schemas.py | ExperimentConfig, DataSource, RunSpec, metric defs |

---

## 4. `src/ab_test/data/`

| File | Features |
|------|----------|
| io.py | CSV, Parquet, SQLAlchemy, Delta Lake |
| sampler.py | Random, stratified, cluster-level |
| validator.py | Pandera/GE, SRM, balance, randomization tests |
| transform.py | Winsorize, CUPED, log, Box-Cox, imputation, outlier flagging |

---

## 5. `src/ab_test/metrics/`

| File | Features |
|------|----------|
| registry.py | Dynamic KPI registration |
| primary.py | Mean, sum, ratio, conversion |
| robust.py | Trimmed mean, Huber, MAD |
| composite.py | Weighted indices, guardrail bundles |
| monitoring.py | KS, PSI, CVM, autocorrelation, carryover, attrition |

---

## 6. `src/ab_test/stats/`

| File | Features |
|------|----------|
| inference.py | Welch, MW, permutation, BCa bootstrap |
| adjustments.py | Bonferroni, Holm, BH, closed testing |
| power.py | Analytical + simulation, MDE |
| sequential.py | OBF, Pocock, Bayesian beta-binomial |
| diagnostics.py | Shapiro, Levene, QQ, Durbin-Watson, parallel trends |
| heterogeneity.py | Subgroups, causal forest, meta-learners |
| modeling.py | ANCOVA, GLS, mixed-effects, GAM, multi-arm/factorial |

---

## 7. `src/ab_test/viz/`

| File | Features |
|------|----------|
| distributions.py | Hist, KDE, ECDF |
| effects.py | Cohen’s d, Cliff’s delta, forest, lift, CI |
| diagnostics.py | Balance tables, love plots, residuals |
| reporting.py | Auto-layout, captions, summary tables |

---

## 8. `src/ab_test/report/`

| File | Features |
|------|----------|
| builder.py | Exec summary, methods, results |
| renderers.py | MD → HTML, PDF, JSON |
| templates/ | Jinja2: report.md.j2, summary.html.j2, supports interactive collapsible plots via HTML + Plotly |

---

## 9. `src/ab_test/workflows/`

| File | Features |
|------|----------|
| pipeline.py | DAG: load → validate → analyze → report → register |
| presets.py | Revenue, engagement, multi-arm templates |

---

## 10. `src/ab_test/engine/`

| File | Features |
|------|----------|
| parallel.py | joblib → mp → Dask → Ray (config-driven) |

---

## 11. `src/ab_test/api/`

| File | Features |
|------|----------|
| server.py | FastAPI: /run, /results/{id} |
| dashboard.py | Streamlit explorer |
| security.py | JWT, API keys, RBAC |

---

## 12. `src/ab_test/utils/`

| File | Features |
|------|----------|
| logging.py | JSON, rotating, structured |
| functional.py | compose, partial, safe_divide |
| decorators.py | @timer, @retry, @cache, @validate_args |

---

## 13. CLI

| File | Features |
|------|----------|
| cli.py | ab-test run, generate, serve |

---

## 14. Tests

| Level | Coverage |
|-------|----------|
| unit/ | All modules |
| integration/ | Workflow, registry |
| e2e/ | CLI + API |
| conftest.py | Temp dir, mock DB, synthetic data |

---

## 15. Documentation

| File | Features |
|------|----------|
| docs/*.md | Guides |
| mkdocs.yml | Material theme, mkdocstrings, auto-generate from docstrings |

---

## 16. Remaining Statistical/Methodological Fixes

| Feature | Implementation |
|---------|----------------|
| Auto normality/variance checks | diagnostics.py → report/builder.py |
| Multi-metric & segment tables | reporting.py + builder.py |
| Warnings banners | Jinja2 templates |
| Interactive HTML | Plotly + collapsible sections |
| Guardrail alerts | monitoring.py → report |
