# SupeRR Selector Season 2 — Rajasthan Royals IPL 2027 Analytics War Room

**Model Version:** `2.4.0-RR-WarRoom` | **Data Snapshot:** `2022–2026 Official Data Pack (200,999 deliveries across 883 matches)` | **Random Seed:** `42`

## 1. Architecture & Reproducibility
This repository separates the reproducible statistical computation engine (`analysis/`) and automated test suite (`tests/`) from the interactive presentation layer (`index.html`). Every player statistic, pressure classification, Bayesian shrinkage adjustment, bootstrap rank interval, and Rajasthan Royals squad gap metric displayed in the dashboard is generated directly from the official CSV data pack via:

```bash
python3 analysis/build_analysis.py
```

## 2. Directory Structure
- `analysis/config.json`: Authoritative model hyperparameters, thresholds, chronological split definitions, and random seed (`42`).
- `analysis/data_loader.py`: SHA-256 verification, duplicate player ID merging, cross-country homonym disambiguation (`Abdul Samad`, `Haider Ali`, `Rashid Khan`), and categorical `extras1_runs` reconstruction.
- `analysis/data_validation.py`: Automated 13-point Data Quality and Scorecard Reconciliation Engine.
- `analysis/pressure_engine.py`: Strictly causal pre-delivery Situational Pressure Index ($P_i \in [0, 1]$) with separated 1st-Innings (Setting) vs. 2nd-Innings (Chasing) formulations and Clutch Micro-Moment detection.
- `analysis/batting_metrics.py`: Empirical Run-Expectancy & Context-Dependent Wicket Cost ($C_w$) engine for Batting & Bowling Value Above Par (VAP).
- `analysis/bowling_metrics.py`: Dataset-derived competition eligibility, IPL vs. T20I provenance, and Uncapped Indian target pool resolution.
- `analysis/reliability_model.py`: Variance-components Empirical Bayes shrinkage ($\hat{k}$), Beta-Binomial match consistency, Bayesian Lower Credible Bound downside protection, Phase/Matchup splits, and Pressure Reliability Score (PRS).
- `analysis/stability_analysis.py`: 36-Scenario Sensitivity Grid ($3 \times 4 \times 3$) and 500-Iteration Match-Cluster Bootstrap resampler.
- `analysis/player_similarity.py`: Standardized feature-vector Cosine & Weighted Euclidean similarity engine (`MARQUEE PLAYER -> STATISTICAL ALTERNATIVE`).
- `analysis/auction_optimizer.py`: Rajasthan Royals squad gap quantification, Sawai Mansingh Stadium (`venue_id=6`) compatibility model, chronological out-of-sample validation (`2022–2024` train -> `2025` val -> `2026` test), 2027 projections, and constrained portfolio scenarios.
- `tests/`: Automated test suite (`test_extras.py`, `test_scoring.py`, `test_bowling.py`, `test_pressure.py`, `test_reliability.py`) verifying 10/10 invariants.
- `outputs/`: Generated CSV and JSON artifacts consumed by the frontend and technical report.
