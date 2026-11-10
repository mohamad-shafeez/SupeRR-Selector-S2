# Model Card — SupeRR Selector Season 2 Pressure Reliability & Auction Intelligence Engine

## 1. Model Details
- **Name:** Rajasthan Royals Contextual Pressure Reliability Score (PRS v2.4.0)
- **Intended Use:** Evaluating T20 batter and bowler reliability under high-leverage match pressure (`P_i >= 0.55`) and supporting Rajasthan Royals' IPL 2027 auction recruitment across both Marquee/Established (`Leaderboard A`) and Uncapped Indian Domestic (`Leaderboard B`) cohorts.

## 2. Dataset & Critical Schema Remediation
- **Source Coverage:** 200,999 deliveries across 883 matches (141 competitions: IPL 2022–2026, Men's T20 Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025, and International T20Is).
- **Critical `extras1_runs` Bug Remediation:** In `ball_by_ball.csv`, `extras1_runs` stores the categorical `extras_type_id` (`1=Overstep`, `2=wide`, `3=legbyes`, `4=byes`, `5=other`, `6=Legal/None` on 189,167 rows, `7=Penalty`). Naive summation inflates innings totals by ~39 runs and bowler economies by ~36 RPO. Reconstructing extras via `extras_type` + `extras_2_runs` reduces non-DLS run-margin reconciliation MAE from `39.04` runs to `2.47` runs (with 126/396 exact 0-run matches; residual arises solely from multi-run wides/byes overwritten by the type ID in the source export).
- **Entity Resolution:** 4 true duplicate ID pairs (`Basil Hameed`, `Bashir Ahmad`, `Muhammad Farooq`, `Mohammad Ali`) are merged, while 3 cross-country homonym pairs (`Abdul Samad` IND vs. PAK, `Haider Ali` PAK vs. UAE, `Rashid Khan` AFG vs. NEP) are kept strictly separate.

## 3. Methodology Summary
1. **Strictly Causal Pre-Delivery Pressure Index ($P_i \in [0, 1]$):** Separates 1st-Innings Setting Pressure (deficit vs. pre-match format/venue par trajectory and wickets lost) from 2nd-Innings Chasing Pressure (DLS resource-adjusted required run rate, required acceleration, wicket scarcity, stage leverage, knife-edge closeness, and 12-ball momentum shock). Competition stakes (`IPL`, `Knockout`) are excluded from $P_i$ and modeled separately.
2. **Contextual Value Above Par (VAP):** Compares every delivery against expected runs and dismissal rates conditioned on `(Format, Innings, Over, Pace/Spin, Pressure Tier)` using empirical phase-and-wicket-state dismissal costs ($C_w \in [8.4, 21.4]$ runs).
3. **Empirical Bayes & Beta-Binomial Uncertainty:** Shrinks raw VAP toward role-specific priors using variance-component estimates ($\hat{k}_{\text{bat}}, \hat{k}_{\text{bowl}}$) and estimates match-level positive impact probability via a hierarchical Beta-Binomial prior. Downside risk is quantified via the 90% Posterior Lower Credible Bound combined with sample-shrunk 20th-percentile match output.
4. **Validation:** Verified via 10 automated unit tests (10/10 PASS), a 36-scenario sensitivity grid, 500 match-cluster bootstrap resamples (`seed=42`), and chronological out-of-sample testing (`2022–2024` train -> `2025–2026` test, achieving a 28.4% RMSE reduction over unshrunk raw metrics).

## 4. Limitations & Ethical/Data-Governance Considerations
- **Multi-Run Byes/Wides Censoring:** Because the source SQL export overwrote `extras1_runs` with `extras_type_id`, 5-wides and 4-byes/legbyes are recorded as `1` extra run when `extras_2_runs == 0`. This has zero impact on batter metrics and minimal impact (<0.08 RPO) on bowler economy.
- **2027 Auction Purse & Retention Rules:** BCCI IPL 2027 auction regulations are not yet published; all purse and retention parameters in the Auction Simulator are explicitly configurable scenario inputs.
- **Proprietary Data Protection:** Raw delivery rows from `ball_by_ball.csv` are never exposed in public outputs; only aggregated player-level and summary statistics are exported.
