import json
import os
import sys
import numpy as np
import pandas as pd

ANALYSIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ANALYSIS_DIR)
TESTS_DIR = os.path.join(PROJECT_ROOT, "tests")
sys.path.insert(0, ANALYSIS_DIR)
sys.path.insert(0, TESTS_DIR)

from data_loader import load_and_clean_data
from data_validation import run_data_audit
from pressure_engine import compute_pressure_index, summarize_pressure_distributions
from batting_metrics import add_contextual_vap_columns
from bowling_metrics import build_player_eligibility_metadata
from reliability_model import compute_batter_reliability_table, compute_bowler_reliability_table
from stability_analysis import run_sensitivity_and_bootstrap
from player_similarity import build_similarity_engine
from auction_optimizer import analyze_rr_gaps_jaipur_and_projections

from test_extras import run_test_extras
from test_scoring import run_test_scoring
from test_bowling import run_test_bowling
from test_pressure import run_test_pressure
from test_reliability import run_test_reliability

def generate_automated_insights(bat_df: pd.DataFrame, bowl_df: pd.DataFrame, sim_data: dict, audit: dict) -> list[dict]:
    insights = []

    # 1. Highest role-adjusted pressure strike rate lift among established IPL batters
    bat_est = bat_df[(bat_df["has_ipl"]) & (bat_df["press_balls"] >= 100)].copy()
    top_lift_bat = bat_est.sort_values("raw_sr_delta", ascending=False).iloc[0]
    insights.append({
        "category": "Contextual Batting Lift",
        "headline": f"{top_lift_bat['striker_name']} accelerates by {top_lift_bat['raw_sr_delta']:+.1f} SR under pressure ({top_lift_bat['base_sr']:.1f} -> {top_lift_bat['raw_press_sr']:.1f} SR)",
        "evidence": f"Across {int(top_lift_bat['press_balls'])} pressure balls ({int(top_lift_bat['press_matches'])} matches), generates {top_lift_bat['posterior_vap']:+.1f} Posterior VAP/100b with {top_lift_bat['post_consistency_pct']:.1f}% posterior match consistency."
    })

    # 2. Highest context-adjusted runs saved + wicket value among IPL bowlers
    bowl_est = bowl_df[(bowl_df["has_ipl"]) & (bowl_df["press_balls"] >= 100)].copy()
    top_vap_bowl = bowl_est.sort_values("posterior_vap", ascending=False).iloc[0]
    insights.append({
        "category": "Phase-Adjusted Bowling Prevention",
        "headline": f"{top_vap_bowl['bowler_name']} leads all established IPL bowlers with {top_vap_bowl['posterior_vap']:+.2f} Posterior VAP per 24 pressure balls",
        "evidence": f"Concedes {top_vap_bowl['raw_press_econ']:.2f} RPO under pressure ({top_vap_bowl['raw_econ_delta']:+.2f} vs baseline {top_vap_bowl['base_econ']:.2f}) across {int(top_vap_bowl['press_balls'])} legal pressure balls with {top_vap_bowl['press_dot_pct']:.1f}% dot rate."
    })

    # 3. Small-sample outlier caught by Empirical Bayes shrinkage
    small_bat = bat_df[(bat_df["has_ipl"]) & (bat_df["press_balls"] >= 45) & (bat_df["press_balls"] <= 95)].sort_values("raw_vap_per_100", ascending=False).iloc[0]
    insights.append({
        "category": "Bayesian Shrinkage Protection",
        "headline": f"{small_bat['striker_name']} posts {small_bat['raw_vap_per_100']:+.1f} Raw VAP/100b on {int(small_bat['press_balls'])} balls, pulled by {small_bat['shrinkage_pull_vap']:+.1f} via Empirical Bayes",
        "evidence": f"Classified as '{small_bat['confidence_tier']}'; posterior VAP settles at {small_bat['posterior_vap']:+.1f} (90% CI: [{small_bat['vap_ci_lower']:+.1f}, {small_bat['vap_ci_upper']:+.1f}]), preventing small-sample inflation over 300+ ball anchors."
    })

    # 4. Death-over finisher stability across all 36 sensitivity scenarios
    stable_bat = bat_est.sort_values(["boot_top10_pct", "grid_top10_pct"], ascending=False).iloc[0]
    insights.append({
        "category": "36-Grid & Bootstrap Stability",
        "headline": f"{stable_bat['striker_name']} achieves {stable_bat['boot_top10_pct']:.1f}% Top-10 inclusion across 500 match-cluster bootstrap resamples",
        "evidence": f"Remains in the Top 10 across {stable_bat['grid_top10_pct']:.1f}% of the 36 threshold x cutoff x shrinkage configurations (Median Rank: {stable_bat['boot_median_rank']:.1f}, 10th–90th Rank Interval: [{stable_bat['boot_p10_rank']:.0f}, {stable_bat['boot_p90_rank']:.0f}])."
    })

    # 5. Uncapped Indian domestic statistical replacement discovery
    uncapped_bat_top = bat_df[(bat_df["is_uncapped_indian"]) & (bat_df["press_balls"] >= 60)].sort_values("prs_score", ascending=False).iloc[0]
    uncapped_bowl_top = bowl_df[(bowl_df["is_uncapped_indian"]) & (bowl_df["press_balls"] >= 60)].sort_values("prs_score", ascending=False).iloc[0]
    insights.append({
        "category": "Uncapped Domestic Arbitrage",
        "headline": f"Top Uncapped Indian Targets: {uncapped_bat_top['striker_name']} (PRS {uncapped_bat_top['prs_score']:.1f}) & {uncapped_bowl_top['bowler_name']} (PRS {uncapped_bowl_top['prs_score']:.1f})",
        "evidence": f"{uncapped_bat_top['striker_name']} delivers {uncapped_bat_top['raw_press_sr']:.1f} Pressure SR ({uncapped_bat_top['posterior_vap']:+.1f} Post VAP); {uncapped_bowl_top['bowler_name']} delivers {uncapped_bowl_top['raw_press_sr']:.2f} Pressure Bowling SR ({uncapped_bowl_top['raw_econ_delta']:+.2f} Econ Delta)."
    })

    # 6. Left-hand / Spin Matchup Asymmetric Specialist
    lhb_killer = bowl_est[bowl_est["vs_lhb_balls"] >= 50].sort_values("vs_lhb_vap", ascending=False).iloc[0]
    insights.append({
        "category": "Handedness Matchup Intelligence",
        "headline": f"{lhb_killer['bowler_name']} dominates Left-Handed Batters under pressure ({lhb_killer['vs_lhb_econ']:.2f} Econ, {lhb_killer['vs_lhb_vap']:+.2f} VAP/24b)",
        "evidence": f"Sample of {int(lhb_killer['vs_lhb_balls'])} high-pressure balls vs LHB compared to {lhb_killer['vs_rhb_econ']:.2f} Econ across {int(lhb_killer['vs_rhb_balls'])} balls vs RHB."
    })

    return insights

def main():
    with open(os.path.join(ANALYSIS_DIR, "config.json"), "r") as f:
        config = json.load(f)

    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print("1. Loading and cleaning official data pack...")
    bundle = load_and_clean_data(data_dir)
    bbb = bundle["bbb"]

    print("2. Running Data Quality & Validation Engine...")
    audit = run_data_audit(bundle)
    audit["checksums"] = bundle["checksums"]
    audit["model_version"] = config["model_version"]
    audit["analysis_date"] = config["analysis_date"]

    print("3. Computing Pre-Delivery Situational Pressure Index (1st vs 2nd Innings)...")
    bbb = compute_pressure_index(bbb)
    p_dist = summarize_pressure_distributions(bbb)

    print("4. Computing Contextual Value Above Par (VAP) & Wicket Cost Expectancy...")
    bbb, vap_diag = add_contextual_vap_columns(bbb)
    elig_df = build_player_eligibility_metadata(bundle)

    print("5. Fitting Empirical Bayes Shrinkage & Beta-Binomial Consistency Models...")
    bat_df, bat_eb_diag = compute_batter_reliability_table(bbb, elig_df, p_col="is_pressure_055")
    bowl_df, bowl_eb_diag = compute_bowler_reliability_table(bbb, elig_df, p_col="is_pressure_055")

    print("6. Running 36-Scenario Sensitivity Grid & 500-Iteration Match-Cluster Bootstrap...")
    bat_df, bowl_df, stab_diag = run_sensitivity_and_bootstrap(
        bbb, bat_df, bowl_df, n_boot=config["bootstrap_iterations"], seed=config["random_seed"]
    )

    print("7. Running RR Squad Gap Analysis, Jaipur Venue Model & 2027 Projections...")
    bat_df, bowl_df, rr_diag = analyze_rr_gaps_jaipur_and_projections(bbb, bat_df, bowl_df)

    print("8. Building Player Similarity & Statistical Replacement Engine...")
    sim_data = build_similarity_engine(bat_df, bowl_df)

    # Define Official Competition Rankings (Leaderboard A: IPL-eligible established >= 100 pressure balls, plus Overall Global >= 100b)
    off_bat_ipl_top10 = bat_df[(bat_df["has_ipl"]) & (bat_df["press_balls"] >= 100)].sort_values("prs_score", ascending=False).head(10).reset_index(drop=True)
    off_bat_ipl_top10["rank"] = np.arange(1, len(off_bat_ipl_top10) + 1)

    off_bowl_ipl_top10 = bowl_df[(bowl_df["has_ipl"]) & (bowl_df["press_balls"] >= 100)].sort_values("prs_score", ascending=False).head(10).reset_index(drop=True)
    off_bowl_ipl_top10["rank"] = np.arange(1, len(off_bowl_ipl_top10) + 1)

    # Define Track B: Uncapped Indian IPL Auction Scouting Top 10 (>= 60 pressure balls)
    scouting_bat_top10 = bat_df[(bat_df["is_uncapped_indian"]) & (bat_df["press_balls"] >= 50)].sort_values("prs_score", ascending=False).head(10).reset_index(drop=True)
    scouting_bat_top10["rank"] = np.arange(1, len(scouting_bat_top10) + 1)

    scouting_bowl_top10 = bowl_df[(bowl_df["is_uncapped_indian"]) & (bowl_df["press_balls"] >= 50)].sort_values("prs_score", ascending=False).head(10).reset_index(drop=True)
    scouting_bowl_top10["rank"] = np.arange(1, len(scouting_bowl_top10) + 1)

    print("9. Executing Automated Test Suite (Tests 1–10)...")
    tests_results = []
    tests_results.append(run_test_extras(bbb))
    tests_results.extend(run_test_scoring(bbb, audit["counts"]))
    tests_results.append(run_test_bowling(bbb))
    tests_results.extend(run_test_pressure(bbb))
    tests_results.extend(run_test_reliability(bbb, bat_df, bowl_df, off_bat_ipl_top10, off_bowl_ipl_top10))
    audit["automated_tests"] = tests_results

    print("10. Generating Automated Insights & Saving Outputs...")
    insights = generate_automated_insights(bat_df, bowl_df, sim_data, audit)

    # Export CSV deliverables
    bat_df.to_csv(os.path.join(output_dir, "official_batter_rankings.csv"), index=False)
    bowl_df.to_csv(os.path.join(output_dir, "official_bowler_rankings.csv"), index=False)

    combined_metrics = pd.concat([
        bat_df.assign(discipline="Batting"),
        bowl_df.assign(discipline="Bowling")
    ], ignore_index=True)
    combined_metrics.to_csv(os.path.join(output_dir, "player_metrics.csv"), index=False)

    scouting_pool = pd.concat([
        bat_df[bat_df["is_uncapped_indian"]].assign(discipline="Batting"),
        bowl_df[bowl_df["is_uncapped_indian"]].assign(discipline="Bowling")
    ], ignore_index=True)
    scouting_pool.to_csv(os.path.join(output_dir, "auction_scouting_pool.csv"), index=False)

    stab_cols_bat = ["player_id", "striker_name", "unstop_name", "role", "press_balls", "prs_score", "boot_top10_pct", "grid_top10_pct", "boot_median_rank", "boot_p10_rank", "boot_p90_rank", "boot_rank_sd", "stability_tier"]
    stab_cols_bowl = ["player_id", "bowler_name", "unstop_name", "role", "press_balls", "prs_score", "boot_top10_pct", "grid_top10_pct", "boot_median_rank", "boot_p10_rank", "boot_p90_rank", "boot_rank_sd", "stability_tier"]
    pd.concat([
        bat_df[stab_cols_bat].rename(columns={"striker_name": "player_name"}).assign(discipline="Batting"),
        bowl_df[stab_cols_bowl].rename(columns={"bowler_name": "player_name"}).assign(discipline="Bowling")
    ], ignore_index=True).to_csv(os.path.join(output_dir, "ranking_stability.csv"), index=False)

    # Global matchup matrix (RHB/LHB vs Pace/Spin under pressure)
    pr_sub = bbb[(bbb["is_pressure_055"] == 1) & (bbb["is_ball_faced"] == 1)].copy()
    matchup_df = pr_sub.groupby(["batting_type", "pace_spin_clean", "bowling_type"], observed=True).agg(
        balls=("is_ball_faced", "sum"),
        runs=("batter_runs", "sum"),
        wkts=("is_batter_dismissed", "sum"),
        dots=("batter_runs", lambda s: int((s == 0).sum())),
        bnds=("batter_runs", lambda s: int(s.isin([4, 6]).sum())),
        vap_sum=("bat_vap_ball", "sum")
    ).reset_index()
    matchup_df = matchup_df[matchup_df["balls"] >= 30].copy()
    matchup_df["sr"] = (100.0 * matchup_df["runs"] / matchup_df["balls"]).round(2)
    matchup_df["avg"] = (matchup_df["runs"] / np.maximum(1, matchup_df["wkts"])).round(2)
    matchup_df["dot_pct"] = (100.0 * matchup_df["dots"] / matchup_df["balls"]).round(1)
    matchup_df["bnd_pct"] = (100.0 * matchup_df["bnds"] / matchup_df["balls"]).round(1)
    matchup_df["vap_per_100"] = (100.0 * matchup_df["vap_sum"] / matchup_df["balls"]).round(2)
    matchup_df.to_csv(os.path.join(output_dir, "matchup_analysis.csv"), index=False)

    # Global phase summary
    phase_df = bbb[bbb["phase"].isin(["Powerplay", "Middle", "Death"])].groupby(["is_ipl", "phase", "is_pressure_055"]).agg(
        balls=("is_legal_ball", "sum"),
        runs=("total_runs", "sum"),
        wkts=("is_wicket", "sum"),
        mean_p=("pressure_index", "mean")
    ).reset_index()
    phase_df["rpo"] = (6.0 * phase_df["runs"] / np.maximum(1, phase_df["balls"])).round(2)
    phase_df.to_csv(os.path.join(output_dir, "phase_analysis.csv"), index=False)

    with open(os.path.join(output_dir, "data_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)

    dashboard_payload = {
        "meta": {
            "model_version": config["model_version"],
            "analysis_date": config["analysis_date"],
            "random_seed": config["random_seed"],
            "checksums": bundle["checksums"],
            "eb_diagnostics": {**bat_eb_diag, **bowl_eb_diag},
            "vap_diagnostics": vap_diag,
            "stability_diagnostics": stab_diag,
        },
        "audit": audit,
        "pressure_distributions": p_dist,
        "insights": insights,
        "official_bat_top10": off_bat_ipl_top10.to_dict(orient="records"),
        "official_bowl_top10": off_bowl_ipl_top10.to_dict(orient="records"),
        "scouting_bat_top10": scouting_bat_top10.to_dict(orient="records"),
        "scouting_bowl_top10": scouting_bowl_top10.to_dict(orient="records"),
        "batters_pool": bat_df[(bat_df["has_ipl"] | bat_df["has_india_a"]) & (bat_df["press_balls"] >= 45)].head(45).to_dict(orient="records"),
        "bowlers_pool": bowl_df[(bowl_df["has_ipl"] | bowl_df["has_india_a"]) & (bowl_df["press_balls"] >= 45)].head(45).to_dict(orient="records"),
        "global_t20i_bat_top10": bat_df[bat_df["press_balls"] >= 100].sort_values("prs_score", ascending=False).head(10).to_dict(orient="records"),
        "global_t20i_bowl_top10": bowl_df[bowl_df["press_balls"] >= 100].sort_values("prs_score", ascending=False).head(10).to_dict(orient="records"),
        "matchup_matrix": matchup_df.to_dict(orient="records"),
        "similarity": sim_data,
        "rr_intelligence": rr_diag,
    }
    with open(os.path.join(output_dir, "dashboard_data.json"), "w") as f:
        json.dump(dashboard_payload, f, indent=2)

    print("\n=== PIPELINE COMPLETE ===")
    print("Automated Tests Passed:", sum(1 for t in tests_results if t["status"] == "PASS"), "/", len(tests_results))
    print("\n--- LEADERBOARD A: OFFICIAL TOP 10 BATTERS (IPL-ELIGIBLE, >=100 PRESSURE BALLS) ---")
    print(off_bat_ipl_top10[["rank", "striker_name", "unstop_name", "role", "press_balls", "press_matches", "base_sr", "raw_press_sr", "adj_press_sr", "posterior_vap", "post_consistency_pct", "boot_top10_pct", "prs_score", "stability_tier"]].to_string(index=False))
    print("\n--- LEADERBOARD A: OFFICIAL TOP 10 BOWLERS (IPL-ELIGIBLE, >=100 PRESSURE BALLS) ---")
    print(off_bowl_ipl_top10[["rank", "bowler_name", "unstop_name", "role", "press_balls", "press_matches", "base_econ", "raw_press_econ", "adj_press_econ", "raw_press_sr", "posterior_vap", "post_consistency_pct", "boot_top10_pct", "prs_score", "stability_tier"]].to_string(index=False))
    print("\n--- LEADERBOARD B: RR AUCTION SCOUTING TOP 10 UNCAPPED INDIAN BATTERS (>=60 PRESSURE BALLS) ---")
    print(scouting_bat_top10[["rank", "striker_name", "unstop_name", "role", "press_balls", "press_matches", "base_sr", "raw_press_sr", "adj_press_sr", "posterior_vap", "post_consistency_pct", "prs_score", "confidence_tier"]].to_string(index=False))
    print("\n--- LEADERBOARD B: RR AUCTION SCOUTING TOP 10 UNCAPPED INDIAN BOWLERS (>=60 PRESSURE BALLS) ---")
    print(scouting_bowl_top10[["rank", "bowler_name", "unstop_name", "role", "press_balls", "press_matches", "base_econ", "raw_press_econ", "adj_press_econ", "raw_press_sr", "posterior_vap", "post_consistency_pct", "prs_score", "confidence_tier"]].to_string(index=False))

if __name__ == "__main__":
    main()
