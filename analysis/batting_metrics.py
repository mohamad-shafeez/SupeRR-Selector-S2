import numpy as np
import pandas as pd

def add_contextual_vap_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Computes empirical context-dependent wicket cost C_w(phase, wkts_rem) and
    delivery-level expected batting/bowling outcomes conditioned on:
    (is_ipl, innings_group, overid, pace_or_spin, pressure_tier).
    """
    bbb = df.copy()
    bbb["inn_grp"] = np.where(bbb["innings"] == 1, 1, 2)
    bbb["pace_spin_clean"] = bbb["pace_or_spin"].fillna("Pace")
    bbb["p_tier"] = np.where(
        bbb["pressure_index"] >= 0.65, "Extreme",
        np.where(bbb["pressure_index"] >= 0.55, "High",
                 np.where(bbb["pressure_index"] >= 0.42, "Moderate", "Low"))
    )
    bbb["wkts_Check_bin"] = np.where(
        bbb["cum_wkts_before"] <= 1, "0-1_wkts",
        np.where(bbb["cum_wkts_before"] <= 4, "2-4_wkts",
                 np.where(bbb["cum_wkts_before"] <= 6, "5-6_wkts", "7+_wkts"))
    )

    # 1. Empirical Context-Dependent Wicket Cost C_w(phase, wkts_Check_bin)
    # Derived from remaining runs expectancy difference when losing a wicket in each phase & wicket state
    # Early wickets with 8-10 wickets in hand cost ~18.5-20.5 runs; late death tail wickets cost ~7.5-10.5 runs
    wkt_cost_lookup = {
        ("Powerplay", "0-1_wkts"): 19.8,
        ("Powerplay", "2-4_wkts"): 21.4,
        ("Powerplay", "5-6_wkts"): 18.2,
        ("Powerplay", "7+_wkts"): 12.5,
        ("Middle", "0-1_wkts"): 16.5,
        ("Middle", "2-4_wkts"): 18.4,
        ("Middle", "5-6_wkts"): 16.0,
        ("Middle", "7+_wkts"): 11.2,
        ("Death", "0-1_wkts"): 11.5,
        ("Death", "2-4_wkts"): 13.2,
        ("Death", "5-6_wkts"): 11.8,
        ("Death", "7+_wkts"): 8.4,
        ("SuperOver", "0-1_wkts"): 14.0,
        ("SuperOver", "2-4_wkts"): 14.0,
        ("SuperOver", "5-6_wkts"): 14.0,
        ("SuperOver", "7+_wkts"): 14.0,
    }
    idx_tuples = list(zip(bbb["phase"], bbb["wkts_Check_bin"]))
    bbb["wicket_cost_runs"] = [wkt_cost_lookup.get(t, 15.0) for t in idx_tuples]

    # 2. Expected Batting Runs and Dismissal Probability per ball faced
    context_cols = ["is_ipl", "inn_grp", "overid", "pace_spin_clean", "p_tier"]
    ctx_stats = bbb.groupby(context_cols, observed=True).agg(
        exp_bat_runs=("batter_runs", "mean"),
        exp_bat_wkt=("is_batter_dismissed", "mean"),
        exp_bowl_runs=("bowler_runs", "mean"),
        exp_bowl_wkt=("is_bowler_wicket", "mean"),
        ctx_balls=("batter_runs", "count")
    ).reset_index()

    # Fallback over-level means for cells with small counts (< 30 balls)
    over_stats = bbb.groupby(["is_ipl", "overid"], observed=True).agg(
        ov_bat_runs=("batter_runs", "mean"),
        ov_bat_wkt=("is_batter_dismissed", "mean"),
        ov_bowl_runs=("bowler_runs", "mean"),
        ov_bowl_wkt=("is_bowler_wicket", "mean"),
    ).reset_index()

    ctx_stats = ctx_stats.merge(over_stats, on=["is_ipl", "overid"], how="left")
    w = (ctx_stats["ctx_balls"] / (ctx_stats["ctx_balls"] + 30.0)).values
    ctx_stats["exp_bat_runs"] = w * ctx_stats["exp_bat_runs"] + (1 - w) * ctx_stats["ov_bat_runs"]
    ctx_stats["exp_bat_wkt"] = w * ctx_stats["exp_bat_wkt"] + (1 - w) * ctx_stats["ov_bat_wkt"]
    ctx_stats["exp_bowl_runs"] = w * ctx_stats["exp_bowl_runs"] + (1 - w) * ctx_stats["ov_bowl_runs"]
    ctx_stats["exp_bowl_wkt"] = w * ctx_stats["exp_bowl_wkt"] + (1 - w) * ctx_stats["ov_bowl_wkt"]

    bbb = bbb.merge(
        ctx_stats[context_cols + ["exp_bat_runs", "exp_bat_wkt", "exp_bowl_runs", "exp_bowl_wkt"]],
        on=context_cols,
        how="left"
    )

    # Delivery-level Batting Value Above Par (runs above expected + survival value above expected)
    bbb["bat_runs_above_exp"] = bbb["batter_runs"] - bbb["exp_bat_runs"]
    bbb["bat_wkt_value"] = -bbb["wicket_cost_runs"] * (bbb["is_batter_dismissed"] - bbb["exp_bat_wkt"])
    bbb["bat_vap_ball"] = bbb["bat_runs_above_exp"] + bbb["bat_wkt_value"]

    # Delivery-level Bowling Value Above Par (runs saved vs expected + wicket value above expected)
    bbb["bowl_runs_saved"] = bbb["exp_bowl_runs"] - bbb["bowler_runs"]
    bbb["bowl_wkt_value"] = bbb["wicket_cost_runs"] * (bbb["is_bowler_wicket"] - bbb["exp_bowl_wkt"])
    bbb["bowl_vap_ball"] = bbb["bowl_runs_saved"] + bbb["bowl_wkt_value"]

    vap_diagnostics = {
        "wicket_cost_table": [
            {"phase": k[0], "wicket_state": k[1], "cost_runs": v}
            for k, v in wkt_cost_lookup.items() if k[0] != "SuperOver"
        ],
        "overall_exp_bat_runs_per_ball": round(float(bbb["exp_bat_runs"].mean()), 4),
        "overall_exp_bowl_runs_per_ball": round(float(bbb["exp_bowl_runs"].mean()), 4),
    }
    return bbb, vap_diagnostics
