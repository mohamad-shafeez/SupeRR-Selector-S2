import numpy as np
import pandas as pd

def get_confidence_tier(press_balls: int) -> str:
    if press_balls >= 300:
        return "Strong Evidence (300+b)"
    elif press_balls >= 150:
        return "Established Sample (150–299b)"
    elif press_balls >= 50:
        return "Emerging Evidence (50–149b)"
    return "Exploratory (<50b)"

def estimate_eb_k(player_means: np.ndarray, player_counts: np.ndarray, within_var: float, fallback_k: float = 95.0) -> float:
    """
    Estimates Empirical Bayes shrinkage constant k = sigma^2_within / tau^2_between
    using precision-weighted method of moments across players with >= 30 pressure balls.
    """
    mask = player_counts >= 30
    if mask.sum() < 10:
        return fallback_k
    m = player_means[mask]
    n = player_counts[mask]
    total_var = float(np.var(m, ddof=1))
    expected_sampling_var = float(np.mean(within_var / n))
    between_var = max(1e-5, total_var - expected_sampling_var)
    k_est = within_var / between_var
    return float(np.clip(k_est, 45.0, 160.0))

def fit_beta_binomial_prior(successes: np.ndarray, trials: np.ndarray) -> tuple[float, float]:
    """
    Fits Beta-Binomial prior (alpha_0, beta_0) via method of moments on players with >= 5 matches.
    """
    mask = trials >= 5
    if mask.sum() < 10:
        return 4.2, 2.8
    p = successes[mask] / trials[mask]
    mean_p = float(np.clip(np.mean(p), 0.25, 0.80))
    # Prior effective sample size M_0 = 7 matches (prevents 5/5 from beating 35/42)
    m0 = 7.5
    return round(mean_p * m0, 3), round((1.0 - mean_p) * m0, 3)

def percentile_score(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series(50.0, index=series.index)
    r = series.rank(pct=True)
    return (25.0 + 74.5 * r).clip(10.0, 99.8)

def compute_batter_reliability_table(
    bbb: pd.DataFrame,
    elig_df: pd.DataFrame,
    p_col: str = "is_pressure_055",
    k_override: float | None = None,
    ipl_weight_bonus: float = 0.20,
) -> tuple[pd.DataFrame, dict]:
    # Filter to valid balls faced by striker
    bf = bbb[bbb["is_ball_faced"] == 1].copy()
    bf["w_stakes"] = 1.0 + ipl_weight_bonus * bf["is_ipl"] + 0.10 * bf["is_knockout"]
    bf["is_bnd"] = (bf["is_four"].fillna(False) | bf["is_six"].fillna(False)).astype(int)
    bf["is_dot_bat"] = (bf["batter_runs"] == 0).astype(int)

    # Overall baseline per striker_id
    base = bf.groupby("striker_id").agg(
        striker_name=("striker_name", "first"),
        unstop_name=("unstop_striker_name", "first"),
        hand=("batting_type", "first"),
        total_balls=("is_ball_faced", "sum"),
        total_matches=("match_id", "nunique"),
        total_runs=("batter_runs", "sum"),
        total_dismissals=("is_batter_dismissed", "sum"),
        avg_bat_order=("batting_order", "mean"),
        ipl_balls=("is_ipl", "sum"),
    ).reset_index().rename(columns={"striker_id": "player_id"})

    base["base_sr"] = 100.0 * base["total_runs"] / np.maximum(1, base["total_balls"])
    base["base_avg"] = base["total_runs"] / np.maximum(1, base["total_dismissals"])

    # Assign data-driven Batting Role from average batting order
    base["role"] = np.where(
        base["avg_bat_order"] <= 2.35, "Opener (1–2)",
        np.where(base["avg_bat_order"] <= 3.85, "Top-Order (3)",
                 np.where(base["avg_bat_order"] <= 5.45, "Middle-Order (4–5)", "Finisher (6–7)"))
    )

    # Pressure subset
    pr = bf[bf[p_col] == 1].copy()
    pr["weighted_vap"] = pr["bat_vap_ball"] * pr["w_stakes"]
    within_var_vap = float(np.var(pr["bat_vap_ball"] * 100.0)) if len(pr) else 25000.0

    pr_agg = pr.groupby("striker_id").agg(
        press_balls=("is_ball_faced", "sum"),
        press_matches=("match_id", "nunique"),
        press_runs=("batter_runs", "sum"),
        press_dismissals=("is_batter_dismissed", "sum"),
        press_dots=("is_dot_bat", "sum"),
        press_bnds=("is_bnd", "sum"),
        press_fours=("is_four", "sum"),
        press_sixes=("is_six", "sum"),
        raw_vap_sum=("bat_vap_ball", "sum"),
        weighted_vap_sum=("weighted_vap", "sum"),
        mean_p_faced=("pressure_index", "mean"),
        ipl_press_balls=("is_ipl", "sum"),
    ).reset_index().rename(columns={"striker_id": "player_id"})

    df = base.merge(pr_agg, on="player_id", how="inner")
    df = df[df["press_balls"] >= 20].copy()

    df["raw_press_sr"] = 100.0 * df["press_runs"] / df["press_balls"]
    df["raw_press_avg"] = df["press_runs"] / np.maximum(1, df["press_dismissals"])
    df["press_dot_pct"] = 100.0 * df["press_dots"] / df["press_balls"]
    df["press_bnd_pct"] = 100.0 * df["press_bnds"] / df["press_balls"]
    df["raw_sr_delta"] = df["raw_press_sr"] - df["base_sr"]
    df["raw_vap_per_100"] = 100.0 * df["raw_vap_sum"] / df["press_balls"]
    df["weighted_vap_per_100"] = 100.0 * df["weighted_vap_sum"] / df["press_balls"]
    df["press_exposure_pct"] = 100.0 * df["press_balls"] / np.maximum(1, df["total_balls"])

    # Role-specific prior means under pressure
    role_priors = df.groupby("role").apply(
        lambda g: pd.Series({
            "role_prior_sr": float(100.0 * g["press_runs"].sum() / max(1, g["press_balls"].sum())),
            "role_prior_vap": float(100.0 * g["weighted_vap_sum"].sum() / max(1, g["press_balls"].sum())),
        }),
        include_groups=False
    ).reset_index()
    df = df.merge(role_priors, on="role", how="left")

    # Estimate Empirical Bayes shrinkage constant k from data variance components
    k_eb = k_override if k_override is not None else estimate_eb_k(
        df["weighted_vap_per_100"].values, df["press_balls"].values, within_var_vap, fallback_k=92.0
    )
    shrink_w = df["press_balls"] / (df["press_balls"] + k_eb)
    df["adj_press_sr"] = shrink_w * df["raw_press_sr"] + (1.0 - shrink_w) * df["role_prior_sr"]
    df["posterior_vap"] = shrink_w * df["weighted_vap_per_100"] + (1.0 - shrink_w) * df["role_prior_vap"]
    df["shrinkage_pull_vap"] = df["posterior_vap"] - df["weighted_vap_per_100"]
    df["role_adjusted_lift"] = df["adj_press_sr"] - df["role_prior_sr"]

    # Posterior standard error and 90% credible interval for Posterior VAP per 100 balls
    se_vap = np.sqrt(within_var_vap / (df["press_balls"] + k_eb))
    df["vap_ci_lower"] = df["posterior_vap"] - 1.645 * se_vap
    df["vap_ci_upper"] = df["posterior_vap"] + 1.645 * se_vap

    # Match-level consistency & downside metrics (on pressure match appearances with >= 3 balls)
    pm = pr.groupby(["striker_id", "match_id"]).agg(
        m_balls=("is_ball_faced", "sum"),
        m_runs=("batter_runs", "sum"),
        m_vap=("bat_vap_ball", "sum"),
    ).reset_index()
    pm_qual = pm[pm["m_balls"] >= 3].copy()
    pm_qual["m_vap_per_10"] = 10.0 * pm_qual["m_vap"] / pm_qual["m_balls"]
    pm_qual["is_pos_match"] = (pm_qual["m_vap"] > 0).astype(int)

    m_stats = pm_qual.groupby("striker_id").agg(
        qual_press_matches=("match_id", "count"),
        pos_press_matches=("is_pos_match", "sum"),
        match_vap_mean=("m_vap_per_10", "mean"),
        match_vap_median=("m_vap_per_10", "median"),
        match_vap_sd=("m_vap_per_10", lambda s: float(np.std(s, ddof=0))),
        match_vap_iqr=("m_vap_per_10", lambda s: float(np.percentile(s, 75) - np.percentile(s, 25))),
        worst_match_vap=("m_vap_per_10", "min"),
        p10_downside_vap=("m_vap_per_10", lambda s: float(np.percentile(s, 20))),
    ).reset_index().rename(columns={"striker_id": "player_id"})

    df = df.merge(m_stats, on="player_id", how="left")
    df["qual_press_matches"] = df["qual_press_matches"].fillna(df["press_matches"]).clip(1, None)
    df["pos_press_matches"] = df["pos_press_matches"].fillna((df["qual_press_matches"] * 0.5).round())
    for c in ["match_vap_mean", "match_vap_median", "match_vap_sd", "match_vap_iqr", "worst_match_vap", "p10_downside_vap"]:
        df[c] = df[c].fillna(0.0)

    df["obs_consistency_pct"] = 100.0 * df["pos_press_matches"] / df["qual_press_matches"]
    a0, b0 = fit_beta_binomial_prior(df["pos_press_matches"].values, df["qual_press_matches"].values)
    df["post_consistency_pct"] = 100.0 * (df["pos_press_matches"] + a0) / (df["qual_press_matches"] + a0 + b0)

    # Phase splits under pressure (Powerplay, Middle, Death)
    for ph in ["Powerplay", "Middle", "Death"]:
        ph_sub = pr[pr["phase"] == ph].groupby("striker_id").agg(
            balls=("is_ball_faced", "sum"),
            runs=("batter_runs", "sum"),
            vap=("bat_vap_ball", "sum"),
        ).reset_index().rename(columns={"striker_id": "player_id"})
        ph_sub[f"ph_{ph.lower()}_balls"] = ph_sub["balls"]
        ph_sub[f"ph_{ph.lower()}_sr"] = 100.0 * ph_sub["runs"] / np.maximum(1, ph_sub["balls"])
        ph_sub[f"ph_{ph.lower()}_vap"] = 100.0 * ph_sub["vap"] / np.maximum(1, ph_sub["balls"])
        df = df.merge(
            ph_sub[["player_id", f"ph_{ph.lower()}_balls", f"ph_{ph.lower()}_sr", f"ph_{ph.lower()}_vap"]],
            on="player_id", how="left"
        )
        df[f"ph_{ph.lower()}_balls"] = df[f"ph_{ph.lower()}_balls"].fillna(0).astype(int)
        df[f"ph_{ph.lower()}_sr"] = df[f"ph_{ph.lower()}_sr"].fillna(0.0)
        df[f"ph_{ph.lower()}_vap"] = df[f"ph_{ph.lower()}_vap"].fillna(0.0)

    # Pace vs Spin pressure splits
    for style in ["Pace", "Spin"]:
        st_sub = pr[pr["pace_spin_clean"] == style].groupby("striker_id").agg(
            balls=("is_ball_faced", "sum"),
            runs=("batter_runs", "sum"),
            wkts=("is_batter_dismissed", "sum"),
            vap=("bat_vap_ball", "sum"),
        ).reset_index().rename(columns={"striker_id": "player_id"})
        key = style.lower()
        st_sub[f"vs_{key}_balls"] = st_sub["balls"]
        st_sub[f"vs_{key}_sr"] = 100.0 * st_sub["runs"] / np.maximum(1, st_sub["balls"])
        st_sub[f"vs_{key}_vap"] = 100.0 * st_sub["vap"] / np.maximum(1, st_sub["balls"])
        df = df.merge(
            st_sub[["player_id", f"vs_{key}_balls", f"vs_{key}_sr", f"vs_{key}_vap"]],
            on="player_id", how="left"
        )
        df[f"vs_{key}_balls"] = df[f"vs_{key}_balls"].fillna(0).astype(int)
        df[f"vs_{key}_sr"] = df[f"vs_{key}_sr"].fillna(0.0)
        df[f"vs_{key}_vap"] = df[f"vs_{key}_vap"].fillna(0.0)

    # Clutch micro-moment performance
    cl_sub = bf[bf["is_clutch_moment"] == 1].groupby("striker_id").agg(
        clutch_balls=("is_ball_faced", "sum"),
        clutch_matches=("match_id", "nunique"),
        clutch_runs=("batter_runs", "sum"),
        clutch_vap_sum=("bat_vap_ball", "sum"),
    ).reset_index().rename(columns={"striker_id": "player_id"})
    cl_sub["clutch_sr"] = 100.0 * cl_sub["clutch_runs"] / np.maximum(1, cl_sub["clutch_balls"])
    cl_sub["clutch_vap_100"] = 100.0 * cl_sub["clutch_vap_sum"] / np.maximum(1, cl_sub["clutch_balls"])
    cl_sub["adj_clutch_vap"] = (cl_sub["clutch_balls"] / (cl_sub["clutch_balls"] + 45.0)) * cl_sub["clutch_vap_100"]
    df = df.merge(
        cl_sub[["player_id", "clutch_balls", "clutch_matches", "clutch_runs", "clutch_sr", "clutch_vap_100", "adj_clutch_vap"]],
        on="player_id", how="left"
    )
    for c in ["clutch_balls", "clutch_matches", "clutch_runs"]:
        df[c] = df[c].fillna(0).astype(int)
    for c in ["clutch_sr", "clutch_vap_100", "adj_clutch_vap"]:
        df[c] = df[c].fillna(0.0)

    # Data-derived Archetype
    def assign_bat_archetype(r):
        if r["ph_death_balls"] >= 0.42 * r["press_balls"] and r["ph_death_sr"] >= 165.0:
            return "Death-Over Finisher"
        elif r["avg_bat_order"] <= 2.5 and r["ph_powerplay_sr"] >= 145.0:
            return "Powerplay Aggressor"
        elif r["vs_spin_balls"] >= 25 and r["vs_spin_sr"] >= r["vs_pace_sr"] + 12.0:
            return "Middle-Over Spin Attacker"
        elif r["post_consistency_pct"] >= 62.0 and r["raw_press_avg"] >= 35.0:
            return "Collapse Stabilizer & Chase Anchor"
        elif r["raw_sr_delta"] >= 20.0:
            return "High-Gear Pressure Accelerator"
        return "All-Phase Middle-Order Batter"

    df["archetype"] = df.apply(assign_bat_archetype, axis=1)
    df["confidence_tier"] = df["press_balls"].apply(get_confidence_tier)

    # Merge eligibility metadata
    df = df.merge(
        elig_df[["player_id", "has_ipl", "is_active_2027", "has_india_a", "has_t20i", "is_uncapped_indian", "is_overseas", "is_rr_player", "latest_ipl_team", "teams_list"]],
        on="player_id", how="left"
    )

    # Compute Parallel Scores:
    # 1. Pressure Impact Score (PIS, 0-100): Posterior VAP (60%) + Role-Adjusted SR Lift (40%)
    s_vap = percentile_score(df["posterior_vap"])
    s_lift = percentile_score(df["role_adjusted_lift"] + 0.5 * df["raw_sr_delta"])
    s_cons = percentile_score(df["post_consistency_pct"])
    m_w = df["qual_press_matches"] / (df["qual_press_matches"] + 14.0)
    shrunk_floor = m_w * df["p10_downside_vap"] + (1.0 - m_w) * float(df["p10_downside_vap"].median())
    df["p10_downside_vap"] = (0.65 * (df["vap_ci_lower"] / 4.0) + 0.35 * shrunk_floor).round(2)
    s_down = percentile_score(df["p10_downside_vap"])
    s_conf = 100.0 * (df["press_balls"] / (df["press_balls"] + 75.0))

    df["pressure_impact_score"] = (0.60 * s_vap + 0.40 * s_lift).round(2)
    df["prs_score"] = (
        0.35 * s_vap
        + 0.25 * s_cons
        + 0.15 * s_down
        + 0.15 * s_lift
        + 0.10 * s_conf
    ).round(2)

    df = df.sort_values("prs_score", ascending=False).reset_index(drop=True)
    return df, {
        "k_eb_batter": round(k_eb, 2),
        "within_var_batter_vap": round(within_var_vap, 2),
        "beta_prior_alpha": a0,
        "beta_prior_beta": b0,
    }

def compute_bowler_reliability_table(
    bbb: pd.DataFrame,
    elig_df: pd.DataFrame,
    p_col: str = "is_pressure_055",
    k_override: float | None = None,
    ipl_weight_bonus: float = 0.20,
) -> tuple[pd.DataFrame, dict]:
    bf = bbb.copy()
    bf["w_stakes"] = 1.0 + ipl_weight_bonus * bf["is_ipl"] + 0.10 * bf["is_knockout"]
    bf["is_bnd_conceded"] = (bf["is_four"].fillna(False) | bf["is_six"].fillna(False)).astype(int)
    bf["is_dot_bowl"] = ((bf["bowler_runs"] == 0) & (bf["is_legal_ball"] == 1)).astype(int)

    base = bf.groupby("bowler_id").agg(
        bowler_name=("bowler_name", "first"),
        unstop_name=("unstop_bowler_name", "first"),
        bowling_type=("bowling_type", "first"),
        pace_or_spin=("pace_spin_clean", "first"),
        total_legal_balls=("is_legal_ball", "sum"),
        total_matches=("match_id", "nunique"),
        total_runs_conceded=("bowler_runs", "sum"),
        total_wickets=("is_bowler_wicket", "sum"),
        avg_over=("overid", "mean"),
        ipl_balls=("is_ipl", "sum"),
    ).reset_index().rename(columns={"bowler_id": "player_id"})

    base["base_econ"] = 6.0 * base["total_runs_conceded"] / np.maximum(1, base["total_legal_balls"])
    base["base_sr"] = base["total_legal_balls"] / np.maximum(1, base["total_wickets"])

    # Phase shares for role assignment
    ph_counts = bf.groupby(["bowler_id", "phase"])["is_legal_ball"].sum().unstack(fill_value=0)
    for ph in ["Powerplay", "Middle", "Death"]:
        if ph not in ph_counts.columns:
            ph_counts[ph] = 0
    ph_shares = ph_counts.div(ph_counts.sum(axis=1).replace(0, 1), axis=0).reset_index().rename(columns={"bowler_id": "player_id"})
    base = base.merge(ph_shares[["player_id", "Powerplay", "Middle", "Death"]], on="player_id", how="left")

    def assign_bowl_role(r):
        if r["Death"] >= 0.35:
            return "Death-Over Specialist"
        elif r["Powerplay"] >= 0.45:
            return "Powerplay New-Ball Specialist"
        elif r["Middle"] >= 0.55:
            return f"Middle-Over {'Spin' if r['pace_or_spin'] == 'Spin' else 'Pace'} Controller"
        return f"All-Phase {'Spinner' if r['pace_or_spin'] == 'Spin' else 'Pacer'}"

    base["role"] = base.apply(assign_bowl_role, axis=1)

    pr = bf[bf[p_col] == 1].copy()
    pr["weighted_vap"] = pr["bowl_vap_ball"] * pr["w_stakes"]
    within_var_vap = float(np.var(pr["bowl_vap_ball"] * 24.0)) if len(pr) else 1800.0

    pr_agg = pr.groupby("bowler_id").agg(
        press_balls=("is_legal_ball", "sum"),
        press_deliveries=("is_legal_ball", "count"),
        press_matches=("match_id", "nunique"),
        press_runs=("bowler_runs", "sum"),
        press_wickets=("is_bowler_wicket", "sum"),
        press_dots=("is_dot_bowl", "sum"),
        press_bnds=("is_bnd_conceded", "sum"),
        raw_vap_sum=("bowl_vap_ball", "sum"),
        weighted_vap_sum=("weighted_vap", "sum"),
        runs_saved_sum=("bowl_runs_saved", "sum"),
        mean_p_bowled=("pressure_index", "mean"),
        ipl_press_balls=("is_ipl", "sum"),
    ).reset_index().rename(columns={"bowler_id": "player_id"})

    df = base.merge(pr_agg, on="player_id", how="inner")
    df = df[df["press_balls"] >= 24].copy()

    df["raw_press_econ"] = 6.0 * df["press_runs"] / df["press_balls"]
    df["raw_press_sr"] = df["press_balls"] / np.maximum(1, df["press_wickets"])
    df["press_dot_pct"] = 100.0 * df["press_dots"] / df["press_balls"]
    df["press_bnd_pct"] = 100.0 * df["press_bnds"] / df["press_balls"]
    df["raw_econ_delta"] = df["raw_press_econ"] - df["base_econ"]  # negative is better!
    df["runs_saved_per_24"] = 24.0 * df["runs_saved_sum"] / df["press_balls"]
    df["raw_vap_per_24"] = 24.0 * df["raw_vap_sum"] / df["press_balls"]
    df["weighted_vap_per_24"] = 24.0 * df["weighted_vap_sum"] / df["press_balls"]
    df["press_exposure_pct"] = 100.0 * df["press_balls"] / np.maximum(1, df["total_legal_balls"])

    role_priors = df.groupby("role").apply(
        lambda g: pd.Series({
            "role_prior_econ": float(6.0 * g["press_runs"].sum() / max(1, g["press_balls"].sum())),
            "role_prior_vap": float(24.0 * g["weighted_vap_sum"].sum() / max(1, g["press_balls"].sum())),
        }),
        include_groups=False
    ).reset_index()
    df = df.merge(role_priors, on="role", how="left")

    k_eb = k_override if k_override is not None else estimate_eb_k(
        df["weighted_vap_per_24"].values, df["press_balls"].values, within_var_vap, fallback_k=102.0
    )
    shrink_w = df["press_balls"] / (df["press_balls"] + k_eb)
    df["adj_press_econ"] = shrink_w * df["raw_press_econ"] + (1.0 - shrink_w) * df["role_prior_econ"]
    df["posterior_vap"] = shrink_w * df["weighted_vap_per_24"] + (1.0 - shrink_w) * df["role_prior_vap"]
    df["shrinkage_pull_vap"] = df["posterior_vap"] - df["weighted_vap_per_24"]
    df["role_adjusted_lift"] = df["role_prior_econ"] - df["adj_press_econ"]  # positive = beats role par economy

    se_vap = np.sqrt(within_var_vap / (df["press_balls"] + k_eb))
    df["vap_ci_lower"] = df["posterior_vap"] - 1.645 * se_vap
    df["vap_ci_upper"] = df["posterior_vap"] + 1.645 * se_vap

    # Match-level consistency & downside risk
    pm = pr.groupby(["bowler_id", "match_id"]).agg(
        m_balls=("is_legal_ball", "sum"),
        m_runs=("bowler_runs", "sum"),
        m_wkts=("is_bowler_wicket", "sum"),
        m_vap=("bowl_vap_ball", "sum"),
    ).reset_index()
    pm_qual = pm[pm["m_balls"] >= 4].copy()
    pm_qual["m_vap_per_12"] = 12.0 * pm_qual["m_vap"] / pm_qual["m_balls"]
    pm_qual["is_pos_match"] = (pm_qual["m_vap"] > 0).astype(int)

    m_stats = pm_qual.groupby("bowler_id").agg(
        qual_press_matches=("match_id", "count"),
        pos_press_matches=("is_pos_match", "sum"),
        match_vap_mean=("m_vap_per_12", "mean"),
        match_vap_median=("m_vap_per_12", "median"),
        match_vap_sd=("m_vap_per_12", lambda s: float(np.std(s, ddof=0))),
        match_vap_iqr=("m_vap_per_12", lambda s: float(np.percentile(s, 75) - np.percentile(s, 25))),
        worst_match_vap=("m_vap_per_12", "min"),
        p10_downside_vap=("m_vap_per_12", lambda s: float(np.percentile(s, 20))),
    ).reset_index().rename(columns={"bowler_id": "player_id"})

    df = df.merge(m_stats, on="player_id", how="left")
    df["qual_press_matches"] = df["qual_press_matches"].fillna(df["press_matches"]).clip(1, None)
    df["pos_press_matches"] = df["pos_press_matches"].fillna((df["qual_press_matches"] * 0.5).round())
    for c in ["match_vap_mean", "match_vap_median", "match_vap_sd", "match_vap_iqr", "worst_match_vap", "p10_downside_vap"]:
        df[c] = df[c].fillna(0.0)

    df["obs_consistency_pct"] = 100.0 * df["pos_press_matches"] / df["qual_press_matches"]
    a0, b0 = fit_beta_binomial_prior(df["pos_press_matches"].values, df["qual_press_matches"].values)
    df["post_consistency_pct"] = 100.0 * (df["pos_press_matches"] + a0) / (df["qual_press_matches"] + a0 + b0)

    # Phase splits under pressure
    for ph in ["Powerplay", "Middle", "Death"]:
        ph_sub = pr[pr["phase"] == ph].groupby("bowler_id").agg(
            balls=("is_legal_ball", "sum"),
            runs=("bowler_runs", "sum"),
            wkts=("is_bowler_wicket", "sum"),
            vap=("bowl_vap_ball", "sum"),
        ).reset_index().rename(columns={"bowler_id": "player_id"})
        ph_sub[f"ph_{ph.lower()}_balls"] = ph_sub["balls"]
        ph_sub[f"ph_{ph.lower()}_econ"] = 6.0 * ph_sub["runs"] / np.maximum(1, ph_sub["balls"])
        ph_sub[f"ph_{ph.lower()}_vap"] = 24.0 * ph_sub["vap"] / np.maximum(1, ph_sub["balls"])
        df = df.merge(
            ph_sub[["player_id", f"ph_{ph.lower()}_balls", f"ph_{ph.lower()}_econ", f"ph_{ph.lower()}_vap"]],
            on="player_id", how="left"
        )
        df[f"ph_{ph.lower()}_balls"] = df[f"ph_{ph.lower()}_balls"].fillna(0).astype(int)
        df[f"ph_{ph.lower()}_econ"] = df[f"ph_{ph.lower()}_econ"].fillna(0.0)
        df[f"ph_{ph.lower()}_vap"] = df[f"ph_{ph.lower()}_vap"].fillna(0.0)

    # Matchups vs RHB and LHB
    for hand in ["RHB", "LHB"]:
        h_sub = pr[pr["batting_type"] == hand].groupby("bowler_id").agg(
            balls=("is_legal_ball", "sum"),
            runs=("bowler_runs", "sum"),
            wkts=("is_bowler_wicket", "sum"),
            vap=("bowl_vap_ball", "sum"),
        ).reset_index().rename(columns={"bowler_id": "player_id"})
        key = hand.lower()
        h_sub[f"vs_{key}_balls"] = h_sub["balls"]
        h_sub[f"vs_{key}_econ"] = 6.0 * h_sub["runs"] / np.maximum(1, h_sub["balls"])
        h_sub[f"vs_{key}_vap"] = 24.0 * h_sub["vap"] / np.maximum(1, h_sub["balls"])
        df = df.merge(
            h_sub[["player_id", f"vs_{key}_balls", f"vs_{key}_econ", f"vs_{key}_vap"]],
            on="player_id", how="left"
        )
        df[f"vs_{key}_balls"] = df[f"vs_{key}_balls"].fillna(0).astype(int)
        df[f"vs_{key}_econ"] = df[f"vs_{key}_econ"].fillna(0.0)
        df[f"vs_{key}_vap"] = df[f"vs_{key}_vap"].fillna(0.0)

    # Clutch micro-moments
    cl_sub = bf[bf["is_clutch_moment"] == 1].groupby("bowler_id").agg(
        clutch_balls=("is_legal_ball", "sum"),
        clutch_matches=("match_id", "nunique"),
        clutch_runs=("bowler_runs", "sum"),
        clutch_wkts=("is_bowler_wicket", "sum"),
        clutch_vap_sum=("bowl_vap_ball", "sum"),
    ).reset_index().rename(columns={"bowler_id": "player_id"})
    cl_sub["clutch_econ"] = 6.0 * cl_sub["clutch_runs"] / np.maximum(1, cl_sub["clutch_balls"])
    cl_sub["clutch_vap_24"] = 24.0 * cl_sub["clutch_vap_sum"] / np.maximum(1, cl_sub["clutch_balls"])
    cl_sub["adj_clutch_vap"] = (cl_sub["clutch_balls"] / (cl_sub["clutch_balls"] + 48.0)) * cl_sub["clutch_vap_24"]
    df = df.merge(
        cl_sub[["player_id", "clutch_balls", "clutch_matches", "clutch_wkts", "clutch_econ", "clutch_vap_24", "adj_clutch_vap"]],
        on="player_id", how="left"
    )
    for c in ["clutch_balls", "clutch_matches", "clutch_wkts"]:
        df[c] = df[c].fillna(0).astype(int)
    for c in ["clutch_econ", "clutch_vap_24", "adj_clutch_vap"]:
        df[c] = df[c].fillna(0.0)

    def assign_bowl_archetype(r):
        if r["ph_death_balls"] >= 0.35 * r["press_balls"] and r["raw_press_sr"] <= 14.5:
            return "Death-Over Strike Specialist"
        elif r["pace_or_spin"] == "Spin" and r["adj_press_econ"] <= 7.6:
            return "Middle-Over Spin Choker"
        elif r["pace_or_spin"] == "Spin" and r["raw_press_sr"] <= 13.5:
            return "Wicket-Taking Mystery/Wrist Spinner"
        elif r["ph_powerplay_balls"] >= 0.40 * r["press_balls"]:
            return "New-Ball Powerplay Enforcer"
        elif r["raw_press_sr"] <= 12.5:
            return "High-Leverage Partnership Breaker"
        return "Multi-Phase Economy Controller"

    df["archetype"] = df.apply(assign_bowl_archetype, axis=1)
    df["confidence_tier"] = df["press_balls"].apply(get_confidence_tier)

    df = df.merge(
        elig_df[["player_id", "has_ipl", "is_active_2027", "has_india_a", "has_t20i", "is_uncapped_indian", "is_overseas", "is_rr_player", "latest_ipl_team", "teams_list"]],
        on="player_id", how="left"
    )

    s_vap = percentile_score(df["posterior_vap"])
    s_lift = percentile_score(df["role_adjusted_lift"] - 0.5 * df["raw_econ_delta"])
    s_cons = percentile_score(df["post_consistency_pct"])
    m_w = df["qual_press_matches"] / (df["qual_press_matches"] + 14.0)
    shrunk_floor = m_w * df["p10_downside_vap"] + (1.0 - m_w) * float(df["p10_downside_vap"].median())
    df["p10_downside_vap"] = (0.65 * (df["vap_ci_lower"] / 4.0) + 0.35 * shrunk_floor).round(2)
    s_down = percentile_score(df["p10_downside_vap"])
    s_conf = 100.0 * (df["press_balls"] / (df["press_balls"] + 80.0))

    df["pressure_impact_score"] = (0.60 * s_vap + 0.40 * s_lift).round(2)
    df["prs_score"] = (
        0.35 * s_vap
        + 0.25 * s_cons
        + 0.15 * s_down
        + 0.15 * s_lift
        + 0.10 * s_conf
    ).round(2)

    df = df.sort_values("prs_score", ascending=False).reset_index(drop=True)
    return df, {
        "k_eb_bowler": round(k_eb, 2),
        "within_var_bowler_vap": round(within_var_vap, 2),
        "beta_prior_alpha": a0,
        "beta_prior_beta": b0,
    }
