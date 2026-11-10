import numpy as np
import pandas as pd

# Fixed pre-match historical venue par adjustments (never computed from the current match's outcomes)
HISTORICAL_VENUE_PAR_LOOKUP = {
    6: 181.0,     # Sawai Mansingh Stadium, Jaipur
    1241: 186.0,  # Barsapara Cricket Stadium, Guwahati
    12: 184.0,    # Wankhede / Maharashtra high-scoring IPL venue
    3: 179.0,
    46: 175.0,
    49: 174.0,
}

def compute_pressure_index(bbb: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a strictly pre-delivery Situational Pressure Index P_i in [0, 1]
    separating First Innings (Setting) from Second Innings (Chasing), and keeping
    Competition Stakes (IPL / Knockout) separate from match-state pressure.
    Strictly causal: delivery t depends ONLY on deliveries 1..(t-1) in the match.
    """
    df = bbb.copy()

    # 1. Pre-delivery cumulative runs, wickets, and legal balls within (match_id, innings)
    grp = df.groupby(["match_id", "innings"], sort=False)
    df["cum_runs_after"] = grp["total_runs"].cumsum()
    df["cum_runs_before"] = df["cum_runs_after"] - df["total_runs"]

    df["is_wkt_int"] = df["is_wicket"].fillna(False).astype(int)
    df["cum_wkts_after"] = grp["is_wkt_int"].cumsum()
    df["cum_wkts_before"] = (df["cum_wkts_after"] - df["is_wkt_int"]).clip(0, 9)

    df["legal_balls_after"] = grp["is_legal_ball"].cumsum()
    df["legal_balls_before"] = (df["legal_balls_after"] - df["is_legal_ball"]).clip(0, 120)

    # Max scheduled legal balls in innings
    max_overs = np.where(
        df["innings"] == 1,
        df["first_inn_revised_overs"].fillna(20.0),
        np.where(df["innings"] == 2, df["second_inn_revised_overs"].fillna(20.0), 1.0)
    )
    df["max_balls"] = (max_overs * 6.0).clip(6.0, 120.0)
    df["balls_rem_before"] = (df["max_balls"] - df["legal_balls_before"]).clip(1.0, 120.0)
    df["wkts_rem_before"] = (10.0 - df["cum_wkts_before"]).clip(1.0, 10.0)

    # First innings total map ONLY used for 2nd innings (after Innings 1 is already complete!)
    inn1_totals = df[df["innings"] == 1].groupby("match_id")["total_runs"].sum().to_dict()
    df["inn1_total"] = np.where(df["innings"] >= 2, df["match_id"].map(inn1_totals).fillna(165.0), np.nan)
    df["target_runs"] = np.where(
        df["innings"] >= 2,
        np.where(
            df["revised_target"].notna() & (df["revised_target"] > 0),
            df["revised_target"],
            df["inn1_total"] + 1.0
        ),
        np.nan
    )

    # Strictly pre-match par total for 1st innings (zero leakage from current match innings 1 outcome)
    format_par = pd.Series(np.where(df["is_ipl"] == 1, 178.0, 162.0), index=df.index)
    venue_par = df["venue_id"].map(HISTORICAL_VENUE_PAR_LOOKUP).fillna(format_par)
    df["par_total_1st"] = 0.60 * format_par + 0.40 * venue_par

    # DLS-inspired resource percentage remaining before delivery: Z in (0, 1]
    balls_frac = (df["balls_rem_before"] / df["max_balls"]).clip(0.01, 1.0)
    wkts_frac = (df["wkts_rem_before"] / 10.0).clip(0.10, 1.0)
    df["resources_rem"] = (balls_frac ** 0.85) * (wkts_frac ** 0.55)
    prog_frac = (df["legal_balls_before"] / df["max_balls"]).clip(0.0, 1.0)

    # --- COMPONENT A: Required Rate / Par Trajectory Pressure (P_rate) ---
    req_runs = (df["target_runs"].fillna(0.0) - df["cum_runs_before"]).clip(0.0, None)
    df["req_runs_before"] = np.where(df["innings"] == 2, req_runs, np.nan)
    rrr = 6.0 * req_runs / df["balls_rem_before"]
    df["rrr_before"] = np.where(df["innings"] == 2, rrr, np.nan)
    crr = 6.0 * df["cum_runs_before"] / np.maximum(1.0, df["legal_balls_before"])
    df["crr_before"] = crr

    rrr_logistic = 1.0 / (1.0 + np.exp(-0.42 * (rrr - 9.2)))
    accel_logistic = 1.0 / (1.0 + np.exp(-0.30 * ((rrr - crr) - 1.0)))
    p_rate_inn2 = np.where(req_runs <= 0, 0.0, 0.65 * rrr_logistic + 0.35 * accel_logistic)

    # 1st Innings (Setting): scales in as the innings progresses so ball 1 at 0/0 has low rate pressure
    par_score_now = df["par_total_1st"] * (prog_frac ** 1.08)
    par_deficit = par_score_now - df["cum_runs_before"]
    rem_runs_to_par = (df["par_total_1st"] - df["cum_runs_before"]).clip(0.0, None)
    par_rrr_1st = 6.0 * rem_runs_to_par / df["balls_rem_before"]
    raw_p_rate_1 = (
        0.55 * (1.0 / (1.0 + np.exp(-0.14 * (par_deficit - 2.0))))
        + 0.45 * (1.0 / (1.0 + np.exp(-0.38 * (par_rrr_1st - 9.4))))
    )
    p_rate_inn1 = raw_p_rate_1 * (0.25 + 0.75 * np.minimum(1.0, prog_frac * 2.2))

    df["p_rate"] = np.where(
        df["innings"] > 2, 1.0,
        np.where(df["innings"] == 2, p_rate_inn2, p_rate_inn1)
    ).clip(0.0, 1.0)

    # --- COMPONENT B: Wicket Resource Scarcity Pressure (P_wkt) ---
    expected_wkts_lost = 7.5 * (prog_frac ** 1.15)
    wkt_excess = (df["cum_wkts_before"] - expected_wkts_lost).clip(-3.0, 8.0)
    p_wkt_scarcity = (df["cum_wkts_before"] / 7.0).clip(0.0, 1.0)
    p_wkt_excess = np.where(
        df["cum_wkts_before"] == 0, 0.05,
        1.0 / (1.0 + np.exp(-0.85 * (wkt_excess - 0.5)))
    )
    df["p_wkt"] = np.where(
        df["innings"] > 2, 1.0,
        0.55 * p_wkt_scarcity + 0.45 * p_wkt_excess
    ).clip(0.0, 1.0)

    # --- COMPONENT C: Innings Stage Leverage (P_stage) ---
    over_num = df["overid"].clip(1, 20)
    base_stage = np.where(
        df["innings"] > 2, 1.0,
        np.where(
            over_num <= 6,
            0.22 + 0.03 * over_num,                   # Powerplay: 0.25 - 0.40
            np.where(
                over_num <= 15,
                0.35 + 0.025 * (over_num - 6),        # Middle: 0.375 - 0.575
                0.66 + 0.065 * (over_num - 15)        # Death: 0.725 - 0.985
            )
        )
    )
    df["p_stage"] = base_stage.clip(0.0, 1.0)

    # --- COMPONENT D: Match Closeness / Knife-Edge Leverage (P_close) ---
    inn2_live_gaussian = np.exp(-0.5 * ((rrr - 11.2) / 3.4) ** 2)
    inn2_late_multiplier = 0.55 + 0.45 * prog_frac
    p_close_inn2 = np.where(
        (req_runs <= 2) & (df["balls_rem_before"] >= 12), 0.10,
        np.where(rrr > 26.0, 0.25, inn2_live_gaussian * inn2_late_multiplier)
    )
    inn1_par_gaussian = np.exp(-0.5 * ((par_deficit - 2.0) / 16.0) ** 2)
    p_close_inn1 = inn1_par_gaussian * (0.25 + 0.75 * prog_frac)

    df["p_close"] = np.where(
        df["innings"] > 2, 1.0,
        np.where(df["innings"] == 2, p_close_inn2, p_close_inn1)
    ).clip(0.0, 1.0)

    # --- COMPONENT E: Recent Pre-Delivery Momentum Shock (P_mom) ---
    df["is_dot_ball"] = (df["total_runs"] == 0).astype(int)
    wkts_prev_12 = grp["is_wkt_int"].transform(
        lambda s: s.shift(1, fill_value=0).rolling(12, min_periods=1).sum()
    )
    dots_prev_6 = grp["is_dot_ball"].transform(
        lambda s: s.shift(1, fill_value=0).rolling(6, min_periods=1).sum()
    )
    df["wkts_prev_12_before"] = wkts_prev_12
    df["dots_prev_6_before"] = dots_prev_6

    p_wkt_shock = np.where(wkts_prev_12 >= 2, 1.0, np.where(wkts_prev_12 == 1, 0.58, 0.05))
    p_dot_squeeze = np.clip((dots_prev_6 - 1.0) / 4.0, 0.0, 1.0)
    df["p_mom"] = np.where(
        df["innings"] > 2, 1.0,
        0.65 * p_wkt_shock + 0.35 * p_dot_squeeze
    ).clip(0.0, 1.0)

    # --- AUTHORITATIVE SITUATIONAL PRESSURE INDEX P_i in [0, 1] ---
    df["pressure_index"] = (
        0.30 * df["p_rate"]
        + 0.22 * df["p_wkt"]
        + 0.18 * df["p_stage"]
        + 0.15 * df["p_close"]
        + 0.15 * df["p_mom"]
    ).clip(0.0, 1.0)

    df["match_stakes_weight"] = 1.0 + 0.15 * df["is_ipl"] + 0.10 * df["is_knockout"]

    df["is_pressure_050"] = (df["pressure_index"] >= 0.50).astype(int)
    df["is_pressure_055"] = (df["pressure_index"] >= 0.55).astype(int)
    df["is_pressure_058"] = (df["pressure_index"] >= 0.58).astype(int)
    df["is_pressure_065"] = (df["pressure_index"] >= 0.65).astype(int)
    df["is_high_pressure"] = df["is_pressure_055"]

    df["is_clutch_chase_end"] = (
        (df["innings"] == 2) & (df["overid"] >= 19) & (df["req_runs_before"] > 0) & (df["req_runs_before"] <= 36)
    ).astype(int)
    df["is_clutch_collapse"] = ((df["wkts_prev_12_before"] >= 2) & (df["cum_wkts_before"] >= 3)).astype(int)
    df["is_clutch_tail"] = ((df["cum_wkts_before"] >= 6) & (df["pressure_index"] >= 0.58)).astype(int)
    df["is_clutch_knockout"] = ((df["is_knockout"] == 1) & (df["pressure_index"] >= 0.58)).astype(int)
    df["is_clutch_moment"] = (
        (df["innings"] > 2)
        | (df["is_clutch_chase_end"] == 1)
        | (df["is_clutch_collapse"] == 1)
        | (df["is_clutch_tail"] == 1)
        | (df["is_clutch_knockout"] == 1)
    ).astype(int)

    return df

def summarize_pressure_distributions(df: pd.DataFrame) -> dict:
    counts, edges = np.histogram(df["pressure_index"], bins=np.linspace(0.0, 1.0, 11))
    histogram = [
        {"bin": f"{edges[i]:.1f}–{edges[i+1]:.1f}", "count": int(counts[i]), "pct": round(100.0 * counts[i] / len(df), 2)}
        for i in range(len(counts))
    ]

    by_innings = []
    for inn, g in df[df["innings"].isin([1, 2])].groupby("innings"):
        by_innings.append({
            "innings": f"Innings {int(inn)} ({'Setting' if inn == 1 else 'Chasing'})",
            "balls": int(len(g)),
            "mean_p": round(float(g["pressure_index"].mean()), 3),
            "median_p": round(float(g["pressure_index"].median()), 3),
            "p90": round(float(g["pressure_index"].quantile(0.90)), 3),
            "pct_ge_050": round(100.0 * float(g["is_pressure_050"].mean()), 1),
            "pct_ge_055": round(100.0 * float(g["is_pressure_055"].mean()), 1),
            "pct_ge_058": round(100.0 * float(g["is_pressure_058"].mean()), 1),
            "pct_ge_065": round(100.0 * float(g["is_pressure_065"].mean()), 1),
        })

    by_comp = []
    for is_ipl, g in df.groupby("is_ipl"):
        by_comp.append({
            "competition": "IPL (2022–2026)" if is_ipl == 1 else "International T20I & Emerging",
            "balls": int(len(g)),
            "mean_p": round(float(g["pressure_index"].mean()), 3),
            "pct_ge_055": round(100.0 * float(g["is_pressure_055"].mean()), 1),
            "pct_ge_058": round(100.0 * float(g["is_pressure_058"].mean()), 1),
        })

    by_phase = []
    for ph in ["Powerplay", "Middle", "Death"]:
        g = df[df["phase"] == ph]
        by_phase.append({
            "phase": ph,
            "balls": int(len(g)),
            "mean_p": round(float(g["pressure_index"].mean()), 3),
            "pct_ge_050": round(100.0 * float(g["is_pressure_050"].mean()), 1),
            "pct_ge_055": round(100.0 * float(g["is_pressure_055"].mean()), 1),
            "pct_ge_058": round(100.0 * float(g["is_pressure_058"].mean()), 1),
            "pct_ge_065": round(100.0 * float(g["is_pressure_065"].mean()), 1),
        })

    thresholds = []
    for th, col, label in [
        (0.50, "is_pressure_050", "Broad Situational Tension (P >= 0.50)"),
        (0.55, "is_pressure_055", "Validated Primary Pressure (P >= 0.55)"),
        (0.58, "is_pressure_058", "Strict High-Leverage Pressure (P >= 0.58)"),
        (0.65, "is_pressure_065", "Extreme Crisis / Death Leverage (P >= 0.65)"),
    ]:
        sub = df[df[col] == 1]
        thresholds.append({
            "threshold": th,
            "label": label,
            "balls": int(len(sub)),
            "pct_of_dataset": round(100.0 * len(sub) / len(df), 2),
            "mean_sr": round(100.0 * float(sub["batter_runs"].sum() / max(1, sub["is_ball_faced"].sum())), 2),
            "wkt_rate_per_100b": round(100.0 * float(sub["is_wicket"].fillna(False).mean()), 2),
        })

    low_row = df[(df["is_ipl"] == 1) & (df["pressure_index"] >= 0.16) & (df["pressure_index"] <= 0.24)].iloc[0]
    med_row = df[(df["is_ipl"] == 1) & (df["pressure_index"] >= 0.54) & (df["pressure_index"] <= 0.56)].iloc[0]
    high_row = df[(df["is_ipl"] == 1) & (df["innings"] == 2) & (df["pressure_index"] >= 0.85)].iloc[0]

    def fmt_example(r, tier):
        return {
            "tier": tier,
            "match_id": int(r["match_id"]),
            "comp": str(r["comp_name"]),
            "match": f"{r['batting_team_name']} vs {r['bowling_team_name']}",
            "innings": int(r["innings"]),
            "over_ball": f"{int(r['overid'])}.{int(r['over_ball_num'])}",
            "striker": str(r["striker_name"]),
            "bowler": str(r["bowler_name"]),
            "score_before": f"{int(r['cum_runs_before'])}/{int(r['cum_wkts_before'])}",
            "rrr_or_par": f"Req {int(r['req_runs_before'])} off {int(r['balls_rem_before'])}b (RRR {r['rrr_before']:.1f})" if r["innings"] == 2 else f"1st Inn Score {int(r['cum_runs_before'])}/{int(r['cum_wkts_before'])} (Par {r['par_total_1st']:.0f})",
            "p_index": round(float(r["pressure_index"]), 3),
            "components": f"Rate={r['p_rate']:.2f}, Wkt={r['p_wkt']:.2f}, Stage={r['p_stage']:.2f}, Close={r['p_close']:.2f}, Mom={r['p_mom']:.2f}"
        }

    return {
        "histogram": histogram,
        "by_innings": by_innings,
        "by_comp": by_comp,
        "by_phase": by_phase,
        "thresholds": thresholds,
        "examples": [
            fmt_example(low_row, "Low Pressure (Platform Routine)"),
            fmt_example(med_row, "Medium/Threshold Pressure (P ~ 0.55)"),
            fmt_example(high_row, "High/Extreme Pressure (Crunch Chase)"),
        ]
    }
