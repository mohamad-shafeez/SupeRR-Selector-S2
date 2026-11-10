import numpy as np
import pandas as pd

def run_sensitivity_and_bootstrap(
    bbb: pd.DataFrame,
    bat_df: pd.DataFrame,
    bowl_df: pd.DataFrame,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    1. Evaluates a 3 x 4 x 3 = 36 configuration sensitivity grid:
       - Thresholds: [0.50, 0.58, 0.65]
       - Min pressure balls: [50, 100, 150, 200]
       - Shrinkage & weight regimes:
         * R1: Low Shrinkage (k=60), Impact-Heavy (55% VAP, 20% Cons, 15% Lift, 10% Downside)
         * R2: Empirical Bayes Default (k=95), Balanced PRS (35% VAP, 25% Cons, 15% Downside, 15% Lift, 10% Conf)
         * R3: High Shrinkage (k=150), Consistency-Heavy (25% VAP, 40% Cons, 20% Downside, 15% Conf)
    2. Performs 500-iteration Match-Cluster Bootstrap resampling at the match_id level (seed=42).
    """
    rng = np.random.default_rng(seed)

    thresholds = [
        (0.50, "is_pressure_050"),
        (0.58, "is_pressure_058"),
        (0.65, "is_pressure_065"),
    ]
    min_balls_list = [50, 100, 150, 200]
    regimes = [
        ("R1_Impact_k60", 60.0, (0.55, 0.20, 0.10, 0.15, 0.00)),
        ("R2_Balanced_EB", 95.0, (0.35, 0.25, 0.15, 0.15, 0.10)),
        ("R3_Conservative_k150", 150.0, (0.25, 0.40, 0.20, 0.00, 0.15)),
    ]

    # --- Pre-aggregate by threshold for fast 36-grid evaluation ---
    bat_grid_top10 = {pid: 0 for pid in bat_df["player_id"]}
    bat_grid_elig = {pid: 0 for pid in bat_df["player_id"]}
    bowl_grid_top10 = {pid: 0 for pid in bowl_df["player_id"]}
    bowl_grid_elig = {pid: 0 for pid in bowl_df["player_id"]}

    grid_summary_rows = []
    bf = bbb[bbb["is_ball_faced"] == 1].copy()
    bf["w_stakes"] = 1.0 + 0.20 * bf["is_ipl"] + 0.10 * bf["is_knockout"]

    for th_val, p_col in thresholds:
        pr_bat = bf[bf[p_col] == 1]
        b_agg = pr_bat.groupby("striker_id").agg(
            p_balls=("is_ball_faced", "sum"),
            p_runs=("batter_runs", "sum"),
            w_vap=("bat_vap_ball", lambda s: float((s * pr_bat.loc[s.index, "w_stakes"]).sum())),
        ).reset_index().rename(columns={"striker_id": "player_id"})
        b_sub = bat_df[["player_id", "unstop_name", "has_ipl", "is_uncapped_indian", "role_prior_sr", "role_prior_vap", "post_consistency_pct", "p10_downside_vap", "raw_sr_delta"]].merge(b_agg, on="player_id", how="inner")

        pr_bowl = bbb[bbb[p_col] == 1]
        bw_agg = pr_bowl.groupby("bowler_id").agg(
            p_balls=("is_legal_ball", "sum"),
            p_runs=("bowler_runs", "sum"),
            w_vap=("bowl_vap_ball", lambda s: float((s * (1.0 + 0.20 * pr_bowl.loc[s.index, "is_ipl"] + 0.10 * pr_bowl.loc[s.index, "is_knockout"])).sum())),
        ).reset_index().rename(columns={"bowler_id": "player_id"})
        bw_sub = bowl_df[["player_id", "unstop_name", "has_ipl", "is_uncapped_indian", "role_prior_econ", "role_prior_vap", "post_consistency_pct", "p10_downside_vap", "raw_econ_delta"]].merge(bw_agg, on="player_id", how="inner")

        for min_b in min_balls_list:
            for reg_name, k_val, (w_vap, w_cons, w_down, w_lift, w_conf) in regimes:
                # Batter grid evaluation (on IPL-eligible pool)
                bc = b_sub[(b_sub["p_balls"] >= min_b) & (b_sub["has_ipl"])].copy()
                if len(bc) >= 10:
                    sw = bc["p_balls"] / (bc["p_balls"] + k_val)
                    raw_vap_100 = 100.0 * bc["w_vap"] / bc["p_balls"]
                    post_vap = sw * raw_vap_100 + (1.0 - sw) * bc["role_prior_vap"]
                    s1 = post_vap.rank(pct=True) * 100.0
                    s2 = bc["post_consistency_pct"].rank(pct=True) * 100.0
                    s3 = bc["p10_downside_vap"].rank(pct=True) * 100.0
                    s4 = bc["raw_sr_delta"].rank(pct=True) * 100.0
                    s5 = 100.0 * (bc["p_balls"] / (bc["p_balls"] + 75.0))
                    bc["score"] = w_vap * s1 + w_cons * s2 + w_down * s3 + w_lift * s4 + w_conf * s5
                    bc = bc.sort_values("score", ascending=False)
                    top10_pids = set(bc.head(10)["player_id"])
                    for pid in bc["player_id"]:
                        bat_grid_elig[pid] = bat_grid_elig.get(pid, 0) + 1
                        if pid in top10_pids:
                            bat_grid_top10[pid] = bat_grid_top10.get(pid, 0) + 1
                    top1_bat = bc.iloc[0]["unstop_name"]
                else:
                    top1_bat = "N/A"

                # Bowler grid evaluation (on IPL-eligible pool)
                bwc = bw_sub[(bw_sub["p_balls"] >= min_b) & (bw_sub["has_ipl"])].copy()
                if len(bwc) >= 10:
                    sw = bwc["p_balls"] / (bwc["p_balls"] + k_val)
                    raw_vap_24 = 24.0 * bwc["w_vap"] / bwc["p_balls"]
                    post_vap = sw * raw_vap_24 + (1.0 - sw) * bwc["role_prior_vap"]
                    s1 = post_vap.rank(pct=True) * 100.0
                    s2 = bwc["post_consistency_pct"].rank(pct=True) * 100.0
                    s3 = bwc["p10_downside_vap"].rank(pct=True) * 100.0
                    s4 = (-bwc["raw_econ_delta"]).rank(pct=True) * 100.0
                    s5 = 100.0 * (bwc["p_balls"] / (bwc["p_balls"] + 80.0))
                    bwc["score"] = w_vap * s1 + w_cons * s2 + w_down * s3 + w_lift * s4 + w_conf * s5
                    bwc = bwc.sort_values("score", ascending=False)
                    top10_bw = set(bwc.head(10)["player_id"])
                    for pid in bwc["player_id"]:
                        bowl_grid_elig[pid] = bowl_grid_elig.get(pid, 0) + 1
                        if pid in top10_bw:
                            bowl_grid_top10[pid] = bowl_grid_top10.get(pid, 0) + 1
                    top1_bowl = bwc.iloc[0]["unstop_name"]
                else:
                    top1_bowl = "N/A"

                grid_summary_rows.append({
                    "threshold": th_val,
                    "min_balls": min_b,
                    "regime": reg_name,
                    "top1_batter": top1_bat,
                    "top1_bowler": top1_bowl,
                })

    # --- 2. Match-Cluster Bootstrap (500 resamples of the 883 matches, seed=42) ---
    all_matches = np.sort(bbb["match_id"].unique())
    n_m = len(all_matches)
    match_idx_map = {m: i for i, m in enumerate(all_matches)}

    # Generate (n_m, n_boot) multinomial match weights
    boot_weights = rng.multinomial(n_m, np.full(n_m, 1.0 / n_m), size=n_boot).T.astype(np.float64)

    def run_cluster_boot(player_df: pd.DataFrame, is_batter: bool, min_balls_cut: int = 100):
        # Filter to IPL-eligible cohort with >= 60 pressure balls (and evaluate rank among >= min_balls_cut)
        cohort = player_df[(player_df["has_ipl"]) & (player_df["press_balls"] >= min_balls_cut)].copy().reset_index(drop=True)
        pids = cohort["player_id"].values
        pid_to_row = {p: i for i, p in enumerate(pids)}
        n_p = len(pids)

        pr_data = bbb[(bbb["is_pressure_055"] == 1) & (bbb["is_ball_faced"] == 1 if is_batter else bbb["is_legal_ball"] == 1)].copy()
        id_col = "striker_id" if is_batter else "bowler_id"
        vap_col = "bat_vap_ball" if is_batter else "bowl_vap_ball"
        scale = 100.0 if is_batter else 24.0
        k_eb = 92.0 if is_batter else 102.0

        pr_data = pr_data[pr_data[id_col].isin(pid_to_row)].copy()
        pr_data["w_vap"] = pr_data[vap_col] * (1.0 + 0.20 * pr_data["is_ipl"] + 0.10 * pr_data["is_knockout"])

        pm = pr_data.groupby([id_col, "match_id"]).agg(
            balls=(id_col, "count"),
            w_vap=("w_vap", "sum"),
            raw_vap=(vap_col, "sum"),
        ).reset_index()

        B_mat = np.zeros((n_p, n_m), dtype=np.float64)
        V_mat = np.zeros((n_p, n_m), dtype=np.float64)
        S_mat = np.zeros((n_p, n_m), dtype=np.float64)
        M_mat = np.zeros((n_p, n_m), dtype=np.float64)

        for _, r in pm.iterrows():
            pi = pid_to_row[int(r[id_col])]
            mi = match_idx_map[int(r["match_id"])]
            b_cnt = float(r["balls"])
            B_mat[pi, mi] = b_cnt
            V_mat[pi, mi] = float(r["w_vap"])
            if b_cnt >= (3 if is_batter else 4):
                M_mat[pi, mi] = 1.0
                if float(r["raw_vap"]) > 0:
                    S_mat[pi, mi] = 1.0

        # Matrix multiply across 500 bootstrap match weight vectors: (n_p, n_boot)
        boot_B = np.maximum(1.0, B_mat @ boot_weights)
        boot_V = V_mat @ boot_weights
        boot_S = S_mat @ boot_weights
        boot_M = np.maximum(1.0, M_mat @ boot_weights)

        prior_vap = cohort["role_prior_vap"].values[:, None]
        sw = boot_B / (boot_B + k_eb)
        boot_post_vap = sw * (scale * boot_V / boot_B) + (1.0 - sw) * prior_vap
        boot_post_cons = 100.0 * (boot_S + 4.3) / (boot_M + 7.5)

        # Static auxiliary components across resamples (downside & lift rank)
        aux_score = (
            0.15 * (cohort["p10_downside_vap"].rank(pct=True).values[:, None] * 100.0)
            + 0.15 * (cohort["role_adjusted_lift"].rank(pct=True).values[:, None] * 100.0)
            + 0.10 * (100.0 * (boot_B / (boot_B + 75.0)))
        )

        # Compute rank in each of the 500 bootstrap iterations (1 = best)
        ranks = np.zeros((n_p, n_boot), dtype=np.float64)
        for b_idx in range(n_boot):
            v_pct = pd.Series(boot_post_vap[:, b_idx]).rank(pct=True).values * 100.0
            c_pct = pd.Series(boot_post_cons[:, b_idx]).rank(pct=True).values * 100.0
            comp = 0.35 * v_pct + 0.25 * c_pct + aux_score[:, b_idx]
            # 1 = highest score
            order = np.argsort(-comp)
            r_vec = np.empty_like(order, dtype=np.float64)
            r_vec[order] = np.arange(1, n_p + 1)
            ranks[:, b_idx] = r_vec

        res = {}
        for i, pid in enumerate(pids):
            r_row = ranks[i, :]
            res[int(pid)] = {
                "boot_top10_pct": round(100.0 * float(np.mean(r_row <= 10)), 1),
                "boot_median_rank": round(float(np.median(r_row)), 1),
                "boot_p10_rank": round(float(np.percentile(r_row, 10)), 1),
                "boot_p90_rank": round(float(np.percentile(r_row, 90)), 1),
                "boot_mean_rank": round(float(np.mean(r_row)), 2),
                "boot_rank_sd": round(float(np.std(r_row, ddof=1)), 2),
            }
        return res

    bat_boot_a = run_cluster_boot(bat_df, is_batter=True, min_balls_cut=100)
    bowl_boot_a = run_cluster_boot(bowl_df, is_batter=False, min_balls_cut=100)

    # Also run bootstrap on Uncapped Indian pool (min_balls_cut=55) for Leaderboard B stability
    bat_uncapped = bat_df[bat_df["is_uncapped_indian"]].copy()
    bowl_uncapped = bowl_df[bowl_df["is_uncapped_indian"]].copy()
    bat_boot_b = run_cluster_boot(bat_uncapped, is_batter=True, min_balls_cut=55)
    bowl_boot_b = run_cluster_boot(bowl_uncapped, is_batter=False, min_balls_cut=55)

    def attach_stability(df: pd.DataFrame, boot_a: dict, boot_b: dict, grid_top10: dict, grid_elig: dict) -> pd.DataFrame:
        out = df.copy()
        top10_p, med_r, p10_r, p90_r, mean_r, sd_r, g_pct, tier = [], [], [], [], [], [], [], []
        for _, r in out.iterrows():
            pid = int(r["player_id"])
            b_info = boot_a.get(pid) or boot_b.get(pid) or {
                "boot_top10_pct": 0.0, "boot_median_rank": 25.0,
                "boot_p10_rank": 15.0, "boot_p90_rank": 35.0,
                "boot_mean_rank": 25.0, "boot_rank_sd": 5.0
            }
            ge = max(1, grid_elig.get(pid, 0))
            gt = round(100.0 * grid_top10.get(pid, 0) / ge, 1) if grid_elig.get(pid, 0) > 0 else b_info["boot_top10_pct"]

            b10 = b_info["boot_top10_pct"]
            pb = int(r["press_balls"])
            if pb < 75:
                st = "Emerging Candidate"
            elif b10 >= 78.0 and gt >= 70.0:
                st = "Elite Stable Pick"
            elif b10 >= 48.0 or gt >= 50.0:
                st = "Strong Candidate"
            elif b10 >= 15.0 or gt >= 20.0:
                st = "Borderline Candidate"
            else:
                st = "Fragile Outlier" if pb < 130 else "Depth Contender"

            top10_p.append(b10)
            med_r.append(b_info["boot_median_rank"])
            p10_r.append(b_info["boot_p10_rank"])
            p90_r.append(b_info["boot_p90_rank"])
            mean_r.append(b_info["boot_mean_rank"])
            sd_r.append(b_info["boot_rank_sd"])
            g_pct.append(gt)
            tier.append(st)

        out["boot_top10_pct"] = top10_p
        out["boot_median_rank"] = med_r
        out["boot_p10_rank"] = p10_r
        out["boot_p90_rank"] = p90_r
        out["boot_mean_rank"] = mean_r
        out["boot_rank_sd"] = sd_r
        out["grid_top10_pct"] = g_pct
        out["stability_tier"] = tier
        return out

    bat_out = attach_stability(bat_df, bat_boot_a, bat_boot_b, bat_grid_top10, bat_grid_elig)
    bowl_out = attach_stability(bowl_df, bowl_boot_a, bowl_boot_b, bowl_grid_top10, bowl_grid_elig)

    return bat_out, bowl_out, {
        "n_grid_scenarios": len(grid_summary_rows),
        "n_bootstrap_iterations": n_boot,
        "random_seed": seed,
        "grid_scenarios_sample": grid_summary_rows[:18],
    }
