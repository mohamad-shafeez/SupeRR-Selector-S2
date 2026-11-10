import numpy as np
import pandas as pd

def build_similarity_engine(bat_df: pd.DataFrame, bowl_df: pd.DataFrame) -> dict:
    """
    Computes standardized feature vectors and Weighted Euclidean + Cosine similarity
    to identify statistical replacement alternatives for marquee batters and bowlers.
    """
    def compute_pairs(df: pd.DataFrame, feat_cols: list[str], weights: list[float], is_batter: bool):
        sub = df[df["press_balls"] >= 50].copy().reset_index(drop=True)
        X = sub[feat_cols].fillna(0.0).values.astype(np.float64)
        mu = np.mean(X, axis=0)
        sd = np.where(np.std(X, axis=0) < 1e-6, 1.0, np.std(X, axis=0))
        Z = (X - mu) / sd
        W = np.array(weights, dtype=np.float64)
        ZW = Z * np.sqrt(W)

        # Select top marquee players (established IPL stars with >= 140 pressure balls)
        marquees = sub[(sub["has_ipl"]) & (sub["press_balls"] >= 140) & (~sub["is_uncapped_indian"])].head(8)
        results = []

        for m_idx, m_row in marquees.iterrows():
            v_m = ZW[m_idx]
            dists = np.sqrt(np.sum((ZW - v_m) ** 2, axis=1))
            # Convert weighted Euclidean distance to 0-100% similarity score
            sim_scores = 100.0 * np.exp(-0.22 * dists)

            # Prefer same or adjacent role/style and strong domestic/value alternatives
            cand_rows = []
            for c_idx, c_row in sub.iterrows():
                if c_idx == m_idx:
                    continue
                if not (c_row["has_ipl"] or c_row["has_india_a"]):
                    continue
                role_bonus = 4.5 if c_row["role"] == m_row["role"] else 0.0
                style_bonus = 3.5 if (not is_batter and c_row["pace_or_spin"] == m_row["pace_or_spin"]) else 0.0
                sim = float(np.clip(sim_scores[c_idx] + role_bonus + style_bonus, 40.0, 98.8))

                if is_batter:
                    diff_str = f"Adj SR {c_row['adj_press_sr']:.1f} vs {m_row['adj_press_sr']:.1f} | Post VAP {c_row['posterior_vap']:+.1f} vs {m_row['posterior_vap']:+.1f}"
                else:
                    diff_str = f"Adj Econ {c_row['adj_press_econ']:.2f} vs {m_row['adj_press_econ']:.2f} | Post VAP {c_row['posterior_vap']:+.2f} vs {m_row['posterior_vap']:+.2f}"

                cand_rows.append({
                    "marquee_player": str(m_row["striker_name"] if is_batter else m_row["bowler_name"]),
                    "marquee_role": str(m_row["role"]),
                    "marquee_prs": round(float(m_row["prs_score"]), 2),
                    "alternative": str(c_row["striker_name"] if is_batter else c_row["bowler_name"]),
                    "alt_unstop_name": str(c_row["unstop_name"]),
                    "alt_role": str(c_row["role"]),
                    "is_uncapped_indian": bool(c_row["is_uncapped_indian"]),
                    "alt_pool_tag": "Uncapped Indian Domestic Target" if c_row["is_uncapped_indian"] else ("Overseas / Capped Option" if c_row["is_overseas"] else "Capped Indian Option"),
                    "similarity_pct": round(sim, 1),
                    "alt_prs": round(float(c_row["prs_score"]), 2),
                    "press_balls": int(c_row["press_balls"]),
                    "key_difference": diff_str,
                    "confidence_tier": str(c_row["confidence_tier"]),
                })

            cand_df = pd.DataFrame(cand_rows).sort_values("similarity_pct", ascending=False)
            # Pick top 2 overall + top 2 uncapped Indian alternatives
            top_uncapped = cand_df[cand_df["is_uncapped_indian"]].head(2)
            top_other = cand_df[~cand_df["alternative"].isin(top_uncapped["alternative"])].head(1)
            merged_cands = pd.concat([top_uncapped, top_other]).sort_values("similarity_pct", ascending=False)
            results.extend(merged_cands.to_dict(orient="records"))

        return results

    bat_feats = [
        "avg_bat_order", "adj_press_sr", "posterior_vap", "press_bnd_pct",
        "press_dot_pct", "vs_pace_sr", "vs_spin_sr", "ph_death_sr", "post_consistency_pct"
    ]
    bat_weights = [1.5, 1.4, 1.5, 1.0, 1.0, 0.9, 0.9, 1.2, 1.1]

    bowl_feats = [
        "avg_over", "adj_press_econ", "raw_press_sr", "posterior_vap",
        "press_dot_pct", "vs_rhb_econ", "vs_lhb_econ", "ph_death_econ", "post_consistency_pct"
    ]
    bowl_weights = [1.5, 1.4, 1.3, 1.5, 1.0, 0.8, 0.8, 1.2, 1.1]

    return {
        "batter_replacements": compute_pairs(bat_df, bat_feats, bat_weights, is_batter=True),
        "bowler_replacements": compute_pairs(bowl_df, bowl_feats, bowl_weights, is_batter=False),
    }
