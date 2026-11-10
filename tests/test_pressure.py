import pandas as pd
from pressure_engine import compute_pressure_index

def run_test_pressure(bbb):
    # Test 5: Pressure Calculation on synthetic low vs extreme situations
    low_balls = bbb[(bbb["innings"] == 1) & (bbb["overid"] == 1) & (bbb["cum_wkts_before"] == 0)]
    high_balls = bbb[(bbb["innings"] == 2) & (bbb["overid"] >= 19) & (bbb["rrr_before"] >= 12.0) & (bbb["rrr_before"] <= 16.0) & (bbb["cum_wkts_before"] >= 5)]
    mean_low = float(low_balls["pressure_index"].mean())
    mean_high = float(high_balls["pressure_index"].mean())
    passed_t5 = (mean_low < 0.35) and (mean_high > 0.72)

    # Test 6: Zero Future Information Leakage Test
    # Take match 13512 and mutate the outcome of ball 50 (runs=6, is_wicket=True) and verify ball 50's pre-delivery pressure_index is identical to 1e-12!
    sample_match = bbb[bbb["match_id"] == 13512].copy().reset_index(drop=True)
    p_orig = compute_pressure_index(sample_match)["pressure_index"].values
    mutated = sample_match.copy()
    mutated.loc[50, "batter_runs"] = 6
    mutated.loc[50, "total_runs"] = 6
    mutated.loc[50, "is_wicket"] = True
    mutated.loc[50:, "batter_runs"] = 6  # mutate ball 50 and all subsequent balls
    mutated.loc[50:, "total_runs"] = 6
    p_mut = compute_pressure_index(mutated)["pressure_index"].values
    max_diff_up_to_50 = float(abs(p_orig[:51] - p_mut[:51]).max())
    passed_t6 = max_diff_up_to_50 < 1e-12

    return [
        {
            "id": "TEST-05",
            "name": "Situational Pressure Index Calibration (1st vs 2nd Innings)",
            "status": "PASS" if passed_t5 else "FAIL",
            "evidence": f"Over 1 (0 wkts down) Mean P = {mean_low:.3f} vs Over 19–20 Chase (RRR 12–16, 5+ wkts down) Mean P = {mean_high:.3f}."
        },
        {
            "id": "TEST-06",
            "name": "Strict Pre-Delivery Zero-Leakage Invariance",
            "status": "PASS" if passed_t6 else "FAIL",
            "evidence": f"Mutating delivery t=50 and all future deliveries t>50 in Match 13512 produced max |Delta P_t| = {max_diff_up_to_50:.2e} across all deliveries t <= 50."
        }
    ]
