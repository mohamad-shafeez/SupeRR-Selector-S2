import hashlib
import json
import os
import re
import numpy as np
import pandas as pd

# True duplicate player_ids representing the exact same real-world cricketer in the dataset
TRUE_DUPLICATE_ID_MAP = {
    39479: 29833,  # Bashir Ahmad (Afghanistan / Afghanistan A)
    39444: 8638,   # Basil Hameed (United Arab Emirates)
    20229: 3083,   # Mohammad Ali (Pakistan)
    28978: 27373,  # Muhammad Farooq (United Arab Emirates)
}

# Homonym player_ids (same name, DIFFERENT real-world cricketers from different countries)
HOMONYM_DISAMBIGUATION = {
    7446: ("Abdul Samad", "Abdul Samad (IND)"),
    6922: ("Abdul Samad", "Abdul Samad (PAK)"),
    7338: ("Haider Ali", "Haider Ali (PAK)"),
    28756: ("Haider Ali", "Haider Ali (UAE)"),
    4722: ("Rashid Khan", "Rashid Khan"),
    20172: ("Rashid Khan", "Rashid Khan (NEP)"),
}

BOWLER_CREDITED_DISMISSAL_IDS = {1.0, 2.0, 3.0, 5.0, 6.0}  # Bowled, Caught, LBW, Stump, Hit Wicket

def file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def load_and_clean_data(data_dir: str):
    files = {
        "ball_by_ball": os.path.join(data_dir, "ball_by_ball.csv"),
        "matches": os.path.join(data_dir, "matches.csv"),
        "competitions": os.path.join(data_dir, "competitions.csv"),
        "teams": os.path.join(data_dir, "teams.csv"),
        "playing_eleven": os.path.join(data_dir, "playing_eleven.csv"),
        "mapping": os.path.join(data_dir, "competition_team_player_mapping.csv"),
    }
    checksums = {k: file_sha256(v) for k, v in files.items()}

    bbb = pd.read_csv(files["ball_by_ball"], low_memory=False)
    matches = pd.read_csv(files["matches"], low_memory=False)
    comps = pd.read_csv(files["competitions"], low_memory=False)
    teams = pd.read_csv(files["teams"], low_memory=False)
    xi = pd.read_csv(files["playing_eleven"], low_memory=False)
    mapping = pd.read_csv(files["mapping"], low_memory=False)

    # Canonicalize true duplicate player IDs across all tables
    bbb["striker_id"] = bbb["striker_id"].replace(TRUE_DUPLICATE_ID_MAP)
    bbb["non_striker_id"] = bbb["non_striker_id"].replace(TRUE_DUPLICATE_ID_MAP)
    bbb["bowler_id"] = bbb["bowler_id"].replace(TRUE_DUPLICATE_ID_MAP)
    bbb["dismissed_player_id"] = bbb["dismissed_player_id"].replace(TRUE_DUPLICATE_ID_MAP)
    xi["player_id"] = xi["player_id"].replace(TRUE_DUPLICATE_ID_MAP)
    mapping["player_id"] = mapping["player_id"].replace(TRUE_DUPLICATE_ID_MAP)

    # Disambiguate display names for homonyms while preserving exact Unstop dropdown string
    bbb["unstop_striker_name"] = bbb["striker_name"]
    bbb["unstop_bowler_name"] = bbb["bowler_name"]
    for pid, (unstop_name, disp_name) in HOMONYM_DISAMBIGUATION.items():
        bbb.loc[bbb["striker_id"] == pid, "striker_name"] = disp_name
        bbb.loc[bbb["bowler_id"] == pid, "bowler_name"] = disp_name
        mapping.loc[mapping["player_id"] == pid, "display_name"] = disp_name
    if "display_name" not in mapping.columns:
        mapping["display_name"] = mapping["player_name"]
    else:
        mapping["display_name"] = mapping["display_name"].fillna(mapping["player_name"])

    # Reconstruct Extra Runs (fixing the extras1_runs == extras_type_id schema bug)
    # extras1_runs is a categorical code: 1=Overstep, 2=wide, 3=legbyes, 4=byes, 5=other (no-ball), 6=None/Legal, 7=Penalty
    bbb["extras_type_clean"] = bbb["extras_type"].fillna("Legal")
    primary_extra = bbb["extras_type_clean"].map({
        "Legal": 0,
        "Overstep": 1,
        "other": 1,
        "wide": 1,
        "legbyes": 1,
        "byes": 1,
        "Penalty": 5,
    }).fillna(0).astype(int)
    secondary_extra = bbb["extras_2_runs"].fillna(0).astype(int)

    bbb["batter_runs"] = bbb["runs"].fillna(0).astype(int)
    bbb["reconstructed_extras"] = primary_extra + secondary_extra
    bbb["total_runs"] = bbb["batter_runs"] + bbb["reconstructed_extras"]

    # Naive broken calculation for audit comparison
    bbb["naive_total_runs"] = bbb["batter_runs"] + bbb["extras1_runs"].fillna(0).astype(int) + secondary_extra

    # Bowler-charged runs: batter_runs + wide runs + no-ball penalty (1 run for Overstep/other)
    # Byes, legbyes, and penalty runs are not charged to the bowler
    bowler_extra = np.where(
        bbb["extras_type_clean"] == "wide",
        1 + secondary_extra,
        np.where(bbb["extras_type_clean"].isin(["Overstep", "other"]), 1, 0)
    )
    bbb["bowler_runs"] = bbb["batter_runs"] + bowler_extra

    # Legal delivery vs ball faced
    bbb["is_legal_ball"] = (~bbb["extras_type_clean"].isin(["wide", "Overstep", "other", "Penalty"])).astype(int)
    bbb["is_ball_faced"] = (~bbb["extras_type_clean"].isin(["wide", "Penalty"])).astype(int)

    # Bowler-credited wicket (excluding RunOut=4, Retire Out=9, Obstructing=10)
    bbb["is_bowler_wicket"] = (
        bbb["is_wicket"].fillna(False) & bbb["dismissal_type_id"].isin(BOWLER_CREDITED_DISMISSAL_IDS)
    ).astype(int)
    bbb["is_batter_dismissed"] = (
        bbb["is_wicket"].fillna(False) & (bbb["dismissed_player_id"] == bbb["striker_id"]) & (bbb["dismissal_type_id"] != 9.0)
    ).astype(int)

    # Merge match and competition metadata
    m_meta = matches.merge(comps, on="comp_id", how="left")
    m_meta["is_ipl"] = m_meta["comp_name"].str.contains("IPL", case=False, na=False).astype(int)
    m_meta["is_knockout"] = m_meta["knock_out"].fillna(0).astype(int)

    # Map team names
    team_lookup = teams.drop_duplicates(subset=["team_id"]).set_index("team_id")["team_name"].to_dict()
    bbb["batting_team_name"] = bbb["batting_team_id"].map(team_lookup).fillna("Unknown")
    bbb["bowling_team_name"] = bbb["bowling_team_id"].map(team_lookup).fillna("Unknown")

    bbb = bbb.merge(
        m_meta[[
            "match_id", "comp_id", "comp_name", "match_format", "season", "match_date",
            "stage", "venue_id", "venue_country", "home_team", "away_team",
            "first_inn_revised_overs", "second_inn_revised_overs", "revised_target",
            "is_ipl", "is_knockout", "win_team_id", "result"
        ]],
        on="match_id",
        how="left"
    )

    # Sort strictly in chronological delivery sequence within each match and innings
    bbb = bbb.sort_values(["match_id", "innings", "innings_ball_num", "overid", "over_ball_num"]).reset_index(drop=True)

    # Phase classification
    bbb["phase"] = np.where(
        bbb["innings"] > 2, "SuperOver",
        np.where(bbb["overid"] <= 6, "Powerplay",
                 np.where(bbb["overid"] <= 15, "Middle", "Death"))
    )

    return {
        "bbb": bbb,
        "matches": matches,
        "comps": comps,
        "teams": teams,
        "xi": xi,
        "mapping": mapping,
        "checksums": checksums
    }
