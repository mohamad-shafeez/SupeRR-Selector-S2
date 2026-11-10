import numpy as np
import pandas as pd

# Capped Indian players or retired/overseas players who played IPL but have senior international caps (excluded from Pure Uncapped Indian pool)
SENIOR_CAPPED_OR_OVERSEAS_INDIAN_TEAM_NAMES = {
    "Virat Kohli", "Rohit Sharma", "Mahendra Singh Dhoni", "KL Rahul", "Rishabh Pant",
    "Hardik Pandya", "Jasprit Bumrah", "Suryakumar Yadav", "Sanju Samson", "Ravindra Jadeja",
    "Shreyas Iyer", "Ishan Kishan", "Shubman Gill", "Yashasvi Jaiswal", "Rinku Singh",
    "Namboori Tilak Varma", "Tilak Varma", "Axar Patel", "Kuldeep Yadav", "Arshdeep Singh",
    "Mohammed Siraj", "Mohammed Shami", "Bhuvneshwar Kumar", "Yuzvendra Chahal", "Ravichandran Ashwin",
    "Shivam Dube", "Ruturaj Gaikwad", "Abhishek Sharma", "Sai Sudharsan", "Riyan Parag",
    "Dhruv Jurel", "Jitesh Sharma", "Washington Sundar", "Ravi Bishnoi", "Avesh Khan",
    "Mukesh Kumar", "Tushar Deshpande", "Harshit Rana", "Nitish Kumar Reddy", "Ramandeep Singh",
    "Mayank Yadav", "Varun Chakravarthy", "Venkatesh Iyer", "Deepak Chahar", "Shardul Thakur",
    "T Natarajan", "Khaleel Ahmed", "Prasidh Krishna", "Harshal Patel", "Umran Malik",
    "Deepak Hooda", "Rahul Tripathi", "Devdutt Padikkal", "Rajat Patidar", "Ajinkya Rahane",
    "Shikhar Dhawan", "Dinesh Karthik", "Ambati Rayudu", "Wriddhiman Saha", "Ishant Sharma",
    "Umesh Yadav", "Mayank Agarwal", "Prithvi Shaw", "Nitish Rana", "Krunal Pandya",
    "Vijay Shankar", "Sandeep Sharma", "Mohit Sharma", "Manish Pandey", "Karun Nair",
    "Jaydev Unadkat", "Siddarth Kaul", "Shahbaz Ahmed", "Rahul Chahar", "Navdeep Saini", "Sai Kishore",
    "Chetan Sakariya", "Sandeep Warrier", "karun nair", "Kedar Jadhav", "Piyush Chawla",
    "Amit Mishra", "Karn Sharma", "Murugan Ashwin", "Rishi Dhawan", "parvez rasool",
    # Overseas players who only appeared in IPL in this dataset
    "Faf du Plessis", "Sunil Narine", "Andre Russell", "Kieron Pollard", "Dwayne Bravo",
    "AB de Villiers", "Chris Gayle", "Shane Watson", "Lasith Malinga", "Imran Tahir",
    "Moeen Ali", " Jonny Bairstow", "Jonny Bairstow", "Jos Buttler", "Ben Stokes",
    "Jofra Archer", "Sam Curran", "Liam Livingstone", "Phil Salt", "Will Jacks",
    "Harry Brook", "Pat Cummins", "Mitchell Starc", "Travis Head", "Glenn Maxwell",
    "Marcus Stoinis", "Cameron Green", "Tim David", "Jake Fraser-McGurk", "Josh Hazlewood",
    "Nathan Ellis", "Adam Zampa", "Spencer Johnson", "Trent Boult", "Lockie Ferguson",
    "Devon Conway", "Rachin Ravindra", "Daryl Mitchell", "Glenn Phillips", "Mitchell Santner",
    "Quinton de Kock", "Heinrich Klaasen", "David Miller", "Aiden Markram", "Tristan Stubbs",
    "Marco Jansen", "Kagiso Rabada", "Anrich Nortje", "Gerald Coetzee", "Dewald Brevis",
    "Nandre Burger", "Lungi Ngidi", "Faf Du Plessis", "Nicholas Pooran", "Shimron Hetmyer",
    "Rovman Powell", "Romario Shepherd", "Sherfane Rutherford", "Alzarri Joseph",
    "Kyle Mayers", "Shai Hope", "Akeal Hosein", "Jason Holder", "Obed McCoy",
    "Rashid Khan", "Noor Ahmad", "Rahmanullah Gurbaz", "Azmatullah Omarzai", "Mohammad Nabi",
    "Fazalhaq Farooqi", "Naveen-ul-Haq", "Mujeeb Ur Rahman", "Allah Ghazanfar",
    "Matheesha Pathirana", "Wanindu Hasaranga", "Maheesh Theekshana", "Nuwan Thushara",
    "Dushmantha Chameera", "Bhanuka Rajapaksa", "Kamindu Mendis", "Mustafizur Rahman",
    "Litton Das", "Shakib Al Hasan", "Sikandar Raza", "Joshua Little", "David Warner",
    "Matthew Short", "Mitch Marsh", "Mitchell Marsh", "Riley Meredith", "Daniel Sams",
    "Tim Southee", "Jimmy Neesham", "Kane Williamson", "Finn Allen", "Kyle Jamieson",
    "Rassie van der Dussen", "Dwaine Pretorius", "Wayne Parnell", "Rilee Rossouw",
    "Donovan Ferreira", "Kwena Maphaka", "Keshav Maharaj", "Tabraiz Shamsi",
    "corbin bosch", "Corbin Bosch", "Ryan Rickelton", "Will O'Rourke", "Jacob Bethell",
    "Brydon Carse", "Jamie Overton", "Reece Topley", "Luke Wood", "Richard Gleeson",
    "Chris Jordan", "Tymal Mills", "Adil Rashid", "Alex Hales", "Jason Roy", "Eoin Morgan",
    "Evins Lewis", "Evin Lewis", "Fabian Allen", "Odean Smith", "Dominic Drakes",
}

def build_player_eligibility_metadata(data_bundle: dict) -> pd.DataFrame:
    mapping = data_bundle["mapping"]
    comps = data_bundle["comps"]
    bbb = data_bundle["bbb"]

    m = mapping.merge(comps, on=["comp_id", "comp_name"], how="left")
    # Identify international T20I teams
    t20i_rows = m[m["match_format"] == "T20I"]
    intl_teams_by_pid = t20i_rows.groupby("player_id")["team_name"].unique().to_dict()

    ipl_rows = m[m["comp_name"].str.contains("IPL", case=False, na=False)]
    ipl_teams_by_pid = ipl_rows.groupby("player_id")["team_name"].unique().to_dict()
    RETIRED_OR_INACTIVE_2027 = {"Dinesh Karthik", "Matthew Wade", "Tim Southee", "Mahendra Singh Dhoni", "Faf du Plessis", "Moeen Ali", "David Warner", "Shikhar Dhawan", "Wriddhiman Saha", "Ambati Rayudu", "Dwayne Bravo", "Kieron Pollard", "Eoin Morgan", "Robin Uthappa", "Piyush Chawla", "Amit Mishra", "Mohit Sharma"}
    ipl_2026_rows = m[m["comp_name"] == "IPL 2026"]
    ipl_2026_team_by_pid = ipl_2026_rows.groupby("player_id")["team_name"].last().to_dict()

    india_a_pids = set(m[m["team_name"] == "India A"]["player_id"].unique())
    rr_pids = set(m[m["team_name"] == "Rajasthan Royals"]["player_id"].unique())

    # Build master player directory
    p_dir = mapping.groupby("player_id").agg(
        unstop_name=("player_name", "first"),
        display_name=("display_name", "first"),
    ).reset_index()

    records = []
    for _, row in p_dir.iterrows():
        pid = int(row["player_id"])
        uname = str(row["unstop_name"])
        dname = str(row["display_name"])
        intl_teams = list(intl_teams_by_pid.get(pid, []))
        ipl_teams = list(ipl_teams_by_pid.get(pid, []))
        has_ipl = len(ipl_teams) > 0
        has_india_a = pid in india_a_pids
        has_t20i = len(intl_teams) > 0
        played_for_india = "India" in intl_teams
        played_other_intl = any(t != "India" for t in intl_teams)

        # Is player an uncapped Indian eligible for Track B?
        is_uncapped_indian = (
            (has_ipl or has_india_a)
            and ((not played_for_india) or (uname in {"Naman Dhir", "Vipraj Nigam", "Anshul Kamboj", "Priyansh Arya", "Prabhsimran Singh", "Nehal Wadhera", "Suyash Sharma", "Vaibhav Arora", "Harpreet Brar", "Yash Thakur", "Mohsin Khan", "Rasikh Salam", "Ashutosh Sharma", "Shashank Singh", "Ayush Badoni", "Sameer Rizvi", "Aniket Verma"}))
            and (not played_other_intl)
            and (uname not in SENIOR_CAPPED_OR_OVERSEAS_INDIAN_TEAM_NAMES)
            and (dname not in SENIOR_CAPPED_OR_OVERSEAS_INDIAN_TEAM_NAMES)
        )
        is_overseas = played_other_intl or (
            uname in SENIOR_CAPPED_OR_OVERSEAS_INDIAN_TEAM_NAMES
            and not played_for_india
            and uname not in {
                "Virat Kohli", "Rohit Sharma", "Mahendra Singh Dhoni", "KL Rahul", "Rishabh Pant",
                "Hardik Pandya", "Jasprit Bumrah", "Suryakumar Yadav", "Sanju Samson", "Ravindra Jadeja",
                "Shreyas Iyer", "Ishan Kishan", "Shubman Gill", "Yashasvi Jaiswal", "Rinku Singh",
                "Namboori Tilak Varma", "Tilak Varma", "Axar Patel", "Kuldeep Yadav", "Arshdeep Singh",
                "Mohammed Siraj", "Mohammed Shami", "Bhuvneshwar Kumar", "Yuzvendra Chahal", "Ravichandran Ashwin",
                "Shivam Dube", "Ruturaj Gaikwad", "Abhishek Sharma", "Sai Sudharsan", "Riyan Parag",
                "Dhruv Jurel", "Jitesh Sharma", "Washington Sundar", "Ravi Bishnoi", "Avesh Khan",
                "Mukesh Kumar", "Tushar Deshpande", "Harshit Rana", "Nitish Kumar Reddy", "Ramandeep Singh",
                "Mayank Yadav", "Varun Chakravarthy", "Venkatesh Iyer", "Deepak Chahar", "Shardul Thakur",
                "T Natarajan", "Khaleel Ahmed", "Prasidh Krishna", "Harshal Patel", "Umran Malik",
                "Deepak Hooda", "Rahul Tripathi", "Devdutt Padikkal", "Rajat Patidar", "Ajinkya Rahane",
                "Shikhar Dhawan", "Dinesh Karthik", "Ambati Rayudu", "Wriddhiman Saha", "Ishant Sharma",
                "Umesh Yadav", "Mayank Agarwal", "Prithvi Shaw", "Nitish Rana", "Krunal Pandya",
                "Vijay Shankar", "Sandeep Sharma", "Mohit Sharma", "Manish Pandey", "Karun Nair",
                "Jaydev Unadkat", "Siddarth Kaul", "Shahbaz Ahmed", "Rahul Chahar", "Navdeep Saini"
            }
        )

        records.append({
            "player_id": pid,
            "unstop_name": uname,
            "display_name": dname,
            "has_ipl": bool(has_ipl and (uname not in RETIRED_OR_INACTIVE_2027)),
            "is_active_2027": bool(uname not in RETIRED_OR_INACTIVE_2027),
            "has_india_a": bool(has_india_a),
            "has_t20i": bool(has_t20i),
            "is_uncapped_indian": bool(is_uncapped_indian),
            "is_overseas": bool(is_overseas),
            "is_rr_player": bool(pid in rr_pids),
            "latest_ipl_team": str(ipl_2026_team_by_pid.get(pid, ipl_teams[-1] if ipl_teams else (intl_teams[-1] if intl_teams else "Emerging"))),
            "teams_list": ", ".join(ipl_teams if ipl_teams else intl_teams[:2]),
        })
    return pd.DataFrame(records)
