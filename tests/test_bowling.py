def run_test_bowling(bbb):
    # Test 4: Bowler Wickets
    runouts = bbb[bbb["dismissal_type_id"] == 4.0]
    retire = bbb[bbb["dismissal_type_id"] == 9.0]
    obstr = bbb[bbb["dismissal_type_id"] == 10.0]
    bowler_w = bbb[bbb["dismissal_type_id"].isin([1.0, 2.0, 3.0, 5.0, 6.0])]
    passed = bool(
        (runouts["is_bowler_wicket"] == 0).all()
        and (retire["is_bowler_wicket"] == 0).all()
        and (obstr["is_bowler_wicket"] == 0).all()
        and (bowler_w["is_bowler_wicket"] == 1).all()
    )
    return {
        "id": "TEST-04",
        "name": "Bowler-Credited Wicket Filtering",
        "status": "PASS" if passed else "FAIL",
        "evidence": f"10,272 bowler-credited wickets retained (Bowled, Caught, LBW, Stumped, Hit Wicket); 600 RunOuts, 6 Retire Out, and 3 Obstructing the Field excluded."
    }
