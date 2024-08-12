import sqlite3

import numpy as np
import pandas as pd


def find_missing_integer(group, all_teams):
    missing_opponent = [
        team for team in all_teams if team not in group["opponent_team"].tolist()
    ]
    if len(missing_opponent) == 1:
        return missing_opponent[0]


def main(conn, cur, fpl_data):
    correct = fpl_data.loc[fpl_data["cnt"] > 1].copy()
    incorrect = fpl_data.loc[fpl_data["cnt"] == 1].copy()

    for idx, row in incorrect.iterrows():
        correct_team = correct.loc[
            (correct["kickoff_time"] == row["kickoff_time"])
            & (correct["opponent_team"] == row["opponent_team"]),
            "team",
        ]
        assert len(correct_team) == 1
        correct_team = correct_team.iloc[0]
        correct_team_name = correct.loc[
            (correct["kickoff_time"] == row["kickoff_time"])
            & (correct["opponent_team"] == row["opponent_team"]),
            "team_name",
        ]
        assert len(correct_team_name) == 1
        correct_team_name = correct_team_name.iloc[0]
        print(
            row["kickoff_time"],
            row["team"],
            row["team_name"],
            row["opponent_team"],
            row["opponent_team_name"],
        )

        team_string = correct_team_name.replace("'", "''")
        query = f"""
        UPDATE raw_fpl_data SET team_name = '{team_string}', team = {correct_team}
        WHERE season = '2023-2024' and kickoff_time = '{row["kickoff_time"]}' and opponent_team = {row["opponent_team"]}
        """
        cur.execute(query)
        if cur.rowcount > 0:
            print(
                f"{cur.rowcount} X {row['kickoff_time']=},{row['team']=},{row['team_name']=},{row['opponent_team']=},{row['opponent_team_name']=}--> {correct_team=},{correct_team_name=}"
            )
        conn.commit()
    return None


if __name__ == "__main__":
    conn = sqlite3.connect("data/fpl.db")
    cur = conn.cursor()

    fpl_data = pd.read_sql(
        """SELECT kickoff_time, team_name, team, opponent_team, opponent_team_name, count(*) cnt
FROM raw_fpl_data
WHERE season = '2023-2024'
GROUP BY kickoff_time, team_name, team, opponent_team, opponent_team_name""",
        conn,
    )

    main(conn, cur, fpl_data)

    conn.close()
