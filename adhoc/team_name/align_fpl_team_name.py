import sqlite3

import numpy as np
import pandas as pd


def find_missing_integer(group, all_teams):
    missing_opponent = [
        team for team in all_teams if team not in group["opponent_team"].tolist()
    ]
    if len(missing_opponent) == 1:
        return missing_opponent[0]


def main(conn, cur, fpl_data, season_team_id):
    for season in fpl_data.season.unique().tolist():
        season_data = fpl_data.loc[fpl_data["season"] == season].copy()
        all_teams = sorted(season_data["opponent_team"].unique().tolist())
        mapping = (
            season_data.groupby(["season", "full_name"])
            .apply(find_missing_integer, all_teams)
            .to_dict()
        )
        season_data["team_id"] = season_data.apply(
            lambda row: mapping.get((row["season"], row["full_name"]), None), axis=1
        )
        mode_id = (
            season_data.groupby(["fixture", "was_home"], group_keys=False)["team_id"]
            .agg(pd.Series.mode)
            .to_dict()
        )
        season_data["team_id"] = season_data.apply(
            lambda row: mode_id[(row["fixture"], row["was_home"])], axis=1
        )

        for idx, group in season_data.groupby(["fixture", "was_home"]):
            if group.team.notna().all() and group.opponent_team_name.notna().all():
                continue
            top_idx = group["total_points"].idxmax()
            if np.isnan(top_idx):
                top_home = group.iloc[0]
            else:
                top_home = group.loc[top_idx]
            fixture = top_home.fixture
            was_home = top_home.was_home
            team_id = top_home.team_id
            opponent_team = top_home.opponent_team
            team = season_team_id[(season, team_id)]
            opponent_team_name = season_team_id[(season, opponent_team)]
            if group.team_id.nunique() != 1 or group.opponent_team.nunique() != 1:
                raise Exception("error")

            team_string = team.replace("'", "''")
            opponent_team_name_string = opponent_team_name.replace("'", "''")
            query = f"""
            UPDATE raw_fpl_data SET team = '{team_string}', opponent_team_name = '{opponent_team_name_string}' 
            WHERE season = '{season}' and fixture = {fixture} and was_home = {was_home}
            """
            cur.execute(query)
            if cur.rowcount > 0:
                print(
                    f"{cur.rowcount} X {season=},{fixture=},{was_home=} --> {team}\te.g. {top_home.full_name}"
                )
            conn.commit()
    return None


if __name__ == "__main__":
    conn = sqlite3.connect("data/fpl.db")
    cur = conn.cursor()

    fpl_data = pd.read_sql(
        f"select season, team, full_name, opponent_team, opponent_team_name, fixture, was_home, total_points from raw_fpl_data",
        conn,
    )

    season_team_id = (
        pd.read_csv("src/fpl/adhoc/season_team_id.csv")
        .set_index(["season", "team"])["team_name"]
        .to_dict()
    )

    main(conn, cur, fpl_data, season_team_id)

    conn.close()
