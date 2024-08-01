import logging
from datetime import datetime
from typing import Any

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_init_elo(match_data: pd.DataFrame) -> pd.DataFrame:
    first_season = match_data.season.min()
    home_xg_mean = (
        match_data.loc[match_data["season"] == first_season]
        .groupby("team")["xg"]
        .mean()
    )
    home_xga_mean = (
        match_data.loc[match_data["season"] == first_season]
        .groupby("team")["xga"]
        .mean()
    )
    away_xg_mean = (
        match_data.loc[match_data["season"] == first_season]
        .groupby("opponent")["xga"]
        .mean()
    )
    away_xga_mean = (
        match_data.loc[match_data["season"] == first_season]
        .groupby("opponent")["xg"]
        .mean()
    )
    xg_mean = (home_xg_mean + away_xg_mean) / 2
    xga_mean = (home_xga_mean + away_xga_mean) / 2
    first_match = match_data.date.min()
    before_first_match = pd.to_datetime(first_match) - pd.DateOffset(days=1)
    init_elo_df = pd.DataFrame(
        {
            "season": first_season,
            "date": before_first_match.strftime("%Y-%m-%d"),
            "att_elo": xg_mean,
            "def_elo": xga_mean,
            "home_att_elo": home_xg_mean,
            "home_def_elo": home_xga_mean,
            "away_att_elo": away_xg_mean,
            "away_def_elo": away_xga_mean,
        }
    )
    init_elo_df.reset_index(inplace=True)
    init_elo_df.rename(columns={"index": "team"}, inplace=True)
    return init_elo_df


def get_promotions_relegations(
    match_data: pd.DataFrame,
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    seasons = sorted(match_data.season.unique().tolist())
    promotions = {}
    relegations = {}
    for prev_season, curr_season in zip(seasons, seasons[1:]):
        prev_teams = match_data[match_data.season == prev_season].team.unique().tolist()
        curr_teams = match_data[match_data.season == curr_season].team.unique().tolist()
        promotions[curr_season] = [
            team for team in curr_teams if team not in prev_teams
        ]
        relegations[curr_season] = [
            team for team in prev_teams if team not in curr_teams
        ]
    return seasons, promotions, relegations


def get_expected_xg(
    att_elo: float,
    home_away_att_elo: float,
    def_elo: float,
    home_away_def_elo: float,
    home_away_weight: float = 0.20,
) -> float:
    expected_xg = (1 - home_away_weight) * (att_elo + def_elo) + home_away_weight * (
        home_away_att_elo + home_away_def_elo
    )
    return expected_xg


def get_next_elo(
    curr_elo: float, actual_xg: float, expected_xg: float, learning_rate: float = 0.12
) -> float:
    updated_elo = curr_elo + learning_rate * (actual_xg - expected_xg)
    return updated_elo


elo_cols = [
    "att_elo",
    "def_elo",
    "home_att_elo",
    "home_def_elo",
    "away_att_elo",
    "away_def_elo",
]


def get_relegated_teams_mean_elos(
    elo_df: pd.DataFrame, relegated_teams: list[str], prev_season
) -> pd.Series:
    relegated_elo_df = elo_df.loc[
        ((elo_df.team.isin(relegated_teams)) & (elo_df.season == prev_season)),
        elo_cols,
    ]
    return relegated_elo_df.mean()


def get_latest_elos(
    season: str,
    prev_season: str,
    team: str,
    elo_df: pd.DataFrame,
) -> pd.Series:
    same_team_within_two_seasons = elo_df.loc[
        (elo_df.season.isin([season, prev_season])) & (elo_df.team == team)
    ]
    latest_elos = same_team_within_two_seasons.drop_duplicates(
        "team", keep="last"
    ).iloc[0]
    latest_elos = latest_elos.loc[elo_cols]
    return latest_elos


def add_promoted_team_rows(
    match_data: pd.DataFrame,
    promotions: dict[str, list[str]],
    relegations: dict[str, list[str]],
    seasons: list[str],
) -> pd.DataFrame:
    for season, promoted_teams in promotions.items():
        for team in promoted_teams:
            first_match_date = match_data.loc[
                ((match_data.team == team) | (match_data.opponent == team))
                & (match_data["season"] == season),
                "date",
            ].min()
            one_day_before = pd.to_datetime(first_match_date) - pd.DateOffset(days=1)
            curr_idx = seasons.index(season)
            prev_season = seasons[curr_idx - 1]

            relegated_teams = relegations[season]
            relegated_match_data = match_data.loc[
                (
                    (
                        (match_data.team.isin(relegated_teams))
                        | (match_data.opponent.isin(relegated_teams))
                    )
                    & (match_data.season == prev_season)
                ),
                ["team", "opponent", "xg", "xga", "date"],
            ]

            relegated_match_data["temp_xg"] = relegated_match_data.apply(
                lambda row: row["xg"] if row["team"] in relegated_teams else row["xga"],
                axis=1,
            )
            relegated_match_data["temp_xga"] = relegated_match_data.apply(
                lambda row: row["xga"] if row["team"] in relegated_teams else row["xg"],
                axis=1,
            )
            relegated_match_data = relegated_match_data.sort_values(
                by="date", ascending=True
            )
            last_matches = relegated_match_data.groupby(
                relegated_match_data.apply(
                    lambda row: (
                        row["team"]
                        if row["team"] in relegated_teams
                        else row["opponent"]
                    ),
                    axis=1,
                )
            ).last()

            mean_xg = last_matches["temp_xg"].mean()
            mean_xga = last_matches["temp_xga"].mean()

            new_row = pd.DataFrame(
                [
                    {
                        "season": season,
                        "team": team,
                        "round": 0,
                        "date": one_day_before.strftime("%Y-%m-%d"),
                        "opponent": relegated_teams[0],
                        "xg": mean_xg,
                        "xga": mean_xga,
                    }
                ]
            )

            match_data = pd.concat([match_data, new_row], ignore_index=True)
    return match_data


def is_first_match_after_promotion(
    team: str, promoted_teams: list[str], round=int
) -> bool:
    return team in promoted_teams and round == 0


def calculate_elo_score(
    data_check_complete,
    match_data: pd.DataFrame,
    read_elo_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    home_away_weight = parameters["home_away_weight"]
    learning_rate = parameters["elo_learning_rate"]
    logger.info(
        f"Calculating elo score with lr={learning_rate}; h/a_weight={home_away_weight}"
    )

    match_data["xg"] = match_data["xg"].fillna(match_data["gf"])
    match_data["xga"] = match_data["xga"].fillna(match_data["ga"])
    match_data = match_data.loc[
        (match_data["comp"] == "Premier League") & (match_data["venue"] == "Home"),
        ["season", "team", "round", "date", "opponent", "xg", "xga"],
    ]

    if parameters["use_cache"]:
        elo_df = read_elo_data
        cached_date = elo_df["date"].max()
        match_data = match_data[match_data["date"] > cached_date]
    else:
        elo_df = get_init_elo(match_data)
    if match_data.empty:
        return elo_df

    seasons, promotions, relegations = get_promotions_relegations(match_data)
    match_data = add_promoted_team_rows(
        match_data=match_data,
        promotions=promotions,
        relegations=relegations,
        seasons=seasons,
    )

    match_data = match_data.sort_values(["season", "date", "team"]).reset_index(
        drop=True
    )

    for _, row in tqdm(match_data.iterrows(), total=match_data.shape[0]):
        curr_idx = seasons.index(row.season)
        prev_season = seasons[curr_idx - 1]

        promoted_teams = promotions[row.season] if row.season in promotions else []
        if is_first_match_after_promotion(row.team, promoted_teams, row["round"]):
            home_team_elos = get_relegated_teams_mean_elos(
                elo_df=elo_df,
                relegated_teams=relegations[row.season],
                prev_season=prev_season,
            )
        else:
            home_team_elos = get_latest_elos(
                season=row.season,
                prev_season=prev_season,
                team=row.team,
                elo_df=elo_df,
            )
        if is_first_match_after_promotion(row.opponent, promoted_teams, row["round"]):
            away_team_elos = get_relegated_teams_mean_elos(
                elo_df=elo_df,
                relegated_teams=relegations[row.season],
                prev_season=prev_season,
            )
        else:
            away_team_elos = get_latest_elos(
                season=row.season,
                prev_season=prev_season,
                team=row.opponent,
                elo_df=elo_df,
            )

        expected_xg = get_expected_xg(
            home_team_elos.att_elo,
            home_team_elos.home_att_elo,
            away_team_elos.def_elo,
            away_team_elos.away_def_elo,
            home_away_weight,
        )
        expected_xga = get_expected_xg(
            away_team_elos.att_elo,
            away_team_elos.away_att_elo,
            home_team_elos.def_elo,
            home_team_elos.home_def_elo,
            home_away_weight,
        )

        next_home_team_elos = {
            "team": row.team,
            "season": row.season,
            "date": row.date,
            "att_elo": get_next_elo(
                home_team_elos.att_elo, row.xg, expected_xg, learning_rate
            ),
            "def_elo": get_next_elo(
                home_team_elos.def_elo, row.xga, expected_xga, learning_rate
            ),
            "home_att_elo": get_next_elo(
                home_team_elos.home_att_elo, row.xg, expected_xg, learning_rate
            ),
            "home_def_elo": get_next_elo(
                home_team_elos.home_def_elo, row.xga, expected_xga, learning_rate
            ),
            "away_att_elo": home_team_elos.away_att_elo,
            "away_def_elo": home_team_elos.away_def_elo,
        }
        next_home_team_elos = pd.DataFrame(next_home_team_elos, index=[0])
        elo_df = pd.concat([elo_df, next_home_team_elos], ignore_index=True)

        next_away_team_elos = {
            "team": row.opponent,
            "season": row.season,
            "date": row.date,
            "att_elo": get_next_elo(
                away_team_elos.att_elo, row.xga, expected_xga, learning_rate
            ),
            "def_elo": get_next_elo(
                away_team_elos.def_elo, row.xg, expected_xg, learning_rate
            ),
            "home_att_elo": away_team_elos.home_att_elo,
            "home_def_elo": away_team_elos.home_def_elo,
            "away_att_elo": get_next_elo(
                away_team_elos.away_att_elo, row.xga, expected_xga, learning_rate
            ),
            "away_def_elo": get_next_elo(
                away_team_elos.away_def_elo, row.xg, expected_xg, learning_rate
            ),
        }
        next_away_team_elos = pd.DataFrame(next_away_team_elos, index=[0])
        elo_df = pd.concat([elo_df, next_away_team_elos], ignore_index=True)

    ffill_cols = [
        "att_elo",
        "def_elo",
        "home_att_elo",
        "home_def_elo",
        "away_att_elo",
        "away_def_elo",
    ]
    elo_df[ffill_cols] = elo_df.groupby("team")[ffill_cols].transform(
        lambda x: x.ffill()
    )

    return elo_df


if __name__ == "__main__":
    import sqlite3

    conn = sqlite3.connect("data/fpl.db")
    match_log = pd.read_sql(f"select * from raw_team_match_log", conn)
    parameters = dict(
        elo_learning_rate=0.1,
        home_away_weight=0.5,
        use_cache=False,
    )
    elo_data = calculate_elo_score(match_log, None, parameters)
    print(elo_data)
