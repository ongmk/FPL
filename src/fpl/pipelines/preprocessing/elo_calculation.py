import logging
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def get_init_elo(team_match_log: pd.DataFrame) -> pd.DataFrame:
    first_season = team_match_log.season.min()
    home_xg_mean = (
        team_match_log.loc[team_match_log["season"] == first_season]
        .groupby("team")["team_xg"]
        .mean()
    )
    home_xga_mean = (
        team_match_log.loc[team_match_log["season"] == first_season]
        .groupby("team")["team_xga"]
        .mean()
    )
    away_xg_mean = (
        team_match_log.loc[team_match_log["season"] == first_season]
        .groupby("opponent")["team_xga"]
        .mean()
    )
    away_xga_mean = (
        team_match_log.loc[team_match_log["season"] == first_season]
        .groupby("opponent")["team_xg"]
        .mean()
    )
    xg_mean = (home_xg_mean + away_xg_mean) / 2
    xga_mean = (home_xga_mean + away_xga_mean) / 2
    first_match = team_match_log.date.min()
    before_first_match = pd.to_datetime(first_match) - pd.DateOffset(days=1)
    init_elo_df = pd.DataFrame(
        {
            "season": first_season,
            "date": before_first_match,
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


def get_teams_set(team_match_log, elo_df, season):
    return set(
        np.concatenate(
            [
                team_match_log.loc[team_match_log["season"] == season, "team"].unique(),
                team_match_log.loc[
                    team_match_log["season"] == season, "opponent"
                ].unique(),
                elo_df.loc[elo_df["season"] == season, "team"].unique(),
            ]
        )
    )


def get_promotions_relegations(
    team_match_log: pd.DataFrame,
    elo_df: pd.DataFrame,
) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:

    seasons = sorted(
        set(team_match_log.season.unique().tolist() + elo_df.season.unique().tolist())
    )
    promotions = {}
    relegations = {}
    for prev_season, curr_season in zip(seasons, seasons[1:]):
        prev_teams = get_teams_set(team_match_log, elo_df, prev_season)
        curr_teams = get_teams_set(team_match_log, elo_df, curr_season)
        promotions[curr_season] = list(curr_teams - prev_teams)
        relegations[curr_season] = list(prev_teams - curr_teams)
        assert len(promotions[curr_season]) == len(relegations[curr_season]) == 3
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
    if pd.isna(actual_xg):
        return curr_elo
    updated_elo = curr_elo + learning_rate * (actual_xg - expected_xg)
    assert pd.notna(
        updated_elo
    ), f"Updated elo is NaN: {curr_elo}, {actual_xg}, {expected_xg}"
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
    team_match_log: pd.DataFrame,
    promotions: dict[str, list[str]],
    relegations: dict[str, list[str]],
    seasons: list[str],
) -> pd.DataFrame:
    for season, promoted_teams in promotions.items():
        for team in promoted_teams:
            first_match_date = team_match_log.loc[
                ((team_match_log.team == team) | (team_match_log.opponent == team))
                & (team_match_log["season"] == season),
                "date",
            ].min()
            one_day_before = pd.to_datetime(first_match_date) - pd.DateOffset(days=1)
            curr_idx = seasons.index(season)
            prev_season = seasons[curr_idx - 1]

            relegated_teams = relegations[season]
            relegated_match_data = team_match_log.loc[
                (
                    (
                        (team_match_log.team.isin(relegated_teams))
                        | (team_match_log.opponent.isin(relegated_teams))
                    )
                    & (team_match_log.season == prev_season)
                ),
                ["team", "opponent", "team_xg", "team_xga", "date"],
            ]

            relegated_match_data["temp_xg"] = relegated_match_data.apply(
                lambda row: (
                    row["team_xg"]
                    if row["team"] in relegated_teams
                    else row["team_xga"]
                ),
                axis=1,
            )
            relegated_match_data["temp_xga"] = relegated_match_data.apply(
                lambda row: (
                    row["team_xga"]
                    if row["team"] in relegated_teams
                    else row["team_xg"]
                ),
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
                        "date": one_day_before,
                        "opponent": relegated_teams[0],
                        "team_xg": mean_xg,
                        "team_xga": mean_xga,
                    }
                ]
            )

            team_match_log = pd.concat([team_match_log, new_row], ignore_index=True)
    return team_match_log


def is_first_match_after_promotion(
    team: str, promoted_teams: list[str], round=int
) -> bool:
    return team in promoted_teams and round == 0


def calculate_elo_score(
    team_match_log: pd.DataFrame,
    fpl_data: pd.DataFrame,
    read_elo_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    home_away_weight = parameters["home_away_weight"]
    learning_rate = parameters["elo_learning_rate"]
    logger.info(
        f"Calculating elo score with lr={learning_rate}; h/a_weight={home_away_weight}"
    )

    team_match_log, elo_df = data_preparation(
        team_match_log, fpl_data, read_elo_data, parameters
    )
    assert (
        team_match_log.groupby(["season", "team", "date", "opponent"]).size().max() == 1
    )

    if team_match_log.empty:
        return elo_df
    seasons, promotions, relegations = get_promotions_relegations(
        team_match_log, elo_df
    )
    team_match_log = add_promoted_team_rows(
        team_match_log=team_match_log,
        promotions=promotions,
        relegations=relegations,
        seasons=seasons,
    )

    team_match_log = team_match_log.sort_values(["season", "date", "team"]).reset_index(
        drop=True
    )

    for _, row in tqdm(team_match_log.iterrows(), total=team_match_log.shape[0]):
        curr_idx = seasons.index(row.season)
        prev_season = seasons[curr_idx - 1]

        promoted_teams = promotions.get(row.season, [])
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
                home_team_elos.att_elo, row.team_xg, expected_xg, learning_rate
            ),
            "def_elo": get_next_elo(
                home_team_elos.def_elo, row.team_xga, expected_xga, learning_rate
            ),
            "home_att_elo": get_next_elo(
                home_team_elos.home_att_elo, row.team_xg, expected_xg, learning_rate
            ),
            "home_def_elo": get_next_elo(
                home_team_elos.home_def_elo, row.team_xga, expected_xga, learning_rate
            ),
            "away_att_elo": home_team_elos.away_att_elo,
            "away_def_elo": home_team_elos.away_def_elo,
        }
        next_home_team_elos = pd.DataFrame(next_home_team_elos, index=[0])
        elo_df = pd.concat([elo_df, next_home_team_elos], ignore_index=True)

        if row.opponent in relegations.get(row.season, []):
            continue
        next_away_team_elos = {
            "team": row.opponent,
            "season": row.season,
            "date": row.date,
            "att_elo": get_next_elo(
                away_team_elos.att_elo, row.team_xga, expected_xga, learning_rate
            ),
            "def_elo": get_next_elo(
                away_team_elos.def_elo, row.team_xg, expected_xg, learning_rate
            ),
            "home_att_elo": away_team_elos.home_att_elo,
            "home_def_elo": away_team_elos.home_def_elo,
            "away_att_elo": get_next_elo(
                away_team_elos.away_att_elo, row.team_xga, expected_xga, learning_rate
            ),
            "away_def_elo": get_next_elo(
                away_team_elos.away_def_elo, row.team_xg, expected_xg, learning_rate
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


def data_preparation(team_match_log, fpl_data, read_elo_data, parameters):
    team_match_log = pd.merge(
        fpl_data[
            ["season", "round", "venue", "team", "date", "opponent"]
        ].drop_duplicates(),
        team_match_log.drop(columns=["opponent"]),
        on=["team", "date"],
        how="outer",
    )

    team_match_log["team_xg"] = team_match_log["team_xg"].fillna(
        team_match_log["team_gf"]
    )
    team_match_log["team_xga"] = team_match_log["team_xga"].fillna(
        team_match_log["team_ga"]
    )
    team_match_log = team_match_log.loc[
        team_match_log["venue"] == "Home",
        ["season", "team", "round", "date", "opponent", "team_xg", "team_xga"],
    ]

    if parameters["use_cache"]:
        elo_df = read_elo_data
        cached_date = elo_df["date"].max()
        team_match_log = team_match_log[team_match_log["date"] > cached_date]
    else:
        elo_df = get_init_elo(team_match_log)
    return team_match_log, elo_df


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
