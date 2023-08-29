import itertools
import re
from datetime import timedelta
from typing import Any

import pandas as pd


def filter_data(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    parameters: dict[str, Any],
):
    team_match_log["days_till_next"] = (
        (team_match_log.groupby("team")["date"].shift(-1) - team_match_log["date"])
        .apply(lambda x: x.days)
        .clip(upper=7)
    )
    team_match_log["days_since_last"] = (
        (team_match_log["date"] - team_match_log.groupby("team")["date"].shift(1))
        .apply(lambda x: x.days)
        .clip(upper=7)
    )
    player_match_log = player_match_log.loc[
        (player_match_log["comp"] == "Premier League")
        & (player_match_log["season"] >= parameters["start_year"]),
        [
            "season",
            "player",
            "date",
            "round",
            "venue",
            "squad",
            "opponent",
            "start",
            "pos",
            "min",
            "gls",
            "ast",
            "pk",
            "pkatt",
            "sh",
            "sot",
            "touches",
            "xg",
            "npxg",
            "xag",
            "sca",
            "gca",
            "sota",
            "ga",
            "saves",
            "savepct",
            "cs",
            "psxg",
        ],
    ].reset_index(drop=True)
    team_match_log = team_match_log.loc[
        (team_match_log["comp"] == "Premier League")
        & (team_match_log["season"] >= parameters["start_year"]),
        [
            "team",
            "date",
            "opponent",
            "poss",
            "gf",
            "ga",
            "xg",
            "xga",
        ],
    ].reset_index(drop=True)
    return player_match_log, team_match_log


def combine_data(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    elo_data: pd.DataFrame,
    fpl_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    combined_data = pd.merge(
        player_match_log,
        team_match_log,
        on=["team", "opponent", "date"],
        how="inner",
    )
    combined_data = pd.merge(
        combined_data,
        elo_data,
        left_on=["team", "date"],
        right_on=["team", "next_match"],
        how="left",
    )
    combined_data = pd.merge(
        combined_data,
        elo_data,
        left_on=["opponent", "date"],
        right_on=["team", "next_match"],
        how="left",
        suffixes=("", "_opp"),
    )
    combined_data = pd.merge(
        combined_data,
        fpl_data,
        left_on=["date", "player"],
        right_on=["date", "name"],
        how="left",
    )
    combined_data = combined_data.drop("next_match", axis=1)

    return combined_data


def calculate_score_from_odds(df):
    h_score_sum = (df["h_score"] * (1 / (df["odds"] + 1))).sum()
    a_score_sum = (df["a_score"] * (1 / (df["odds"] + 1))).sum()
    inverse_odds_sum = (1 / (df["odds"] + 1)).sum()
    return pd.Series(
        {
            "h_odds_2_score": h_score_sum / inverse_odds_sum,
            "a_odds_2_score": a_score_sum / inverse_odds_sum,
        }
    )


def aggregate_odds_data(odds_data, parameters):
    odds_data = odds_data.loc[odds_data["season"] >= parameters["start_year"]]

    agg_odds_data = (
        odds_data.groupby(["season", "h_team", "a_team"])
        .apply(lambda group: calculate_score_from_odds(group))
        .reset_index()
    )
    mapping = pd.read_csv("./src/fpl/pipelines/model_pipeline/team_mapping.csv")
    mapping = mapping.set_index("odds_portal_name")["fbref_name"].to_dict()
    agg_odds_data["h_team"] = agg_odds_data["h_team"].map(mapping)
    agg_odds_data["a_team"] = agg_odds_data["a_team"].map(mapping)

    temp_df = agg_odds_data.copy()
    temp_df.columns = [
        "season",
        "team",
        "opponent",
        "team_odds_2_score",
        "opponent_odds_2_score",
    ]

    # Create two copies of the temporary dataframe, one for home matches and one for away matches
    home_df = temp_df.copy()
    away_df = temp_df.copy()

    # Add a 'venue' column to each dataframe
    home_df["venue"] = "Home"
    away_df["venue"] = "Away"

    # Swap the 'team' and 'opponent' columns in the away_df
    away_df["team"], away_df["opponent"] = away_df["opponent"], away_df["team"]
    away_df["team_odds_2_score"], away_df["opponent_odds_2_score"] = (
        away_df["opponent_odds_2_score"],
        away_df["team_odds_2_score"],
    )

    # Concatenate the two dataframes to get the final unpivoted dataframe
    unpivoted_df = pd.concat([home_df, away_df], ignore_index=True)

    return unpivoted_df


def add_unplayed_matches(player_match_log: pd.DataFrame):
    output_data = pd.DataFrame()
    for season, season_data in player_match_log.groupby("season"):
        player_round = pd.DataFrame(
            list(
                itertools.product(
                    [season],
                    season_data["player"].unique(),
                    season_data["round"].dropna().unique(),
                )
            ),
            columns=["season", "player", "round"],
        )
        fill_dates = player_round.merge(
            season_data, on=["season", "player", "round"], how="left"
        )
        fill_dates = fill_dates.sort_values("date")
        output_data = pd.concat([output_data, fill_dates])
    output_data = output_data.reset_index(drop=True)
    return output_data


def align_data_structure(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    elo_data: pd.DataFrame,
    fpl_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_match_log = add_unplayed_matches(player_match_log)
    player_match_log = player_match_log.sort_values(
        ["date", "squad", "player"]
    ).reset_index(drop=True)
    player_match_log = player_match_log.rename(
        columns={"squad": "team"}, errors="raise"
    )

    team_match_log = team_match_log.sort_values(["date", "team"]).reset_index(drop=True)
    team_match_log = team_match_log.rename(
        columns={
            "poss": "team_poss",
            "gf": "team_gf",
            "ga": "team_ga",
            "xg": "team_xg",
            "xga": "team_xga",
        },
        errors="raise",
    )

    # ELO DATA stores elo ratings AFTER playing the game on that DATE.
    # This shifts ratings back to BEFORE playing the game.
    elo_data["next_match"] = elo_data.groupby("team")["date"].shift(-1)
    elo_data = elo_data.drop(["date", "season"], axis=1)

    fpl_data = fpl_data[["date", "name", "fpl_points", "value"]]

    return player_match_log, team_match_log, elo_data, fpl_data


def get_week_number(round_str: str) -> int:
    if re.match(r"Matchweek \d+", round_str):
        return int(re.findall(r"Matchweek (\d+)", round_str)[0])
    else:
        return None


def ensure_proper_dtypes(
    player_match_log: pd.DataFrame,
    team_match_log: pd.DataFrame,
    elo_data: pd.DataFrame,
    fpl_data: pd.DataFrame,
) -> list[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    player_match_log["date"] = pd.to_datetime(player_match_log["date"])
    player_match_log["round"] = player_match_log["round"].apply(get_week_number)
    player_match_log["start"] = player_match_log["start"].replace({"Y": 1, "N": 0})

    team_match_log["date"] = pd.to_datetime(team_match_log["date"])
    team_match_log["round"] = team_match_log["round"].apply(get_week_number)

    elo_data["date"] = pd.to_datetime(elo_data["date"])

    fpl_data["date"] = pd.to_datetime(fpl_data["kickoff_time"].str[:10])
    fpl_data["fpl_points"] = fpl_data["total_points"]

    return player_match_log, team_match_log, elo_data, fpl_data


def impute_missing_values():
    pass


def clean_data(player_match_log, team_match_log, elo_data, fpl_data, parameters):
    player_match_log, team_match_log, elo_data, fpl_data = ensure_proper_dtypes(
        player_match_log, team_match_log, elo_data, fpl_data
    )
    player_match_log, team_match_log = filter_data(
        player_match_log, team_match_log, parameters
    )
    player_match_log, team_match_log, elo_data, fpl_data = align_data_structure(
        player_match_log, team_match_log, elo_data, fpl_data, parameters
    )
    combined_data = combine_data(
        player_match_log, team_match_log, elo_data, fpl_data, parameters
    )
    return combined_data


def agg_home_away_elo(data: pd.DataFrame) -> pd.DataFrame:
    data["att_total"] = data.att_elo + data.def_elo_opp
    data["home_att_total"] = data.home_att_elo + data.away_def_elo_opp
    data["away_att_total"] = data.away_att_elo + data.home_def_elo_opp
    data["def_total"] = data.def_elo + data.att_elo_opp
    data["home_def_total"] = data.home_def_elo + data.away_att_elo_opp
    data["away_def_total"] = data.away_def_elo + data.home_att_elo_opp
    data = data.drop(data.filter(regex="_elo$").columns, axis=1)
    data = data.drop(data.filter(regex="_opp$").columns, axis=1)
    return data


def calculate_points(row):
    if row["team_gf"] > row["team_ga"]:
        return 3
    elif row["team_gf"] == row["team_ga"]:
        return 1
    else:
        return 0


def calculate_daily_rank(date_data: pd.DataFrame) -> pd.Series:
    date_data = date_data.sort_values(["pts_b4_match", "team"], ascending=[False, True])
    date_data["rank_b4_match"] = date_data.reset_index().index + 1
    return date_data


def features_from_pts(points_data: pd.DataFrame) -> pd.DataFrame:
    output_data = pd.DataFrame(
        columns=[
            "season",
            "team",
            "date",
            "pts_b4_match",
            "rank_b4_match",
            "pts_gap_above",
            "pts_gap_below",
        ]
    )
    for season, season_data in points_data.groupby("season"):
        team_date = pd.DataFrame(
            list(
                itertools.product(
                    [season], season_data["team"].unique(), season_data["date"].unique()
                )
            ),
            columns=["season", "team", "date"],
        )
        season_data["matchday"] = True
        fill_dates = team_date.merge(
            season_data, on=["season", "team", "date"], how="left"
        )

        fill_dates = fill_dates.sort_values("date")
        fill_dates["pts_b4_match"] = (
            fill_dates.groupby("team")["pts_b4_match"].ffill().fillna(0)
        )
        fill_dates = (
            fill_dates.groupby("date")
            .apply(calculate_daily_rank)
            .reset_index(drop=True)
        )

        fill_dates = fill_dates.sort_values(["date", "pts_b4_match"], ascending=False)
        fill_dates["pts_gap_above"] = (
            fill_dates.groupby("date")["pts_b4_match"].shift(1)
            - fill_dates["pts_b4_match"]
        )
        fill_dates["pts_gap_below"] = fill_dates["pts_b4_match"] - fill_dates.groupby(
            "date"
        )["pts_b4_match"].shift(-1)
        fill_dates["pts_gap_above"] = fill_dates.groupby("date")[
            "pts_gap_above"
        ].transform(lambda x: x.fillna(x.mean()))
        fill_dates["pts_gap_below"] = fill_dates.groupby("date")[
            "pts_gap_below"
        ].transform(lambda x: x.fillna(x.mean()))
        fill_dates = fill_dates.loc[fill_dates["matchday"] == True].drop(
            "matchday", axis=1
        )
        output_data = pd.concat([output_data, fill_dates])

    output_data = output_data.reset_index(drop=True)
    return output_data


def calculate_pts_data(data: pd.DataFrame) -> pd.DataFrame:
    pts_data = data.copy().drop_duplicates(["season", "team", "date"])

    pts_data = pts_data[["season", "team", "date", "team_gf", "team_ga"]]

    pts_data["match_points"] = pts_data.apply(calculate_points, axis=1)
    pts_data = pts_data.drop(columns=["team_gf", "team_ga"])
    pts_data["league_points"] = (
        pts_data.groupby(["season", "team"])["match_points"]
    ).shift(1)
    pts_data["pts_b4_match"] = (
        pts_data.groupby(["season", "team"])["league_points"].cumsum().fillna(0)
    )

    pts_data = features_from_pts(pts_data)

    return pts_data


def create_lag_features(df: pd.DataFrame, match_stat_col: str, lag: int, drop=True):
    df = df.sort_values(by=["player", "date"])

    # Create lag features
    for i in range(1, lag + 1):
        shifted = df.groupby("player")[match_stat_col].shift(i)
        date_diff = df["date"] - df.groupby("player")["date"].shift(i)
        within_one_year = date_diff <= timedelta(days=365)
        df[match_stat_col + "_" + str(i)] = shifted.where(within_one_year, None)

    # Drop the original column
    if drop:
        df = df.drop(columns=[match_stat_col])
    return df

def create_weighted_moving_average(df: pd.DataFrame, match_stat_col: str, opp_strength_col: str, lag: int, drop=True):
    df = df.sort_values(by=["player", "date"])

    # Create a new column which is the product of the stat and the opponent strength
    df[match_stat_col + "_weighted"] = df[match_stat_col] * df[opp_strength_col]

    # Create weighted moving average
    df[match_stat_col + "_wma"] = df.groupby("player") \
        .apply(lambda x: calculate_wma(x, match_stat_col, opp_strength_col, lag)) \
        .reset_index(level=0, drop=True)

    # Drop the original column and the intermediate weighted column
    df = df.drop(columns=[match_stat_col + "_weighted"])
    if drop:
        df = df.drop(columns=[match_stat_col])
    return df

def calculate_wma(group, match_stat_col, opp_strength_col, lag):
    wma = []
    for i in range(len(group)):
        if i >= lag:
            date_diff = group["date"].iloc[i] - group["date"].iloc[i-lag]
            if date_diff <= timedelta(days=365):
                wma_val = group[match_stat_col + "_weighted"].iloc[i-lag:i].sum() / group[opp_strength_col].iloc[i-lag:i].sum()
            else:
                wma_val = None
        else:
            wma_val = None
        wma.append(wma_val)
    return pd.Series(wma, index=group.index)


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data = agg_home_away_elo(data)

    pts_data = calculate_pts_data(data)
    data = data.merge(pts_data, on=["season", "team", "date"])

    data = create_weighted_moving_average(data, "touches", "att_total", 2)
    data = create_lag_features(data, "value", 2)
    data = create_lag_features(data, "xg", 2)
    data = create_lag_features(data, "xag", 2)
    data = create_lag_features(data, "fpl_points", 2, drop=False)
    return data
