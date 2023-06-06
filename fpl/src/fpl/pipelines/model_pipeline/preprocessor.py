from typing import Any
import pandas as pd
import re
import itertools


def combine_data(
    team_match_log: pd.DataFrame,
    elo_data: pd.DataFrame,
    odds_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> pd.DataFrame:
    team_match_log = team_match_log.loc[
        (team_match_log["comp"] == "Premier League")
        & (team_match_log["season"] >= parameters["start_year"]),
        [
            "season",
            "team",
            "date",
            "round",
            "venue",
            "gf",
            "ga",
            "opponent",
            "xg",
            "xga",
            "poss",
        ],
    ].reset_index(drop=True)

    elo_data = elo_data.drop(["date", "season"], axis=1)

    # Combine team's and opponent's ELO Scores to get total scores
    combined_data = pd.merge(
        team_match_log,
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
    combined_data = combined_data.drop("next_match", axis=1)

    combined_data = (
        pd.merge(combined_data, odds_data, on=["season", "team", "opponent", "venue"])
        .reset_index()
        .rename(columns={"index": "id"})
    )

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


def align_data_structure(
    odds_data: pd.DataFrame,
    team_match_log: pd.DataFrame,
    elo_data: pd.DataFrame,
    parameters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    odds_data = aggregate_odds_data(odds_data, parameters)
    team_match_log = team_match_log.sort_values(["season", "date", "team"]).reset_index(
        drop=True
    )
    # ELO DATA stores elo ratings AFTER playing the game on that DATE.
    # This shifts ratings back to BEFORE playing the game.
    elo_data["next_match"] = elo_data.groupby("team")["date"].shift(-1)
    return odds_data, team_match_log, elo_data


def get_week_number(round_str: str) -> int:
    if re.match(r"Matchweek \d+", round_str):
        return int(re.findall(r"Matchweek (\d+)", round_str)[0])
    else:
        return None


def ensure_proper_dtypes(combined_data: pd.DataFrame) -> pd.DataFrame:
    combined_data["date"] = pd.to_datetime(combined_data["date"])
    combined_data["round"] = combined_data["round"].apply(get_week_number)
    return combined_data


def impute_missing_values():
    pass


def clean_data(team_match_log, elo_data, odds_data, parameters):
    odds_data, team_match_log, elo_data = align_data_structure(
        odds_data, team_match_log, elo_data, parameters
    )
    combined_data = combine_data(team_match_log, elo_data, odds_data, parameters)
    combined_data = ensure_proper_dtypes(combined_data)
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
    if row["gf"] > row["ga"]:
        return 3
    elif row["gf"] == row["ga"]:
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
    pts_data = data.copy()

    pts_data["points"] = pts_data.apply(calculate_points, axis=1)
    pts_data["total_points"] = (pts_data.groupby(["season", "team"])["points"]).shift(1)
    pts_data["pts_b4_match"] = (
        pts_data.groupby(["season", "team"])["total_points"].cumsum().fillna(0)
    )
    data["days_till_next"] = (
        (data.groupby("team")["date"].shift(-1) - data["date"])
        .apply(lambda x: x.days)
        .clip(upper=7)
    )
    columns_to_drop = [
        col
        for col in pts_data.columns
        if col not in ["date", "team", "pts_b4_match", "season"]
    ]
    pts_data = pts_data.drop(columns=columns_to_drop)

    pts_data = features_from_pts(pts_data)

    return pts_data


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    data = agg_home_away_elo(data)
    data["days_till_next"] = (
        (data.groupby("team")["date"].shift(-1) - data["date"])
        .apply(lambda x: x.days)
        .clip(upper=7)
    )

    pts_data = calculate_pts_data(data)
    data = data.merge(pts_data, on=["season", "team", "date"])
    data["days_since_last"] = (
        (data["date"] - data.groupby("team")["date"].shift(1))
        .apply(lambda x: x.days)
        .clip(upper=7)
    )
    data["xg_ma"] = data.groupby("team")["xg"].apply(
        lambda x: x.shift(1).rolling(window=5).mean()
    )
    data["xga_ma"] = data.groupby("team")["xga"].apply(
        lambda x: x.shift(1).rolling(window=5).mean()
    )
    return data
