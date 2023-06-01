from typing import Any
import pandas as pd
import re


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
    combined_data["att_total"] = combined_data.att_elo + combined_data.def_elo_opp
    combined_data["home_att_total"] = (
        combined_data.home_att_elo + combined_data.away_def_elo_opp
    )
    combined_data["away_att_total"] = (
        combined_data.away_att_elo + combined_data.home_def_elo_opp
    )
    combined_data["def_total"] = combined_data.def_elo + combined_data.att_elo_opp
    combined_data["home_def_total"] = (
        combined_data.home_def_elo + combined_data.away_att_elo_opp
    )
    combined_data["away_def_total"] = (
        combined_data.away_def_elo + combined_data.home_att_elo_opp
    )
    combined_data = combined_data.drop(
        combined_data.filter(regex="_elo$").columns, axis=1
    )
    combined_data = combined_data.drop(
        combined_data.filter(regex="_opp$").columns, axis=1
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


def feature_engineering(combined_data: pd.DataFrame) -> pd.DataFrame:
    combined_data["days_till_next"] = (
        (combined_data.groupby("team")["date"].shift(-1) - combined_data["date"])
        .apply(lambda x: x.days)
        .clip(upper=7)
    )
    combined_data["days_since_last"] = (
        (combined_data["date"] - combined_data.groupby("team")["date"].shift(1))
        .apply(lambda x: x.days)
        .clip(upper=7)
    )
    combined_data["xg_ma"] = combined_data.groupby("team")["xg"].apply(
        lambda x: x.shift(1).rolling(window=5).mean()
    )
    combined_data["xga_ma"] = combined_data.groupby("team")["xga"].apply(
        lambda x: x.shift(1).rolling(window=5).mean()
    )
    return combined_data
