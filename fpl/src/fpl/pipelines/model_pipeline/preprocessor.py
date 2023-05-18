import pandas as pd


def _combine_data(team_match_log, elo_data, odds_data, parameters):
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
            "days_till_next",
            "days_since_last",
        ],
    ].reset_index(drop=True)
    team_match_log["date"] = team_match_log["date"].dt.strftime("%Y-%m-%d")

    # ELO DATA stores elo ratings AFTER playing the game on that DATE.
    # This shifts ratings back to BEFORE playing the game.
    elo_data["next_match"] = elo_data.groupby("team")["date"].shift(-1)
    elo_data = elo_data.drop("date", axis=1)

    # Combine team's and opponent's ELO Scores to get total scores
    combined_data = pd.merge(
        team_match_log,
        elo_data,
        left_on=["team", "season", "date"],
        right_on=["team", "season", "next_match"],
        how="left",
    )
    combined_data = pd.merge(
        combined_data,
        elo_data,
        left_on=["opponent", "season", "date"],
        right_on=["team", "season", "next_match"],
        how="left",
        suffixes=("", "_opp"),
    )
    combined_data["att_total"] = combined_data.att_elo + combined_data.def_elo_opp
    combined_data.loc[combined_data.venue == "Home", "home_att_total"] = (
        combined_data.home_att_elo + combined_data.away_def_elo_opp
    )
    combined_data.loc[combined_data.venue == "Away", "away_att_total"] = (
        combined_data.away_att_elo + combined_data.home_def_elo_opp
    )
    combined_data["def_total"] = combined_data.def_elo + combined_data.att_elo_opp
    combined_data.loc[combined_data.venue == "Home", "home_def_total"] = (
        combined_data.home_def_elo + combined_data.away_att_elo_opp
    )
    combined_data.loc[combined_data.venue == "Away", "away_def_total"] = (
        combined_data.away_def_elo + combined_data.home_att_elo_opp
    )
    combined_data = combined_data.drop(
        combined_data.filter(regex="_elo$").columns, axis=1
    )
    combined_data = combined_data.drop(
        combined_data.filter(regex="_opp$").columns, axis=1
    )
    combined_data = combined_data.drop("next_match", axis=1)

    combined_data = pd.merge(
        combined_data, odds_data, on=["season", "team", "opponent", "venue"]
    )

    return combined_data


def _weighted_average(df):
    h_score_sum = (df["h_score"] * (1 / (df["odds"] + 1))).sum()
    a_score_sum = (df["a_score"] * (1 / (df["odds"] + 1))).sum()
    inverse_odds_sum = (1 / (df["odds"] + 1)).sum()
    return pd.Series(
        {
            "h_odds_2_score": h_score_sum / inverse_odds_sum,
            "a_odds_2_score": a_score_sum / inverse_odds_sum,
        }
    )


def _aggregate_odds_data(odds_data, parameters):
    odds_data = odds_data.loc[odds_data["season"] >= parameters["start_year"]]

    agg_odds_data = (
        odds_data.groupby(["season", "h_team", "a_team"])
        .apply(lambda group: _weighted_average(group))
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


def preprocess_data(team_match_log, elo_data, odds_data, parameters):
    odds_data = _aggregate_odds_data(odds_data, parameters)
    team_match_log = team_match_log.sort_values(["season", "date", "team"]).reset_index(
        drop=True
    )
    team_match_log["date"] = pd.to_datetime(team_match_log["date"])
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
    combined_df = _combine_data(team_match_log, elo_data, odds_data, parameters)
    combined_df["round"] = combined_df["round"].apply(lambda x: int(x.split()[-1]))
    combined_df["xg_ma"] = combined_df.groupby("team")["xg"].apply(
        lambda x: x.shift(1).rolling(window=5).mean()
    )
    combined_df["xga_ma"] = combined_df.groupby("team")["xga"].apply(
        lambda x: x.shift(1).rolling(window=5).mean()
    )
    combined_df = combined_df.reset_index().rename(columns={"index": "id"})
    return combined_df
