import pandas as pd
from tqdm import tqdm
import numpy as np
import logging
import statistics
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
import os

color_pal = sns.color_palette()
plt.style.use("ggplot")

logger = logging.getLogger(__name__)


def get_start_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _get_init_elo(match_data):
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


def _get_promotions_relegations(match_data):
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


def _get_expected_xg(
    att_elo, home_away_att_elo, def_elo, home_away_def_elo, home_away_weight=0.20
):
    expected_xg = (1 - home_away_weight) * (att_elo + def_elo) + home_away_weight * (
        home_away_att_elo + home_away_def_elo
    )
    return expected_xg


def _get_next_elo(curr_elo, actual_xg, expected_xg, learning_rate=0.12):
    updated_elo = curr_elo + learning_rate * (actual_xg - expected_xg)
    return updated_elo


def _get_latest_elos(elo_df, season, prev_season, team, promotions, relegations):
    same_team = elo_df.team == team
    within_2_seasons = elo_df.season.isin([season, prev_season])
    matched = elo_df.loc[same_team & within_2_seasons]
    elo_cols = [
        "att_elo",
        "def_elo",
        "home_att_elo",
        "home_def_elo",
        "away_att_elo",
        "away_def_elo",
    ]
    if len(matched) == 0:
        assert team in promotions[season]
        relegated_teams = elo_df.team.isin(relegations[season])
        is_prev_season = elo_df.season == prev_season
        matched = elo_df.loc[relegated_teams & is_prev_season].drop_duplicates(
            "team", keep="last"
        )
        latest_elos = matched.loc[:, elo_cols].mean()
    else:
        latest_elos = matched.drop_duplicates("team", keep="last").iloc[0]
        latest_elos = latest_elos.loc[elo_cols]
    return latest_elos


def calculate_elo_score(match_data, parameters):
    home_away_weight = parameters["home_away_weight"]
    learning_rate = parameters["elo_learning_rate"]
    logger.info(
        f"Calculating elo score with lr={learning_rate}; h/a_weight={home_away_weight}"
    )
    match_data = match_data.loc[
        (match_data["comp"] == "Premier League")
        & (match_data["season"] >= parameters["start_year"])
        & (match_data["venue"] == "Home"),
        ["season", "team", "round", "date", "opponent", "xg", "xga"],
    ]
    match_data = match_data.sort_values(["season", "date", "team"]).reset_index(
        drop=True
    )
    elo_df = _get_init_elo(match_data)
    seasons, promotions, relegations = _get_promotions_relegations(match_data)
    for idx, row in tqdm(match_data.iterrows(), total=match_data.shape[0]):
        curr_idx = seasons.index(row.season)
        prev_season = seasons[curr_idx - 1]
        home_team_elos = _get_latest_elos(
            elo_df, row.season, prev_season, row.team, promotions, relegations
        )
        away_team_elos = _get_latest_elos(
            elo_df, row.season, prev_season, row.opponent, promotions, relegations
        )

        expected_xg = _get_expected_xg(
            home_team_elos.att_elo,
            home_team_elos.home_att_elo,
            away_team_elos.def_elo,
            away_team_elos.away_def_elo,
            home_away_weight,
        )
        expected_xga = _get_expected_xg(
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
            "att_elo": _get_next_elo(
                home_team_elos.att_elo, row.xg, expected_xg, learning_rate
            ),
            "def_elo": _get_next_elo(
                home_team_elos.def_elo, row.xga, expected_xga, learning_rate
            ),
            "home_att_elo": _get_next_elo(
                home_team_elos.home_att_elo, row.xg, expected_xg, learning_rate
            ),
            "home_def_elo": _get_next_elo(
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
            "att_elo": _get_next_elo(
                away_team_elos.att_elo, row.xga, expected_xga, learning_rate
            ),
            "def_elo": _get_next_elo(
                away_team_elos.def_elo, row.xg, expected_xg, learning_rate
            ),
            "home_att_elo": away_team_elos.home_att_elo,
            "home_def_elo": away_team_elos.home_def_elo,
            "away_att_elo": _get_next_elo(
                away_team_elos.away_att_elo, row.xga, expected_xga, learning_rate
            ),
            "away_def_elo": _get_next_elo(
                away_team_elos.away_def_elo, row.xg, expected_xg, learning_rate
            ),
        }
        next_away_team_elos = pd.DataFrame(next_away_team_elos, index=[0])
        elo_df = pd.concat([elo_df, next_away_team_elos], ignore_index=True)

    return elo_df


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


def xg_elo_correlation(processed_data, parameters):
    att_corr = processed_data["xg"].corr(processed_data["att_total"])
    home_att_corr = processed_data["xg"].corr(processed_data["home_att_total"])
    away_att_corr = processed_data["xg"].corr(processed_data["away_att_total"])
    def_corr = processed_data["xga"].corr(processed_data["def_total"])
    home_def_corr = processed_data["xga"].corr(processed_data["home_def_total"])
    away_def_corr = processed_data["xga"].corr(processed_data["away_def_total"])

    correlation = statistics.mean(
        [att_corr, home_att_corr, away_att_corr, def_corr, home_def_corr, away_def_corr]
    )
    logger.info(
        f"lr={parameters['elo_learning_rate']}; h/a_weight={parameters['home_away_weight']} ==> Mean Correlation = {correlation}"
    )
    return correlation


def split_data(processed_data, parameters):
    holdout_year = parameters["holdout_year"]
    train_val_data = processed_data[processed_data["season"] != holdout_year]
    holdout_data = processed_data[processed_data["season"] == holdout_year]

    return train_val_data, holdout_data


def _encode_features(X, categorical_features, numerical_features, encoder):
    X_cat = X[categorical_features]
    X_num = X[numerical_features]
    X_encoded = np.hstack([encoder.transform(X_cat).toarray(), X_num])
    return X_encoded


def train_model(train_val_data, parameters):
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    target = parameters["target"]

    X_train_val = train_val_data[numerical_features + categorical_features]
    y_train_val = train_val_data[target]
    groups = train_val_data[parameters["group_by"]]

    n_splits = groups.nunique()
    logger.info(f"{groups.unique() = }")
    group_kfold = GroupKFold(n_splits=n_splits)

    X_train_val_cat = X_train_val[categorical_features]
    categories = [
        np.append(X_train_val_cat[col].unique(), "Unknown")
        for col in X_train_val_cat.columns
    ]
    encoder = OneHotEncoder(
        handle_unknown="infrequent_if_exist", categories=categories, min_frequency=1
    )
    encoder.fit(X_train_val_cat)

    model = XGBRegressor(
        random_state=parameters["random_seed"],
        **parameters["xgboost_params"],
    )

    cross_val_scores = []
    for train_index, val_index in group_kfold.split(X_train_val, y_train_val, groups):
        # Remove outliers
        X_train, y_train = X_train_val.iloc[train_index], y_train_val.iloc[train_index]
        outlier = y_train > 4
        X_train, y_train = X_train.loc[~outlier], y_train.loc[~outlier]

        X_train_encoded = _encode_features(
            X_train, categorical_features, numerical_features, encoder
        )
        X_val_encoded = _encode_features(
            X_train_val.iloc[val_index],
            categorical_features,
            numerical_features,
            encoder,
        )
        y_val = y_train_val.iloc[val_index]

        model.fit(
            X_train_encoded,
            y_train,
            eval_set=[(X_train_encoded, y_train), (X_val_encoded, y_val)],
            verbose=100,
        )

        val_predictions = model.predict(X_val_encoded)
        val_accuracy = mean_squared_error(y_val, val_predictions)
        cross_val_scores.append(val_accuracy)

    avg_cv_accuracy = sum(cross_val_scores) / n_splits
    logger.info(f"Average cross-validation accuracy: {avg_cv_accuracy}")
    logger.info(cross_val_scores)
    return model, encoder


def _ordered_set(input_list):
    seen = set()
    return [x for x in input_list if not (x in seen or seen.add(x))]


def evaluate_model(holdout_data, model, encoder, start_time, parameters):
    categorical_features = parameters["categorical_features"]
    numerical_features = parameters["numerical_features"]
    target = parameters["target"]
    baseline_columns = parameters["baseline_columns"]
    output_plots = {}

    # Feature Importance
    encoded_cat_cols = encoder.get_feature_names_out(
        input_features=categorical_features
    )
    features_importance = pd.DataFrame(
        data=model.feature_importances_,
        index=encoded_cat_cols.tolist() + numerical_features,
        columns=["importance"],
    )
    features_importance = features_importance.sort_values(
        by="importance", ascending=False
    ).head(10)

    ax = features_importance.sort_values("importance").plot(
        kind="barh", title="Feature Importance"
    )
    output_plots[f"{start_time}__feature_importance.png"] = ax.get_figure()

    # Holdout set evaluation
    X_holdout = holdout_data[numerical_features + categorical_features]

    X_holdout_encoded = _encode_features(
        X_holdout, categorical_features, numerical_features, encoder
    )
    holdout_predictions = model.predict(X_holdout_encoded)

    output_cols = _ordered_set(
        ["id"]
        + numerical_features
        + categorical_features
        + [target]
        + baseline_columns
    )
    output_df = holdout_data[output_cols].copy()
    eval_cols = ["prediction"] + baseline_columns
    output_df["prediction"] = holdout_predictions

    fig, axes = plt.subplots(
        nrows=1, ncols=len(eval_cols), figsize=(20, 5), sharey=True
    )

    for i, col in enumerate(eval_cols):
        output_df[f"{col}_error"] = output_df[col] - output_df[target]
        output_df[f"{col}_error"].hist(
            ax=axes[i], bins=np.arange(-3.5, 3.5, 0.1), color=color_pal[i]
        )
        mae = output_df[f"{col}_error"].abs().mean()
        axes[i].set_title(f"{col} MAE: {mae:.2f}")
        axes[i].set_xlabel(f"{col}_error")
    output_df.head()
    score = output_df["prediction_error"].abs().mean()
    logger.info(f"Model MAE: {score}")
    plt.subplots_adjust(wspace=0.1)
    output_plots[f"{start_time}__errors.png"] = fig

    output_df["start_time"] = start_time
    columns = ["start_time"] + [col for col in output_df.columns if col != "start_time"]
    output_df = output_df[columns]
    plt.close("all")

    return output_df, output_plots, score


def _delete_from_db(table_name, recent_start_times, conn):
    cursor = conn.cursor()
    cursor.execute(
        f"DELETE FROM {table_name} WHERE start_time NOT IN ({','.join('?' * len(recent_start_times))})",
        recent_start_times,
    )
    conn.commit()
    return None


def _delete_from_path(relative_path, recent_start_times):
    file_list = os.listdir(relative_path)

    for file in file_list:
        timestamp = file.split("__")[0]

        if timestamp not in recent_start_times:
            file_path = os.path.join(relative_path, file)
            os.remove(file_path)
    return None


def run_housekeeping(loss, parameters):
    to_keep = parameters["to_keep"]

    conn = sqlite3.connect("./data/fpl.db")
    cursor = conn.cursor()

    # Get all unique start_time values
    cursor.execute('SELECT DISTINCT start_time FROM "evaluation_result"')
    unique_start_times = [row[0] for row in cursor.fetchall()]

    # Sort start_time values in descending order and select the most recent ones
    unique_start_times.sort(reverse=True)
    recent_start_times = unique_start_times[:to_keep]

    _delete_from_db('"evaluation_result"', recent_start_times, conn)
    _delete_from_path("./data/evaluation", recent_start_times)

    conn.close()
    return loss


if __name__ == "__main__":
    pass
