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
    first_season = match_data.SEASON.min()
    home_xg_mean = (
        match_data.loc[match_data["SEASON"] == first_season]
        .groupby("TEAM")["XG"]
        .mean()
    )
    home_xga_mean = (
        match_data.loc[match_data["SEASON"] == first_season]
        .groupby("TEAM")["XGA"]
        .mean()
    )
    away_xg_mean = (
        match_data.loc[match_data["SEASON"] == first_season]
        .groupby("OPPONENT")["XGA"]
        .mean()
    )
    away_xga_mean = (
        match_data.loc[match_data["SEASON"] == first_season]
        .groupby("OPPONENT")["XG"]
        .mean()
    )
    xg_mean = (home_xg_mean + away_xg_mean) / 2
    xga_mean = (home_xga_mean + away_xga_mean) / 2
    first_match = match_data.DATE.min()
    before_first_match = pd.to_datetime(first_match) - pd.DateOffset(days=1)
    init_elo_df = pd.DataFrame(
        {
            "SEASON": first_season,
            "DATE": before_first_match.strftime("%Y-%m-%d"),
            "ATT_ELO": xg_mean,
            "DEF_ELO": xga_mean,
            "HOME_ATT_ELO": home_xg_mean,
            "HOME_DEF_ELO": home_xga_mean,
            "AWAY_ATT_ELO": away_xg_mean,
            "AWAY_DEF_ELO": away_xga_mean,
        }
    )
    init_elo_df.reset_index(inplace=True)
    init_elo_df.rename(columns={"index": "TEAM"}, inplace=True)
    return init_elo_df


def _get_promotions_relegations(match_data):
    seasons = sorted(match_data.SEASON.unique().tolist())
    promotions = {}
    relegations = {}
    for prev_season, curr_season in zip(seasons, seasons[1:]):
        prev_teams = match_data[match_data.SEASON == prev_season].TEAM.unique().tolist()
        curr_teams = match_data[match_data.SEASON == curr_season].TEAM.unique().tolist()
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
    same_team = elo_df.TEAM == team
    within_2_seasons = elo_df.SEASON.isin([season, prev_season])
    matched = elo_df.loc[same_team & within_2_seasons]
    elo_cols = [
        "ATT_ELO",
        "DEF_ELO",
        "HOME_ATT_ELO",
        "HOME_DEF_ELO",
        "AWAY_ATT_ELO",
        "AWAY_DEF_ELO",
    ]
    if len(matched) == 0:
        assert team in promotions[season]
        relegated_teams = elo_df.TEAM.isin(relegations[season])
        is_prev_season = elo_df.SEASON == prev_season
        matched = elo_df.loc[relegated_teams & is_prev_season].drop_duplicates(
            "TEAM", keep="last"
        )
        latest_elos = matched.loc[:, elo_cols].mean()
    else:
        latest_elos = matched.drop_duplicates("TEAM", keep="last").iloc[0]
        latest_elos = latest_elos.loc[elo_cols]
    return latest_elos


def calculate_elo_score(match_data, parameters):
    home_away_weight = parameters["home_away_weight"]
    learning_rate = parameters["elo_learning_rate"]
    logger.info(
        f"Calculating elo score with lr={learning_rate}; h/a_weight={home_away_weight}"
    )
    match_data = match_data.loc[
        (match_data["COMP"] == "Premier League")
        & (match_data["SEASON"] >= parameters["start_year"])
        & (match_data["VENUE"] == "Home"),
        ["SEASON", "TEAM", "ROUND", "DATE", "OPPONENT", "XG", "XGA"],
    ]
    match_data = match_data.sort_values(["SEASON", "DATE", "TEAM"]).reset_index(
        drop=True
    )
    elo_df = _get_init_elo(match_data)
    seasons, promotions, relegations = _get_promotions_relegations(match_data)
    for idx, row in tqdm(match_data.iterrows(), total=match_data.shape[0]):
        curr_idx = seasons.index(row.SEASON)
        prev_season = seasons[curr_idx - 1]
        home_team_elos = _get_latest_elos(
            elo_df, row.SEASON, prev_season, row.TEAM, promotions, relegations
        )
        away_team_elos = _get_latest_elos(
            elo_df, row.SEASON, prev_season, row.OPPONENT, promotions, relegations
        )

        expected_xg = _get_expected_xg(
            home_team_elos.ATT_ELO,
            home_team_elos.HOME_ATT_ELO,
            away_team_elos.DEF_ELO,
            away_team_elos.AWAY_DEF_ELO,
            home_away_weight,
        )
        expected_xga = _get_expected_xg(
            away_team_elos.ATT_ELO,
            away_team_elos.AWAY_ATT_ELO,
            home_team_elos.DEF_ELO,
            home_team_elos.HOME_DEF_ELO,
            home_away_weight,
        )

        next_home_team_elos = {
            "TEAM": row.TEAM,
            "SEASON": row.SEASON,
            "DATE": row.DATE,
            "ATT_ELO": _get_next_elo(
                home_team_elos.ATT_ELO, row.XG, expected_xg, learning_rate
            ),
            "DEF_ELO": _get_next_elo(
                home_team_elos.DEF_ELO, row.XGA, expected_xga, learning_rate
            ),
            "HOME_ATT_ELO": _get_next_elo(
                home_team_elos.HOME_ATT_ELO, row.XG, expected_xg, learning_rate
            ),
            "HOME_DEF_ELO": _get_next_elo(
                home_team_elos.HOME_DEF_ELO, row.XGA, expected_xga, learning_rate
            ),
            "AWAY_ATT_ELO": home_team_elos.AWAY_ATT_ELO,
            "AWAY_DEF_ELO": home_team_elos.AWAY_DEF_ELO,
        }
        next_home_team_elos = pd.DataFrame(next_home_team_elos, index=[0])
        elo_df = pd.concat([elo_df, next_home_team_elos], ignore_index=True)

        next_away_team_elos = {
            "TEAM": row.OPPONENT,
            "SEASON": row.SEASON,
            "DATE": row.DATE,
            "ATT_ELO": _get_next_elo(
                away_team_elos.ATT_ELO, row.XGA, expected_xga, learning_rate
            ),
            "DEF_ELO": _get_next_elo(
                away_team_elos.DEF_ELO, row.XG, expected_xg, learning_rate
            ),
            "HOME_ATT_ELO": away_team_elos.HOME_ATT_ELO,
            "HOME_DEF_ELO": away_team_elos.HOME_DEF_ELO,
            "AWAY_ATT_ELO": _get_next_elo(
                away_team_elos.AWAY_ATT_ELO, row.XGA, expected_xga, learning_rate
            ),
            "AWAY_DEF_ELO": _get_next_elo(
                away_team_elos.AWAY_DEF_ELO, row.XG, expected_xg, learning_rate
            ),
        }
        next_away_team_elos = pd.DataFrame(next_away_team_elos, index=[0])
        elo_df = pd.concat([elo_df, next_away_team_elos], ignore_index=True)

    return elo_df


def _combine_data(team_match_log, elo_data, odds_data, parameters):
    team_match_log = team_match_log.loc[
        (team_match_log["COMP"] == "Premier League")
        & (team_match_log["SEASON"] >= parameters["start_year"]),
        [
            "SEASON",
            "TEAM",
            "DATE",
            "ROUND",
            "VENUE",
            "GF",
            "GA",
            "OPPONENT",
            "XG",
            "XGA",
            "POSS",
            "DAYS_TILL_NEXT",
            "DAYS_SINCE_LAST",
        ],
    ].reset_index(drop=True)
    team_match_log["DATE"] = team_match_log["DATE"].dt.strftime("%Y-%m-%d")

    # ELO DATA stores elo ratings AFTER playing the game on that DATE.
    # This shifts ratings back to BEFORE playing the game.
    elo_data["NEXT_MATCH"] = elo_data.groupby("TEAM")["DATE"].shift(-1)
    elo_data = elo_data.drop("DATE", axis=1)

    # Combine team's and opponent's ELO Scores to get total scores
    combined_data = pd.merge(
        team_match_log,
        elo_data,
        left_on=["TEAM", "SEASON", "DATE"],
        right_on=["TEAM", "SEASON", "NEXT_MATCH"],
        how="left",
    )
    combined_data = pd.merge(
        combined_data,
        elo_data,
        left_on=["OPPONENT", "SEASON", "DATE"],
        right_on=["TEAM", "SEASON", "NEXT_MATCH"],
        how="left",
        suffixes=("", "_OPP"),
    )
    combined_data["ATT_TOTAL"] = combined_data.ATT_ELO + combined_data.DEF_ELO_OPP
    combined_data.loc[combined_data.VENUE == "Home", "HOME_ATT_TOTAL"] = (
        combined_data.HOME_ATT_ELO + combined_data.AWAY_DEF_ELO_OPP
    )
    combined_data.loc[combined_data.VENUE == "Away", "AWAY_ATT_TOTAL"] = (
        combined_data.AWAY_ATT_ELO + combined_data.HOME_DEF_ELO_OPP
    )
    combined_data["DEF_TOTAL"] = combined_data.DEF_ELO + combined_data.ATT_ELO_OPP
    combined_data.loc[combined_data.VENUE == "Home", "HOME_DEF_TOTAL"] = (
        combined_data.HOME_DEF_ELO + combined_data.AWAY_ATT_ELO_OPP
    )
    combined_data.loc[combined_data.VENUE == "Away", "AWAY_DEF_TOTAL"] = (
        combined_data.AWAY_DEF_ELO + combined_data.HOME_ATT_ELO_OPP
    )
    combined_data = combined_data.drop(
        combined_data.filter(regex="_ELO$").columns, axis=1
    )
    combined_data = combined_data.drop(
        combined_data.filter(regex="_OPP$").columns, axis=1
    )
    combined_data = combined_data.drop("NEXT_MATCH", axis=1)

    combined_data = pd.merge(
        combined_data, odds_data, on=["SEASON", "TEAM", "OPPONENT", "VENUE"]
    )

    return combined_data


def _weighted_average(df):
    h_score_sum = (df["H_SCORE"] * (1 / (df["ODDS"] + 1))).sum()
    a_score_sum = (df["A_SCORE"] * (1 / (df["ODDS"] + 1))).sum()
    inverse_odds_sum = (1 / (df["ODDS"] + 1)).sum()
    return pd.Series(
        {
            "H_ODDS_2_SCORE": h_score_sum / inverse_odds_sum,
            "A_ODDS_2_SCORE": a_score_sum / inverse_odds_sum,
        }
    )


def _aggregate_odds_data(odds_data, parameters):
    odds_data = odds_data.loc[odds_data["SEASON"] >= parameters["start_year"]]

    agg_odds_data = (
        odds_data.groupby(["SEASON", "H_TEAM", "A_TEAM"])
        .apply(lambda group: _weighted_average(group))
        .reset_index()
    )
    mapping = pd.read_csv("./src/fpl/pipelines/model_pipeline/team_mapping.csv")
    mapping = mapping.set_index("ODDS_PORTAL_NAME")["FBREF_NAME"].to_dict()
    agg_odds_data["H_TEAM"] = agg_odds_data["H_TEAM"].map(mapping)
    agg_odds_data["A_TEAM"] = agg_odds_data["A_TEAM"].map(mapping)

    temp_df = agg_odds_data.copy()
    temp_df.columns = [
        "SEASON",
        "TEAM",
        "OPPONENT",
        "TEAM_ODDS_2_SCORE",
        "OPPONENT_ODDS_2_SCORE",
    ]

    # Create two copies of the temporary dataframe, one for home matches and one for away matches
    home_df = temp_df.copy()
    away_df = temp_df.copy()

    # Add a 'venue' column to each dataframe
    home_df["VENUE"] = "Home"
    away_df["VENUE"] = "Away"

    # Swap the 'team' and 'opponent' columns in the away_df
    away_df["TEAM"], away_df["OPPONENT"] = away_df["OPPONENT"], away_df["TEAM"]
    away_df["TEAM_ODDS_2_SCORE"], away_df["OPPONENT_ODDS_2_SCORE"] = (
        away_df["OPPONENT_ODDS_2_SCORE"],
        away_df["TEAM_ODDS_2_SCORE"],
    )

    # Concatenate the two dataframes to get the final unpivoted dataframe
    unpivoted_df = pd.concat([home_df, away_df], ignore_index=True)

    return unpivoted_df


def preprocess_data(team_match_log, elo_data, odds_data, parameters):
    odds_data = _aggregate_odds_data(odds_data, parameters)
    team_match_log = team_match_log.sort_values(["SEASON", "DATE", "TEAM"]).reset_index(
        drop=True
    )
    team_match_log["DATE"] = pd.to_datetime(team_match_log["DATE"])
    team_match_log["DAYS_TILL_NEXT"] = (
        (team_match_log.groupby("TEAM")["DATE"].shift(-1) - team_match_log["DATE"])
        .apply(lambda x: x.days)
        .clip(upper=7)
    )
    team_match_log["DAYS_SINCE_LAST"] = (
        (team_match_log["DATE"] - team_match_log.groupby("TEAM")["DATE"].shift(1))
        .apply(lambda x: x.days)
        .clip(upper=7)
    )
    combined_df = _combine_data(team_match_log, elo_data, odds_data, parameters)
    combined_df["ROUND"] = combined_df["ROUND"].apply(lambda x: int(x.split()[-1]))
    combined_df["XG_MA"] = combined_df.groupby("TEAM")["XG"].apply(
        lambda x: x.shift(1).rolling(window=5).mean()
    )
    combined_df["XGA_MA"] = combined_df.groupby("TEAM")["XGA"].apply(
        lambda x: x.shift(1).rolling(window=5).mean()
    )

    return combined_df


def xg_elo_correlation(processed_data, parameters):
    att_corr = processed_data["XG"].corr(processed_data["ATT_TOTAL"])
    home_att_corr = processed_data["XG"].corr(processed_data["HOME_ATT_TOTAL"])
    away_att_corr = processed_data["XG"].corr(processed_data["AWAY_ATT_TOTAL"])
    def_corr = processed_data["XGA"].corr(processed_data["DEF_TOTAL"])
    home_def_corr = processed_data["XGA"].corr(processed_data["HOME_DEF_TOTAL"])
    away_def_corr = processed_data["XGA"].corr(processed_data["AWAY_DEF_TOTAL"])

    correlation = statistics.mean(
        [att_corr, home_att_corr, away_att_corr, def_corr, home_def_corr, away_def_corr]
    )
    logger.info(
        f"lr={parameters['elo_learning_rate']}; h/a_weight={parameters['home_away_weight']} ==> Mean Correlation = {correlation}"
    )
    return correlation


def split_data(processed_data, parameters):
    holdout_year = parameters["holdout_year"]
    train_val_data = processed_data[processed_data["SEASON"] != holdout_year]
    holdout_data = processed_data[processed_data["SEASON"] == holdout_year]

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
        ["index"]
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
    cursor.execute('SELECT DISTINCT start_time FROM "03_EVALUATION_RESULT"')
    unique_start_times = [row[0] for row in cursor.fetchall()]

    # Sort start_time values in descending order and select the most recent ones
    unique_start_times.sort(reverse=True)
    recent_start_times = unique_start_times[:to_keep]

    _delete_from_db('"03_EVALUATION_RESULT"', recent_start_times, conn)
    _delete_from_path("./data/03_evaluation", recent_start_times)

    conn.close()
    return loss


if __name__ == "__main__":
    pass
