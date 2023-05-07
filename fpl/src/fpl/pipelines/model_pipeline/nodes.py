import pandas as pd
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
import statistics


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
        & (match_data["SEASON"] > "2017")
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


def preprocess_data(team_match_log, elo_data):
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
    team_match_log = team_match_log.loc[
        (team_match_log["COMP"] == "Premier League")
        & (team_match_log["SEASON"] > "2017"),
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
    combined_data["XG_MA"] = combined_data.groupby("TEAM")["XG"].apply(
        lambda x: x.shift(1).rolling(window=5).sum()
    )
    combined_data["XGA_MA"] = combined_data.groupby("TEAM")["XGA"].apply(
        lambda x: x.shift(1).rolling(window=5).sum()
    )

    return combined_data


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


def split_data():
    pass


def train_model():
    pass


def evaluate_model():
    pass


if __name__ == "__main__":
    pass
