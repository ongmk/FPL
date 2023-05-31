import pandas as pd
import logging
import statistics
from tqdm import tqdm

logger = logging.getLogger(__name__)


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


def xg_elo_correlation(processed_data: pd.DataFrame) -> float:
    att_corr = processed_data["xg"].corr(processed_data["att_total"])
    home_att_corr = processed_data["xg"].corr(processed_data["home_att_total"])
    away_att_corr = processed_data["xg"].corr(processed_data["away_att_total"])
    def_corr = processed_data["xga"].corr(processed_data["def_total"])
    home_def_corr = processed_data["xga"].corr(processed_data["home_def_total"])
    away_def_corr = processed_data["xga"].corr(processed_data["away_def_total"])

    correlation = statistics.mean(
        [att_corr, home_att_corr, away_att_corr, def_corr, home_def_corr, away_def_corr]
    )
    logger.info(f"Mean Correlation = {correlation}")
    return correlation
