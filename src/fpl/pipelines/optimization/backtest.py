import logging
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from fpl.pipelines.modelling.modelling.ensemble import EnsembleModel
from fpl.pipelines.modelling.modelling.evaluation import evaluate_model_holdout
from fpl.pipelines.optimization.data_classes import TYPE_DATA, LpData
from fpl.pipelines.optimization.lp_constructor import construct_lp
from fpl.pipelines.optimization.optimizer import prepare_data, solve_lp
from fpl.pipelines.optimization.output_formatting import (
    generate_outputs,
    get_gw_results,
    plot_backtest_results,
)
from fpl.pipelines.preprocessing.elo_calculation import calculate_elo_score
from fpl.pipelines.preprocessing.feature_engineering import agg_home_away_elo
from fpl.pipelines.preprocessing.imputer import impute_missing_values
from fpl.pipelines.preprocessing.preprocessor import merge_with_elo_data, split_data

logger = logging.getLogger(__name__)


def fpl_data_to_elements_data(fpl_data):
    elements_data = fpl_data[
        ["element", "full_name", "team_name", "position", "value"]
    ].drop_duplicates()
    elements_data["web_name"] = elements_data["full_name"]
    elements_data["element_type"] = elements_data["position"].map(
        {d["singular_name_short"]: d["id"] for d in TYPE_DATA}
    )
    elements_data["value"] = elements_data["value"].astype(int)
    elements_data = elements_data.rename(
        columns={"element": "id", "team_name": "team", "value": "now_cost"}
    )
    elements_data["chance_of_playing_next_round"] = 100
    return elements_data


def convert_to_live_data(
    fpl_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    elements_data = fpl_data_to_elements_data(fpl_data)
    team_data = (
        elements_data[["team"]]
        .drop_duplicates()
        .reset_index(drop=True)
        .rename(columns={"team": "name"})
    )
    type_data = pd.DataFrame(TYPE_DATA).set_index("id")

    return elements_data, team_data, type_data


def get_dummy_squad(
    fpl_data: pd.DataFrame,
):
    elements_data, _, type_data = convert_to_live_data(fpl_data)
    elements_data = elements_data.sort_values("now_cost", ascending=False)
    squad = []
    for type, row in type_data.iterrows():
        select = row["squad_select"]
        squad.extend(
            elements_data.loc[elements_data["element_type"] == type, "id"]
            .head(select)
            .to_list()
        )
    return squad


def dummy_starting_config(fpl_data: pd.DataFrame, backtest_season: str):
    min_week = 18
    max_week = 22
    in_the_bank = 0
    free_transfers = 5
    initial_squad = get_dummy_squad(
        fpl_data.loc[
            (fpl_data["season"] == backtest_season) & (fpl_data["round"] == min_week)
        ]
    )
    initial_squad[-1] = 108
    initial_squad[-2] = 355
    transfer_data = [
        {"element_in": 108, "element_in_cost": 50},
        {"element_in": 355, "element_in_cost": 50},
    ]
    return min_week, max_week, in_the_bank, free_transfers, initial_squad, transfer_data


def drop_future_data(
    processed_data, numerical_features, backtest_season, start_week, end_week
):
    five_matches_ago = processed_data.groupby("fpl_name")["date"].shift(5)
    past_five_matches_not_within_one_year = (
        processed_data["date"] - five_matches_ago
    ) > pd.Timedelta(days=365)
    previous_rounds = (processed_data["season"] == backtest_season) & (
        processed_data["round"] <= start_week
    )
    future_rounds = (processed_data["season"] == backtest_season) & (
        processed_data["round"] > start_week
    )
    snapshot_data = (
        processed_data.loc[
            (processed_data["season"] == backtest_season)
            & (processed_data["round"] >= start_week)
            & (processed_data["round"] <= end_week)
        ]
        .sort_values(["fpl_name", "round"])
        .copy()
    )

    all_cols = [col for col in numerical_features if col not in ("round", "value")]
    ma_cols = [col for col in all_cols if "_ma" in col]
    snapshot_data.loc[
        past_five_matches_not_within_one_year & previous_rounds, ma_cols
    ] = pd.NA
    snapshot_data.loc[future_rounds, all_cols] = pd.NA
    snapshot_data["value"] = (
        snapshot_data.sort_values("round")
        .groupby("fpl_name")["value"]
        .transform(lambda x: x.iloc[0])
    )  # Some player only have values in future weeks. Assume it is the same as current week
    return snapshot_data


def convert_to_team_match_log(
    snapshot_data: pd.DataFrame, fpl_2_fbref_team_mapping: dict[str, str]
):
    team_match_log = snapshot_data[
        ["team", "date", "opponent", "team_xg", "team_xga", "team_gf", "team_ga"]
    ].drop_duplicates()
    team_match_log["team"] = team_match_log["team"].map(fpl_2_fbref_team_mapping)
    team_match_log["opponent"] = team_match_log["opponent"].map(
        fpl_2_fbref_team_mapping
    )

    return team_match_log


def get_snapshot_data(
    processed_data: pd.DataFrame,
    elo_data: pd.DataFrame,
    backtest_season: str,
    start_week: int,
    end_week: int,
    fpl_2_fbref_team_mapping: dict[str, str],
    model_params: dict[str, Any],
    data_params: dict[str, Any],
):
    numerical_features = model_params["numerical_features"]
    snapshot_data = drop_future_data(
        processed_data, numerical_features, backtest_season, start_week, end_week
    )
    snapshot_data[["team_xg", "team_xga", "team_gf", "team_ga"]] = pd.NA
    snapshot_team_match_log = convert_to_team_match_log(
        snapshot_data, fpl_2_fbref_team_mapping
    )
    elo_data = elo_data.loc[elo_data["date"] < snapshot_data["date"].min()]
    data_params["use_cache"] = True
    elo_data = calculate_elo_score(
        snapshot_team_match_log, snapshot_data, elo_data, data_params
    )
    snapshot_data = merge_with_elo_data(snapshot_data, elo_data)
    snapshot_data = agg_home_away_elo(snapshot_data)
    return snapshot_data


def snapshot_inference(
    train_val_data: pd.DataFrame,
    model: EnsembleModel,
    sklearn_pipeline: Pipeline,
    processed_data: pd.DataFrame,
    elo_data: pd.DataFrame,
    backtest_season: str,
    start_week: int,
    end_week: int,
    fpl_2_fbref_team_mapping: dict[str, str],
    data_params: dict[str, Any],
    model_params: dict[str, Any],
) -> pd.DataFrame:
    snapshot_data = get_snapshot_data(
        processed_data,
        elo_data,
        backtest_season,
        start_week,
        end_week,
        fpl_2_fbref_team_mapping,
        model_params,
        data_params,
    )
    snapshot_data = impute_missing_values(snapshot_data, data_params, model_params)
    _, snapshot_data = split_data(snapshot_data, data_params, model_params)
    snapshot_inference_results, _, _ = evaluate_model_holdout(
        train_val_data, snapshot_data, model, sklearn_pipeline, 1, "", model_params
    )
    return snapshot_inference_results


def backtest(
    experiment_id: int,
    processed_data: pd.DataFrame,
    elo_data: pd.DataFrame,
    train_val_data: pd.DataFrame,
    model: EnsembleModel,
    sklearn_pipeline: Pipeline,
    fpl_data: pd.DataFrame,
    fpl_2_fbref_team_mapping: dict[str, str],
    optimization_params: dict,
    data_params: dict,
    model_params: dict,
) -> tuple[tuple[int, dict[str, float]], float]:

    backtest_season = optimization_params["backtest_season"]
    horizon = optimization_params["horizon"]
    transfer_horizon = optimization_params["transfer_horizon"]
    if transfer_horizon > horizon:
        logger.error("Transfer horizon cannot be greater than the horizon.")
        metrics = (experiment_id, {"total_actual_points": 0})
        return metrics, 0

    in_scope = fpl_data.loc[
        (fpl_data["season"] == backtest_season) & (~fpl_data["total_points"].isna()),
        "round",
    ].astype(int)
    min_week = in_scope.min()
    max_week = in_scope.max()
    initial_squad = []
    transfer_data = []
    in_the_bank = 100
    free_transfers = 1
    backtest_results = []
    total_actual_points = 0

    for start_week in range(min_week, max_week + 1):
        end_week = min(max_week, start_week + horizon - 1)
        gameweeks = [i for i in range(start_week, end_week + 1)]
        elements_data, team_data, type_data = convert_to_live_data(
            fpl_data.loc[
                (fpl_data["season"] == backtest_season)
                & (fpl_data["round"] == start_week)
            ]
        )
        backup_fpl_data = fpl_data.loc[
            (fpl_data["season"] == backtest_season) & (fpl_data["round"] < start_week)
        ].drop_duplicates(subset=["element"], keep="last")
        snapshot_inference_results = snapshot_inference(
            train_val_data,
            model,
            sklearn_pipeline,
            processed_data,
            elo_data,
            backtest_season,
            start_week,
            end_week,
            fpl_2_fbref_team_mapping,
            data_params,
            model_params,
        )
        merged_data = prepare_data(
            snapshot_inference_results,
            elements_data,
            backtest_season,
            gameweeks,
            transfer_data,
            initial_squad,
            backup_fpl_data,
        )

        logger.info(
            f"Optimizing for {horizon} weeks. {gameweeks}. Making transfers for {transfer_horizon} weeks."
        )
        logger.info(f"{in_the_bank = }    {free_transfers = }")

        lp_data = LpData(
            merged_data=merged_data,
            team_data=team_data,
            type_data=type_data,
            gameweeks=gameweeks,
            initial_squad=initial_squad,
            team_name="Backtest Strategy",
            in_the_bank=in_the_bank,
            free_transfers=free_transfers,
            current_season=backtest_season,
        )
        optimization_params["model_name"] = f"backtest_model"
        lp_keys, lp_variables, variable_sums = construct_lp(
            lp_data, optimization_params
        )
        lp_variables, variable_sums, solution_time = solve_lp(
            lp_data, lp_variables, variable_sums, optimization_params
        )
        generate_outputs(lp_data, lp_variables, solution_time, optimization_params)
        next_gw_results = get_gw_results(
            start_week,
            lp_data,
            lp_keys,
            lp_variables,
            variable_sums,
            solution_time,
            previous_squad=initial_squad,
        )
        in_the_bank = next_gw_results.in_the_bank
        free_transfers = next_gw_results.free_transfers
        initial_squad = next_gw_results.lineup + list(next_gw_results.bench.values())
        transfer_data.extend(next_gw_results.transfer_data)
        backtest_results.append(next_gw_results)
        total_actual_points += next_gw_results.total_actual_points
        if start_week % 5 == 0 or start_week == end_week:
            title = f"Backtest {backtest_season} h={horizon} th={transfer_horizon} --> pts={int(total_actual_points)}"
            plot = plot_backtest_results(backtest_results, title)
            plot_name = (
                "plot_backtest_tmp"
                if start_week != max_week
                else title.replace("--> ", "")
            )
            plot.savefig(f"data/optimization/backtest_results/{plot_name}.png")

    metrics = (experiment_id, {"total_actual_points": total_actual_points})
    return metrics, total_actual_points


if __name__ == "__main__":
    pass
