import logging
from typing import Any

logger = logging.getLogger(__name__)
import itertools
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from fpl.pipelines.modelling.modelling.evaluation import evaluate_model


def process_dnp_data(data: pd.DataFrame, parameters: dict[str, Any]) -> pd.DataFrame:
    dnp_lookback = parameters["dnp_lookback"]
    cop = data.copy()

    logger.info("Processing data for Did-Not-Play prediction...")
    team_match_mins = (
        cop.groupby(["team", "season", "round", "date"])["minutes"].max().reset_index()
    )
    team_round_mins = (
        team_match_mins.groupby(["team", "season", "round"])["minutes"].sum().to_frame()
    ).rename(columns={"minutes": "mins_max"})
    player_round_mins = (
        cop.groupby(["fpl_name", "team", "season", "round", "pos"])["minutes"]
        .sum()
        .to_frame()
        .reset_index()
    )
    player_round_mins_filled = []
    for season, season_data in tqdm(
        player_round_mins.groupby("season"),
        desc="Processing season",
    ):
        for player, player_data in season_data.groupby("fpl_name"):
            pos = player_data["pos"].mode().values[0]
            player_team_rounds = pd.DataFrame(
                list(
                    itertools.product(
                        [player],
                        [pos],
                        player_data["team"].unique(),
                        season_data["round"].unique(),
                    )
                ),
                columns=["fpl_name", "pos", "team", "round"],
            )
            player_team_rounds = player_team_rounds.merge(
                player_data, on=["fpl_name", "pos", "team", "round"], how="left"
            )
            player_team_rounds["season"] = player_team_rounds["season"].fillna(season)
            player_team_rounds["minutes"] = player_team_rounds["minutes"].fillna(0)
            player_round_mins_filled.append(player_team_rounds)
    player_round_mins_filled = pd.concat(player_round_mins_filled).rename(
        columns={"minutes": "mins_played"}
    )

    player_round_mins_filled = (
        pd.merge(
            player_round_mins_filled,
            team_round_mins,
            how="left",
            on=["team", "season", "round"],
        )
        .sort_values(["fpl_name", "team", "season", "round"])
        .dropna(subset=["mins_max"])
    )
    player_round_mins_filled["pct_played"] = (
        player_round_mins_filled["mins_played"] / player_round_mins_filled["mins_max"]
    )
    player_round_mins_filled["appearance"] = (
        player_round_mins_filled["mins_played"] > 20
    ).astype(int)

    for i in range(1, dnp_lookback + 1):
        player_round_mins_filled[f"pct_played_{i}"] = player_round_mins_filled.groupby(
            ["team", "fpl_name"]
        )["pct_played"].shift(i)
        player_round_mins_filled[f"pct_played_{i}"] = player_round_mins_filled[
            f"pct_played_{i}"
        ].fillna(-1)
    player_round_mins_filled.loc[
        player_round_mins_filled["mins_max"] == 0, "appearance"
    ] = np.NaN
    player_round_mins_filled["did_not_play"] = (
        1 - player_round_mins_filled["appearance"]
    )
    return player_round_mins_filled


evaluate_dnp_model_holdout = partial(evaluate_model, evaluation_set="dnp_holdout")
