import logging
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt

from fpl.utils import backup_latest_n

logger = logging.getLogger(__name__)


def find_top_players(inference_results, column):
    top_players = inference_results.groupby("round").apply(
        lambda x: x[x[column] >= x[column].quantile(0.95)]
    )
    return top_players.index.levels[1]


def avg_points_gained(inference_results, squad, prev_points_col, next_points_col):
    if len(squad) > 0:
        selected = inference_results["element"].isin(squad)
    else:
        selected = find_top_players(inference_results, prev_points_col)

    top_unselected = find_top_players(inference_results, next_points_col)

    inference_results.loc[selected, "avoided_loss"] = (
        inference_results.loc[selected, prev_points_col]
        - inference_results.loc[selected, next_points_col]
    )
    inference_results.loc[top_unselected, "captured_gain"] = (
        inference_results.loc[top_unselected, next_points_col]
        - inference_results.loc[top_unselected, prev_points_col]
    )
    average_gain = inference_results.groupby("round")["captured_gain"].mean()
    average_loss = inference_results.groupby("round")["avoided_loss"].mean()
    return average_gain + average_loss


def find_step_gains(inference_results, horizon, column, squad):
    inference_results["prev_n_gw_sum"] = inference_results.groupby("fpl_name")[
        column
    ].transform(lambda x: x.shift(1).rolling(horizon, min_periods=1).sum())
    inference_results["next_n_gw_sum"] = inference_results.groupby("fpl_name")[
        column
    ].transform(lambda x: x.iloc[::-1].rolling(horizon, min_periods=1).sum().iloc[::-1])
    total_gains = avg_points_gained(
        inference_results, squad, "prev_n_gw_sum", "next_n_gw_sum"
    )
    total_gains = total_gains.fillna(0)
    return total_gains


def get_excluded_weeks(chips_usage, gap):
    excluded = set()
    for chip, week in chips_usage.items():
        if week is None:
            continue
        if chip in ("wildcard1", "wildcard2", "free_hit"):
            for w in range(week - gap, week + gap + 1):
                excluded.add(w)
        else:
            excluded.add(week)
    return excluded


def get_candidate_weeks(chips_usage, start, end, expanded, skip_first_week):
    gap = 3 if expanded else 0
    candidate_weeks = []
    while gap >= 0:
        excluded = get_excluded_weeks(chips_usage, gap=gap)
        excluded.add(start) if skip_first_week else None
        candidate_weeks = [w for w in range(start, end + 1) if w not in excluded]
        if len(candidate_weeks) > 0:
            return candidate_weeks
        gap -= 1
    assert len(candidate_weeks) > 0, "No candidate weeks available"


def suggest_wildcard_weeks(
    inference_results,
    squad,
    column,
    horizon,
    current_week,
    chips_usage,
    skip_first_week,
):
    wc1_deadline = 19
    start_week = max(horizon, current_week)
    end_week = 38 - horizon + 1
    squad_fixture_swings = find_step_gains(inference_results, horizon, column, squad)
    other_fixture_swings = find_step_gains(inference_results, horizon, column, [])

    if current_week <= wc1_deadline:
        if chips_usage["wildcard1"] is None:
            candidate_weeks = get_candidate_weeks(
                chips_usage,
                start_week,
                wc1_deadline,
                expanded=True,
                skip_first_week=skip_first_week,
            )
            chips_usage["wildcard1"] = squad_fixture_swings.loc[
                candidate_weeks
            ].idxmax()
            logger.info(f"Suggested Wildcard 1 week: {chips_usage['wildcard1']}")
        if chips_usage["wildcard2"] is None:
            candidate_weeks = get_candidate_weeks(
                chips_usage,
                wc1_deadline + 1,
                end_week,
                expanded=True,
                skip_first_week=skip_first_week,
            )
            chips_usage["wildcard2"] = other_fixture_swings.loc[
                candidate_weeks
            ].idxmax()
            logger.info(f"Suggested Wildcard 2 week: {chips_usage['wildcard2']}")
        return pd.concat(
            [
                squad_fixture_swings.loc[start_week : chips_usage["wildcard1"]],
                other_fixture_swings.loc[chips_usage["wildcard1"] + 1 : end_week],
            ]
        )
    else:
        if chips_usage["wildcard2"] is None:
            candidate_weeks = get_candidate_weeks(
                chips_usage,
                max(wc1_deadline + 1, start_week),
                end_week,
                expanded=True,
                skip_first_week=skip_first_week,
            )
            chips_usage["wildcard2"] = squad_fixture_swings.loc[
                candidate_weeks
            ].idxmax()
            logger.info(f"Suggested Wildcard 2 week: {chips_usage['wildcard2']}")
        return pd.concat(
            [
                squad_fixture_swings.loc[start_week : chips_usage["wildcard2"]],
                other_fixture_swings.loc[chips_usage["wildcard2"] + 1 : end_week],
            ]
        )


def suggest_free_hit_week(
    inference_results,
    squad,
    column,
    horizon,
    current_week,
    chips_usage,
    skip_first_week,
):
    inference_results["prev_n_gw_sum"] = inference_results.groupby("fpl_name")[
        column
    ].transform(lambda x: x.shift(1).rolling(horizon - 1).sum())
    inference_results["next_n_gw_sum"] = inference_results.groupby("fpl_name")[
        column
    ].transform(lambda x: x.iloc[::-1].rolling(horizon - 1).sum().iloc[::-1].shift(-1))
    inference_results["outside_avg"] = (
        (inference_results["prev_n_gw_sum"] + inference_results["next_n_gw_sum"])
        / (horizon - 1)
        / 2
    )
    fixture_spikes = avg_points_gained(inference_results, squad, "outside_avg", column)
    fixture_spikes = fixture_spikes.fillna(0)

    start_week = max(horizon, current_week)
    end_week = 38 - horizon
    fixture_spikes = fixture_spikes.loc[start_week:end_week]
    if chips_usage["free_hit"] is None:
        candidate_weeks = get_candidate_weeks(
            chips_usage,
            start_week,
            end_week,
            expanded=True,
            skip_first_week=skip_first_week,
        )
        chips_usage["free_hit"] = fixture_spikes.loc[candidate_weeks].idxmax()
        logger.info(f"Suggested Free Hit week: {chips_usage['free_hit']}")
    return fixture_spikes


def suggest_triple_captain_week(
    inference_results, squad, column, current_week, chips_usage
):
    if len(squad) > 0:
        inference_results = inference_results.loc[
            inference_results["element"].isin(squad)
        ]
    max_points = inference_results.groupby("round")[column].max()
    max_points = max_points.loc[current_week:]

    if chips_usage["triple_captain"] is None:
        candidate_weeks = get_candidate_weeks(
            chips_usage, current_week, 38, expanded=False, skip_first_week=False
        )
        chips_usage["triple_captain"] = max_points.loc[candidate_weeks].idxmax()
        logger.info(f"Suggested Triple Captain week: {chips_usage['triple_captain']}")
    return max_points


def suggest_bench_boost_week(
    inference_results, squad, column, current_week, chips_usage
):

    if len(squad) > 0:
        inference_results = inference_results.loc[
            inference_results["element"].isin(squad)
        ]
        bench_players = inference_results.groupby("round").apply(
            lambda x: x.nsmallest(4, column)
        )
        bench_idx = bench_players.index.levels[1]
        inference_results = inference_results.loc[bench_idx]
    else:
        top_players = find_top_players(inference_results, column)
        inference_results = inference_results.loc[
            ~inference_results.index.isin(top_players)
        ]
    bench_points = inference_results.groupby("round")[column].mean()
    bench_points = bench_points.loc[current_week:]

    if chips_usage["bench_boost"] is None:
        if all(bench_points <= 1):
            logger.info("Suggested Bench Boost week: None - Bench points are too low.")
        else:
            candidate_weeks = get_candidate_weeks(
                chips_usage, current_week, 38, expanded=False, skip_first_week=False
            )
            chips_usage["bench_boost"] = bench_points.loc[candidate_weeks].idxmax()
            logger.info(f"Suggested Bench Boost week: {chips_usage['bench_boost']}")
    return bench_points


chip_name_mapping = {
    "bboost": "bench_boost",
    "freehit": "free_hit",
    "3xc": "triple_captain",
}


def plot_chips_suggestions(
    chips_usage, fixture_swings, fixture_spikes, max_points, bench_points
):
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(20, 5),
    )
    fig.suptitle("Chips Suggestions")
    for ax in axs.flat:
        ax.grid(True)

    axs.flat[0].plot(
        fixture_swings.index, fixture_swings.values, color="grey", linestyle="--"
    )
    if chips_usage["wildcard1"] in fixture_swings.index:
        axs.flat[0].scatter(
            chips_usage["wildcard1"],
            fixture_swings.loc[chips_usage["wildcard1"]],
            color="red",
            label="_nolegend_",
            zorder=5,
        )

    if chips_usage["wildcard2"] in fixture_swings.index:
        axs.flat[0].scatter(
            chips_usage["wildcard2"],
            fixture_swings.loc[chips_usage["wildcard2"]],
            color="red",
            label="_nolegend_",
            zorder=5,
        )
    axs.flat[0].set_title(f"Fixture Swings & Wildcards")

    axs.flat[1].plot(
        fixture_spikes.index, fixture_spikes.values, color="grey", linestyle="--"
    )
    if chips_usage["free_hit"] in fixture_spikes.index:
        axs.flat[1].scatter(
            chips_usage["free_hit"],
            fixture_spikes.loc[chips_usage["free_hit"]],
            color="red",
            label="_nolegend_",
            zorder=5,
        )
    axs.flat[1].set_title(f"Fixture Spikes & Free Hit")

    axs.flat[2].plot(max_points.index, max_points.values, color="grey", linestyle="--")
    if chips_usage["triple_captain"] in max_points.index:
        axs.flat[2].scatter(
            chips_usage["triple_captain"],
            max_points.loc[chips_usage["triple_captain"]],
            color="red",
            label="_nolegend_",
            zorder=5,
        )
    axs.flat[2].set_title("Top Points & Triple Captain")

    axs.flat[3].plot(
        bench_points.index, bench_points.values, color="grey", linestyle="--"
    )
    if chips_usage["bench_boost"] in bench_points.index:
        axs.flat[3].scatter(
            chips_usage["bench_boost"],
            bench_points.loc[chips_usage["bench_boost"]],
            color="red",
            label="_nolegend_",
            zorder=5,
        )
    axs.flat[3].set_title("Benched Points & Bench Boost")
    plt.tight_layout()

    filename = "data/optimization/chips_suggestions.png"
    plt.savefig(filename)
    backup_latest_n(filename, n=5)
    return None


def get_chips_suggestions(
    inference_results: pd.DataFrame,
    squad: list[str],
    chips_history: list[dict[str, Any]],
    current_week: int,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    chips_usage = {
        "wildcard1": parameters.get("wildcard1_week"),
        "wildcard2": parameters.get("wildcard2_week"),
        "free_hit": parameters.get("free_hit_week"),
        "triple_captain": parameters.get("triple_captain_week"),
        "bench_boost": parameters.get("bench_boost_week"),
    }
    skip_first_week = any(v is not None for v in parameters["force_transfers"].values())
    for chip in chips_history:
        if chip["name"] == "wildcard":
            if chip["event"] <= 19:
                chips_usage["wildcard1"] = chip["event"]
            else:
                chips_usage["wildcard2"] = chip["event"]
        else:
            chips_usage[chip_name_mapping[chip["name"]]] = chip["event"]

    fixture_swings = suggest_wildcard_weeks(
        inference_results,
        squad,
        column="predicted_points",
        horizon=3,
        current_week=current_week,
        chips_usage=chips_usage,
        skip_first_week=skip_first_week,
    )
    fixture_spikes = suggest_free_hit_week(
        inference_results,
        squad,
        column="predicted_points",
        horizon=3,
        current_week=current_week,
        chips_usage=chips_usage,
        skip_first_week=skip_first_week,
    )
    max_points = suggest_triple_captain_week(
        inference_results,
        squad,
        column="predicted_points",
        current_week=current_week,
        chips_usage=chips_usage,
    )

    bench_points = suggest_bench_boost_week(
        inference_results,
        squad,
        column="predicted_points",
        current_week=current_week,
        chips_usage=chips_usage,
    )
    plot_chips_suggestions(
        chips_usage, fixture_swings, fixture_spikes, max_points, bench_points
    )
    return chips_usage
