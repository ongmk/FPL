from pulp import (
    LpProblem,
    LpMinimize,
    LpBinary,
    LpVariable,
    LpContinuous,
    LpInteger,
    lpSum,
    LpStatusOptimal,
    LpStatusInfeasible,
    LpStatusUnbounded,
    LpStatusNotSolved,
)
import pandas as pd
import requests
import os
import numpy as np
from subprocess import check_output
import re
import itertools
import difflib
import matplotlib.pyplot as plt
import time
import warnings
from cryptography.utils import CryptographyDeprecationWarning
import logging
from cplex_connection.LpRemoteCplex import RemoteCPLEXSolver
import xml.etree.ElementTree as et

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)


def get_fpl_base_data():
    r = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
    fpl_data = r.json()

    all_gws = {"current": 0, "finished": [], "future": []}
    for w in fpl_data["events"]:
        if w["is_current"] == True:
            all_gws["current"] = w["id"]
        elif w["finished"] == True:
            all_gws["finished"].append(w["id"])
        else:
            all_gws["future"].append(w["id"])

    element_data = pd.DataFrame(fpl_data["elements"])
    team_data = pd.DataFrame(fpl_data["teams"]).set_index("id")
    elements_team = pd.merge(
        element_data, team_data, left_on="team", right_index=True, suffixes=("", "_y")
    )
    elements_team = elements_team.drop(
        elements_team.filter(regex="_y$").columns, axis=1
    )
    elements_team["full_name"] = elements_team["first_name"].str.cat(
        elements_team["second_name"], sep=" "
    )
    elements_team = elements_team[
        [
            "web_name",
            "team",
            "element_type",
            "name",
            "full_name",
            "short_name",
            "now_cost",
            "id",
        ]
    ]

    type_data = pd.DataFrame(fpl_data["element_types"]).set_index(["id"])

    return elements_team, team_data, type_data, all_gws


def get_latest_prediction_csv(gameweeks):
    path = "data/raw/theFPLkiwi/FPL_projections_22_23/"
    files = [
        (int(f.strip("FPL_GW").strip(".csv")), os.path.join(path, f))
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    ]
    files = [f for f in files if f[0] <= gameweeks[0]]
    index = np.argmax([f[0] for f in files])
    return files[index][1]


def get_pred_pts_data(gameweeks):
    latest_csv = get_latest_prediction_csv(gameweeks)
    fpl_name_dict = pd.read_csv(
        "data/raw/theFPLkiwi/ID_Dictionary.csv", encoding="cp1252"
    )[["Name", "FPL name"]]
    pred_pts_data = pd.read_csv(latest_csv)
    pred_pts_data = fpl_name_dict.merge(pred_pts_data, on="Name")
    empty_cols = ["GW xMins", "xPts if play", "Probability to play", "xPts", "npG"]
    pred_pts_data = pred_pts_data.drop(empty_cols, axis=1)
    week_cols = pred_pts_data.columns[9:]
    new_col_name = {
        col: f"{empty_cols[round(float(col)%1*10)]}_{round(float(col))}"
        for col in week_cols
    }
    pred_pts_data = pred_pts_data.rename(columns=new_col_name)
    pred_pts_data = pred_pts_data[
        ["FPL name", "Team", "Pos", "Price"] + [f"xPts_{w}" for w in gameweeks]
    ]
    return pred_pts_data


def fuzzy_match(row, df_to_match):
    combined_name = row["FPL name"] + " " + row["Team"]
    df_to_match["combined_name"] = (
        df_to_match["web_name"] + " " + df_to_match["short_name"]
    )
    all_options = (df_to_match["combined_name"]).tolist()

    matches = difflib.get_close_matches(combined_name, all_options, n=1, cutoff=0.6)
    if matches:
        matched = next(iter(matches))
        return df_to_match.loc[
            df_to_match["combined_name"] == matched, "web_name"
        ].values[0]
    else:
        return None


def resolve_fpl_names(df, fpl_names):
    df["matched"] = df.apply(lambda row: fuzzy_match(row, fpl_names), axis=1)
    df["same"] = df["FPL name"] == df["matched"]
    df = df.sort_values(["same", "Price"], ascending=False)
    df = df.drop_duplicates(["Team", "matched"])
    df.loc[:, "FPL name"] = df["matched"]
    df = df.drop(["matched", "same"], axis=1)
    return df


def get_initial_squad(team_id, gw):
    r = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/"
    )
    picks_data = r.json()
    return picks_data["picks"]


def get_live_data(team_id, horizon):
    elements_team, team_data, type_data, all_gws = get_fpl_base_data()
    gameweeks = all_gws["future"][:horizon]
    current_gw = all_gws["current"]

    pred_pts_data = get_pred_pts_data(gameweeks)
    pred_pts_data = resolve_fpl_names(
        pred_pts_data, elements_team[["web_name", "short_name"]]
    )

    merged_data = elements_team.merge(
        pred_pts_data,
        left_on=["web_name", "short_name"],
        right_on=["FPL name", "Team"],
        how="left",
    ).set_index("id")

    r = requests.get(f"https://fantasy.premierleague.com/api/entry/{team_id}/")
    general_data = r.json()
    itb = general_data["last_deadline_bank"] / 10
    initial_squad = get_initial_squad(team_id, current_gw)
    initial_squad = [p["element"] for p in initial_squad]
    r = requests.get(
        f"https://fantasy.premierleague.com/api/entry/{team_id}/transfers/"
    )
    transfer_data = reversed(r.json())
    merged_data["sell_price"] = merged_data["now_cost"]
    for t in transfer_data:
        if t["element_in"] in initial_squad:
            bought_price = t["element_in_cost"]
            merged_data.loc[t["element_in"], "bought_price"] = bought_price
            current_price = merged_data.loc[t["element_in"], "now_cost"]
            if current_price > bought_price:
                sell_price = np.ceil(np.mean([current_price, bought_price]))
                merged_data.loc[t["element_in"], "sell_price"] = sell_price

    merged_data = merged_data.dropna(
        subset=["FPL name", "Team", "Pos", "Price", "bought_price"], how="all"
    )
    xPts_cols = [
        column for column in merged_data.columns if re.match(r"xPts_\d+", column)
    ]
    merged_data[xPts_cols] = merged_data[xPts_cols].fillna(0)

    logger.info("=" * 50)
    logger.info(f"Team: {general_data['name']}. Current week: {current_gw}")
    logger.info(f"Optimizing for {horizon} weeks. {gameweeks}")
    logger.info("=" * 50 + "\n")

    return {
        "merged_data": merged_data,
        "team_data": team_data,
        "type_data": type_data,
        "gameweeks": gameweeks,
        "initial_squad": initial_squad,
        "itb": itb,
    }


def get_backtest_data(latest_elements_team, gw):
    backtest_data = pd.read_csv("data/raw/backtest_data/merged_gw.csv")[
        ["name", "position", "team", "xP", "GW", "value"]
    ]
    player_gw = pd.DataFrame(
        list(
            itertools.product(
                backtest_data["name"].unique(), backtest_data["GW"].unique()
            )
        ),
        columns=["name", "GW"],
    )
    backtest_data = player_gw.merge(backtest_data, how="left", on=["name", "GW"])
    backtest_data["value"] = backtest_data.groupby("name")["value"].ffill()
    backtest_data["position"] = backtest_data.groupby("name")["position"].ffill()
    backtest_data["team"] = backtest_data.groupby("name")["team"].ffill()
    backtest_data["xP"] = backtest_data["xP"].fillna(0)
    gw_data = backtest_data.loc[backtest_data["GW"] == gw, :]
    elements_team = latest_elements_team.merge(
        gw_data,
        left_on=["full_name", "name"],
        right_on=["name", "team"],
        suffixes=("", "_y"),
    )
    elements_team = (
        elements_team.drop(elements_team.filter(regex="_y$").columns, axis=1)
        .drop("GW", axis=1)
        .rename({"value": "now_cost"}, axis=1)
    )
    return elements_team


def get_name(row):
    name = ""
    if row["captain"] == 1:
        name += "[c] "
    elif row["vicecaptain"] == 1:
        name += "[v] "
    name += f"{row['name']} {row['predicted_xP']}"
    return name


def get_solution_status(solutionXML):
    solution_header = solutionXML.find("header")
    status_string = solution_header.get("solutionStatusString")
    objective_value_string = solution_header.get("objectiveValue")

    cplex_status = {
        "Optimal": LpStatusOptimal,
        "Feasible": LpStatusOptimal,
        "Infeasible": LpStatusInfeasible,
        "Unbounded": LpStatusUnbounded,
        "Stopped": LpStatusNotSolved,
    }

    status_str = "Undefined"
    if "optimal" in status_string:
        status_str = "Optimal"
    elif "feasible" in status_string:
        status_str = "Feasible"
    elif "infeasible" in status_string:
        status_str = "Infeasible"
    elif "integer unbounded" in status_string:
        status_str = "Integer Unbounded"
    elif "time limit exceeded" in status_string:
        status_str = "Feasible"

    return cplex_status[status_str], status_str, objective_value_string


def solve_multi_period_fpl(data, options):
    # Arguments
    ft = options.get("ft", 1)
    decay = options.get("decay", 0.84)
    timeout = options.get("timeout", 60)
    horizon = options.get("horizon", 5)
    tr_horizon = options.get("tr_horizon", 3)
    wc_on = options.get("wc_on", None)
    bb_on = options.get("bb_on", None)
    fh_on = options.get("fh_on", None)
    ft_value = options.get("ft_value", 1.5)
    itb_value = options.get("itb_value", 0.08)
    bench_weights = options.get("bench_weights", {0: 0.03, 1: 0.21, 2: 0.06, 3: 0.002})
    log = options.get("log", True)

    # Data
    merged_data = data["merged_data"]
    keys = [k for k in merged_data.columns.to_list() if "xPts_" in k]
    merged_data["total_ev"] = merged_data[keys].sum(axis=1)
    merged_data = merged_data.sort_values(by=["total_ev"], ascending=[False])
    team_data = data["team_data"]
    type_data = data["type_data"]
    gameweeks = data["gameweeks"]
    initial_squad = data["initial_squad"]
    itb = data["itb"]
    next_gw = gameweeks[0]
    transfer_gws = gameweeks[:tr_horizon]
    wc_on = wc_on if wc_on in transfer_gws else None
    bb_on = bb_on if bb_on in transfer_gws else None
    fh_on = fh_on if fh_on in transfer_gws else None
    wc_limit = 1 if wc_on else 0
    bb_limit = 1 if bb_on else 0
    fh_limit = 1 if fh_on else 0
    if next_gw == 1:
        threshold_gw = 2
    else:
        threshold_gw = next_gw

    # Sets
    players = merged_data.index.to_list()
    element_types = type_data.index.to_list()
    teams = team_data["name"].to_list()
    all_gws = [next_gw - 1] + gameweeks
    order = [0, 1, 2, 3]
    price_modified_players = merged_data.loc[
        merged_data["sell_price"] != merged_data["now_cost"]
    ].index.to_list()

    # Keys
    player_all_gws = list(itertools.product(players, all_gws))
    player_gameweeks = list(itertools.product(players, gameweeks))
    player_gameweeks_order = list(itertools.product(players, gameweeks, order))
    price_modified_players_gameweeks = list(
        itertools.product(price_modified_players, gameweeks)
    )

    # Model
    problem_name = f"multi_period"
    model = LpProblem(problem_name, LpMinimize)

    # Variables
    squad = LpVariable.dicts("squad", player_all_gws, cat=LpBinary)
    squad_fh = LpVariable.dicts("squad_fh", player_gameweeks, cat=LpBinary)
    lineup = LpVariable.dicts("lineup", player_gameweeks, cat=LpBinary)
    captain = LpVariable.dicts("captain", player_gameweeks, cat=LpBinary)
    vicecap = LpVariable.dicts("vicecap", player_gameweeks, cat=LpBinary)
    bench = LpVariable.dicts("bench", player_gameweeks_order, cat=LpBinary)
    transfer_in = LpVariable.dicts("transfer_in", player_gameweeks, cat=LpBinary)
    transfer_out_first = LpVariable.dicts(
        "tr_out_first", price_modified_players_gameweeks, cat=LpBinary
    )
    transfer_out_regular = LpVariable.dicts(
        "tr_out_reg", player_gameweeks, cat=LpBinary
    )
    transfer_out = {
        (p, w): transfer_out_regular[p, w]
        + (transfer_out_first[p, w] if p in price_modified_players else 0)
        for p in players
        for w in gameweeks
    }
    in_the_bank = LpVariable.dicts("itb", all_gws, cat=LpContinuous, lowBound=0)
    free_transfers = LpVariable.dicts(
        "free_transfers", all_gws, cat=LpInteger, lowBound=0, upBound=2
    )
    penalized_transfers = LpVariable.dicts(
        "penalized_transfers", gameweeks, cat=LpInteger, lowBound=0
    )
    aux = LpVariable.dicts("aux", gameweeks, cat=LpBinary)

    use_wc = LpVariable.dicts("use_wc", gameweeks, cat=LpBinary)
    use_bb = LpVariable.dicts("use_bb", gameweeks, cat=LpBinary)
    use_fh = LpVariable.dicts("use_fh", gameweeks, cat=LpBinary)

    # Dictionaries
    lineup_type_count = {
        (t, w): lpSum(
            lineup[p, w] for p in players if merged_data.loc[p, "element_type"] == t
        )
        for t in element_types
        for w in gameweeks
    }
    squad_type_count = {
        (t, w): lpSum(
            squad[p, w] for p in players if merged_data.loc[p, "element_type"] == t
        )
        for t in element_types
        for w in gameweeks
    }
    squad_fh_type_count = {
        (t, w): lpSum(
            squad_fh[p, w] for p in players if merged_data.loc[p, "element_type"] == t
        )
        for t in element_types
        for w in gameweeks
    }
    player_type = merged_data["element_type"].to_dict()
    sell_price = (merged_data["sell_price"] / 10).to_dict()
    buy_price = (merged_data["now_cost"] / 10).to_dict()
    sold_amount = {
        w: lpSum(
            [sell_price[p] * transfer_out_first[p, w] for p in price_modified_players]
        )
        + lpSum([buy_price[p] * transfer_out_regular[p, w] for p in players])
        for w in gameweeks
    }
    fh_sell_price = {
        p: sell_price[p] if p in price_modified_players else buy_price[p]
        for p in players
    }
    bought_amount = {
        w: lpSum([buy_price[p] * transfer_in[p, w] for p in players]) for w in gameweeks
    }
    points_player_week = {
        (p, w): merged_data.loc[p, f"xPts_{w}"] for p in players for w in gameweeks
    }
    squad_count = {w: lpSum(squad[p, w] for p in players) for w in gameweeks}
    squad_fh_count = {w: lpSum(squad_fh[p, w] for p in players) for w in gameweeks}
    number_of_transfers = {
        w: lpSum([transfer_out[p, w] for p in players]) for w in gameweeks
    }
    number_of_transfers[next_gw - 1] = 1
    transfer_diff = {
        w: number_of_transfers[w] - free_transfers[w] - 15 * use_wc[w]
        for w in gameweeks
    }

    # Initial conditions
    model += in_the_bank[next_gw - 1] == itb, "initial_itb"
    model += free_transfers[next_gw] == ft, "initial_ft"

    # Free transfer constraints
    if next_gw == 1 and threshold_gw in gameweeks:
        model += free_transfers[threshold_gw] == ft, "ps_initial_ft"

    # Chip constraints
    model += lpSum(use_wc[w] for w in gameweeks) <= wc_limit, "use_wc_limit"
    model += lpSum(use_bb[w] for w in gameweeks) <= bb_limit, "use_bb_limit"
    model += lpSum(use_fh[w] for w in gameweeks) <= fh_limit, "use_fh_limit"
    if wc_on is not None:
        model += use_wc[wc_on] == 1, "force_wc"
    if bb_on is not None:
        model += use_bb[bb_on] == 1, "force_bb"
    if fh_on is not None:
        model += use_fh[fh_on] == 1, "force_fh"

    # Transfer horizon constraint
    model += (
        lpSum(
            transfer_in[p, w] + transfer_out[p, w]
            for p in players
            for w in gameweeks
            if w not in transfer_gws
        )
        == 0,
        f"no_transfer",
    )

    for p in players:
        # Initial conditions
        if p in initial_squad:
            model += squad[p, next_gw - 1] == 1, f"initial_squad_players_{p}"
        else:
            model += squad[p, next_gw - 1] == 0, f"initial_squad_others_{p}"

        # Multiple-sell fix
        if p in price_modified_players:
            model += (
                lpSum(transfer_out_first[p, w] for w in gameweeks) <= 1,
                f"multi_sell_3_{p}",
            )

    for w in gameweeks:
        # Initial conditions
        if w > next_gw:
            model += free_transfers[w] >= 1, f"future_ft_limit_{w}"

        # Constraints
        model += squad_count[w] == 15, f"squad_count_{w}"
        model += squad_fh_count[w] == 15 * use_fh[w], f"squad_fh_count_{w}"
        model += (
            lpSum([lineup[p, w] for p in players]) == 11 + 4 * use_bb[w],
            f"lineup_count_{w}",
        )
        model += (
            lpSum(bench[p, w, 0] for p in players if player_type[p] == 1)
            == 1 - use_bb[w],
            f"bench_gk_{w}",
        )
        for o in [1, 2, 3]:
            model += (
                lpSum(bench[p, w, o] for p in players) == 1 - use_bb[w],
                f"bench_count_{w}_{o}",
            )
        model += lpSum([captain[p, w] for p in players]) == 1, f"captain_count_{w}"
        model += lpSum([vicecap[p, w] for p in players]) == 1, f"vicecap_count_{w}"

        # Free transfer constraints
        if w > threshold_gw:
            model += free_transfers[w] == aux[w] + 1, f"aux_ft_rel_{w}"
            model += (
                free_transfers[w - 1]
                - number_of_transfers[w - 1]
                - 2 * use_wc[w - 1]
                - 2 * use_fh[w - 1]
                <= 2 * aux[w],
                f"force_aux_1_{w}",
            )
            model += (
                free_transfers[w - 1]
                - number_of_transfers[w - 1]
                - 2 * use_wc[w - 1]
                - 2 * use_fh[w - 1]
                >= aux[w] + (-14) * (1 - aux[w]),
                f"force_aux_2_{w}",
            )
        model += penalized_transfers[w] >= transfer_diff[w], f"pen_transfer_rel_{w}"

        for t in element_types:
            model += (
                lineup_type_count[t, w] >= type_data.loc[t, "squad_min_play"],
                f"valid_formation_lb_{t}_{w}",
            )
            model += (
                lineup_type_count[t, w]
                <= type_data.loc[t, "squad_max_play"] + use_bb[w],
                f"valid_formation_ub_{t}_{w}",
            )
            model += (
                squad_type_count[t, w] == type_data.loc[t, "squad_select"],
                f"valid_squad_{t}_{w}",
            )
            model += (
                squad_fh_type_count[t, w]
                == type_data.loc[t, "squad_select"] * use_fh[w],
                f"valid_squad_fh_{t}_{w}",
            )

        for t in teams:
            model += (
                lpSum(squad[p, w] for p in players if merged_data.loc[p, "name"] == t)
                <= 3,
                f"team_limit_{t}_{w}",
            )
            model += (
                lpSum(
                    squad_fh[p, w] for p in players if merged_data.loc[p, "name"] == t
                )
                <= 3,
                f"team_limit_fh_{t}_{w}",
            )

        # Transfer constraints
        model += (
            in_the_bank[w] == in_the_bank[w - 1] + sold_amount[w] - bought_amount[w],
            f"cont_budget_{w}",
        )
        model += (
            lpSum(fh_sell_price[p] * squad[p, w - 1] for p in players)
            + in_the_bank[w - 1]
            >= lpSum(fh_sell_price[p] * squad_fh[p, w] for p in players),
            f"fh_budget_{w}",
        )

        # Chip constraints
        model += use_wc[w] + use_fh[w] + use_bb[w] <= 1, f"single_chip_{w}"
        if w > next_gw:
            model += aux[w] <= 1 - use_wc[w - 1], f"ft_after_wc_{w}"
            model += aux[w] <= 1 - use_fh[w - 1], f"ft_after_fh_{w}"

        for p in players:
            # Constraints
            model += (
                lineup[p, w] <= squad[p, w] + use_fh[w],
                f"lineup_squad_rel_{p}_{w}",
            )
            model += (
                lineup[p, w] <= squad_fh[p, w] + 1 - use_fh[w],
                f"lineup_squad_fh_rel_{p}_{w}",
            )
            for o in order:
                model += (
                    bench[p, w, o] <= squad[p, w] + use_fh[w],
                    f"bench_squad_rel_{p}_{w}_{o}",
                )
                model += (
                    bench[p, w, o] <= squad_fh[p, w] + 1 - use_fh[w],
                    f"bench_squad_fh_rel_{p}_{w}_{o}",
                )
            model += captain[p, w] <= lineup[p, w], f"captain_lineup_rel_{p}_{w}"
            model += vicecap[p, w] <= lineup[p, w], f"vicecap_lineup_rel_{p}_{w}"
            model += captain[p, w] + vicecap[p, w] <= 1, f"cap_vc_rel_{p}_{w}"
            model += (
                lineup[p, w] + lpSum(bench[p, w, o] for o in order) <= 1,
                f"lineup_bench_rel_{p}_{w}_{o}",
            )

            # Transfer constraints
            model += (
                squad[p, w] == squad[p, w - 1] + transfer_in[p, w] - transfer_out[p, w],
                f"squad_transfer_rel_{p}_{w}",
            )
            model += transfer_in[p, w] <= 1 - use_fh[w], f"no_tr_in_fh_{p}_{w}"
            model += transfer_out[p, w] <= 1 - use_fh[w], f"no_tr_out_fh_{p}_{w}"

            # Chips constraint
            model += squad_fh[p, w] <= use_fh[w], f"fh_squad_logic_{p}_{w}"

            # Multiple-sell fix
            if p in price_modified_players:
                model += (
                    transfer_out_first[p, w] + transfer_out_regular[p, w] <= 1,
                    f"multi_sell_1_{p}_{w}",
                )
                model += (
                    horizon
                    * lpSum(
                        transfer_out_first[p, wbar] for wbar in gameweeks if wbar <= w
                    )
                    >= lpSum(
                        transfer_out_regular[p, wbar] for wbar in gameweeks if wbar >= w
                    ),
                    f"multi_sell_2_{p}_{w}",
                )

            # Transfer in/out fix
            model += (
                transfer_in[p, w] + transfer_out[p, w] <= 1,
                f"tr_in_out_limit_{p}_{w}",
            )

    # Objective
    gw_xp = {
        w: lpSum(
            [
                points_player_week[p, w]
                * (
                    lineup[p, w]
                    + captain[p, w]
                    + 0.1 * vicecap[p, w]
                    + lpSum(bench_weights[o] * bench[p, w, o] for o in order)
                )
                for p in players
            ]
        )
        for w in gameweeks
    }
    gw_total = {
        w: gw_xp[w]
        - 4 * penalized_transfers[w]
        + ft_value * free_transfers[w]
        + itb_value * in_the_bank[w]
        for w in gameweeks
    }

    decay_objective = lpSum(
        [gw_total[w] * pow(decay, i) for i, w in enumerate(gameweeks)]
    )
    model += -decay_objective, "total_decay_xp"

    # t0 = time.time()
    # model.writeLP("./model.lp")
    # command = f"cbc model.lp cost column sec {timeout} solve solu solution.txt"
    # output = check_output(command).decode("utf-8")
    # with open("cbc.log", "w", encoding="utf-8") as f:
    #     f.write(output)
    # solve_time = time.time() - t0
    # pattern = re.compile(r"There were.+errors on input")
    # if log:
    #     logger.info(output)
    # if "No feasible solution found" in output:
    #     raise Exception("NO FEASIBLE SOLUTION")
    # elif pattern.search(output):
    #     raise Exception("ERRORS ON INPUT")
    # # Parsing
    # for variable in model.variables():
    #     variable.varValue = 0
    # with open("solution.txt", "r") as f:
    #     vars = model.variablesDict()
    #     for line in f:
    #         if "objective value" in line:
    #             continue
    #         _, variable, value, _ = line.split()
    #         vars[variable].varValue = float(value)

    lp_file_name = "fplmodel"
    model.writeLP(f"./{lp_file_name}.lp")
    import yaml

    with open("./conf/base/credentials.yml", "r") as file:
        config = yaml.safe_load(file)
        config = config["cplex"]
    solver = RemoteCPLEXSolver(
        fileName=lp_file_name,
        localPath=".",
        config=config,
        log=log,
        cplexTimeOut=timeout,
    )
    cplex_log = solver.solve()
    if cplex_log["infeasible"]:
        raise Exception("Cannot find feasible solution.")
    solutionXML = et.parse(f"./{lp_file_name}.sol").getroot()
    _, status_str, objValString = get_solution_status(solutionXML)
    gap_pct = cplex_log["gap_pct"] if cplex_log else None
    solution_time = cplex_log["solution_time"] if cplex_log else None
    status_str = "Acceptable" if cplex_log and cplex_log["Acceptable"] else status_str
    logger.info(
        f"{status_str} solution found in: {solution_time}s. Gap: {gap_pct}%. Objective: {float(objValString):.2f}"
    )

    variables = solutionXML.find("variables")

    for variable in model.variables():
        variable.varValue = 0
    vars = model.variablesDict()

    for variable in variables:
        var_name, var_value = variable.get("name"), variable.get("value")
        vars[var_name].varValue = float(var_value)

    # DataFrame generation
    picks = []
    for w in gameweeks:
        for p in players:
            if (
                squad[p, w].value()
                + squad_fh[p, w].value()
                + transfer_out[p, w].value()
                > 0.5
            ):
                lp = merged_data.loc[p]
                is_captain = 1 if captain[p, w].value() > 0.5 else 0
                is_squad = (
                    1
                    if (use_fh[w].value() < 0.5 and squad[p, w].value() > 0.5)
                    or (use_fh[w].value() > 0.5 and squad_fh[p, w].value() > 0.5)
                    else 0
                )
                is_lineup = 1 if lineup[p, w].value() > 0.5 else 0
                is_vice = 1 if vicecap[p, w].value() > 0.5 else 0
                is_transfer_in = 1 if transfer_in[p, w].value() > 0.5 else 0
                is_transfer_out = 1 if transfer_out[p, w].value() > 0.5 else 0
                bench_value = -1
                for o in order:
                    if bench[p, w, o].value() > 0.5:
                        bench_value = o
                player_buy_price = 0 if not is_transfer_in else buy_price[p]
                player_sell_price = (
                    0
                    if not is_transfer_out
                    else (
                        sell_price[p]
                        if p in price_modified_players
                        and transfer_out_first[p, w].value() > 0.5
                        else buy_price[p]
                    )
                )
                multiplier = 1 * (is_lineup == 1) + 1 * (is_captain == 1)
                xp_cont = points_player_week[p, w] * multiplier
                position = type_data.loc[lp["element_type"], "singular_name_short"]
                picks.append(
                    [
                        w,
                        p,
                        lp["web_name"],
                        position,
                        lp["element_type"],
                        lp["name"],
                        player_buy_price,
                        player_sell_price,
                        round(points_player_week[p, w], 2),
                        is_squad,
                        is_lineup,
                        bench_value,
                        is_captain,
                        is_vice,
                        is_transfer_in,
                        is_transfer_out,
                        multiplier,
                        xp_cont,
                    ]
                )
    picks_df = pd.DataFrame(
        picks,
        columns=[
            "week",
            "id",
            "name",
            "pos",
            "type",
            "team",
            "buy_price",
            "sell_price",
            "predicted_xP",
            "squad",
            "lineup",
            "bench",
            "captain",
            "vicecaptain",
            "transfer_in",
            "transfer_out",
            "multiplier",
            "xp_cont",
        ],
    ).sort_values(
        by=["week", "lineup", "type", "predicted_xP"],
        ascending=[True, False, True, False],
    )

    # Writing summary
    summary_of_actions = []
    total_xp = 0
    for w in gameweeks:
        header = f" GW {w} "
        gw_in = pd.DataFrame([], columns=["", "In", "xP", "Pos"])
        gw_out = pd.DataFrame([], columns=["Out", "xP", "Pos"])
        net_cost = 0
        net_xp = 0
        for p in players:
            if transfer_in[p, w].value() > 0.5:
                price = merged_data["now_cost"][p] / 10
                name = f'{merged_data["web_name"][p]} ({price})'
                pos = merged_data["element_type"][p]
                xp = round(points_player_week[p, w], 2)
                net_cost += price
                net_xp += xp
                gw_in.loc[len(gw_in)] = ["👉", name, xp, pos]
            if transfer_out[p, w].value() > 0.5:
                price = merged_data["sell_price"][p] / 10
                name = f'{merged_data["web_name"][p]} ({price})'
                pos = merged_data["element_type"][p]
                xp = round(points_player_week[p, w], 2)
                net_cost -= price
                net_xp -= xp
                gw_out.loc[len(gw_out)] = [name, xp, pos]
        gw_in = gw_in.sort_values("Pos").drop("Pos", axis=1).reset_index(drop=True)
        gw_out = gw_out.sort_values("Pos").drop("Pos", axis=1).reset_index(drop=True)
        if use_wc[w].value() > 0.5:
            chip_summary = "[Wildcard Active]\n"
        elif use_fh[w].value() > 0.5:
            chip_summary = "[Free Hit Active]\n"
        elif use_bb[w].value() > 0.5:
            chip_summary = "[Bench Boost Active]\n"
        else:
            chip_summary = ""
        if w in transfer_gws:
            transfer_summary = (
                f"Free Transfers = {free_transfers[w].value()}    Hits = {penalized_transfers[w].value()}\n"
                f"Cost = {net_cost:.1f}    ITB = {in_the_bank[w].value():.2f}   xP Gain = {net_xp:.2f}.\n\n"
                f"{str(gw_in) if gw_out.empty else str(pd.concat([gw_out, gw_in], axis=1, join='inner'))}\n\n"
            )
        else:
            transfer_summary = "\n"

        gw_squad = picks_df.loc[picks_df["week"] == w, :].copy()
        gw_squad.loc[:, "name"] = gw_squad.agg(lambda x: get_name(x), axis=1)
        gw_lineup = gw_squad.loc[gw_squad["lineup"] == 1]
        lineup_str = []
        for p_type in [1, 2, 3, 4]:
            lineup_str.append(
                (gw_lineup.loc[gw_lineup["type"] == p_type, "name"]).str.cat(sep="    ")
            )
        lineup_str.append("")
        gw_bench = gw_squad.loc[gw_squad["bench"] != -1].sort_values("bench")
        lineup_str.append(f'Bench: {gw_bench["name"].str.cat(sep="    ")}')
        length = max([len(s) for s in lineup_str])
        lineup_str = [f"{s:^{length}}" for s in lineup_str]
        lineup_str = "\n".join(lineup_str)
        gw_xp = (
            lpSum(
                [
                    (lineup[p, w] + captain[p, w]) * points_player_week[p, w]
                    for p in players
                ]
            ).value()
            - penalized_transfers[w].value() * 4
        )
        total_xp += gw_xp
        hits = int(penalized_transfers[w].value())
        hit_str = f"({hits} hits)" if hits > 0 else ""
        gw_summary = (
            f"\n"
            f"{header:{'*'}^80}\n\n"
            f"{chip_summary}"
            f"{transfer_summary}"
            f"{lineup_str}\n\n"
            f"Gameweek xP = {gw_xp:.2f} {hit_str}\n"
        )
        summary_of_actions.append(gw_summary)
        if w == next_gw:
            if use_wc[w].value() > 0.5:
                chip_used = "wc"
            elif use_fh[w].value() > 0.5:
                chip_used = "fh"
            elif use_bb[w].value() > 0.5:
                chip_used = "bb"
            else:
                chip_used = None
            next_gw_dict = {
                "itb": round(in_the_bank[w].value(), 1),
                "ft": round(free_transfers[w + 1].value()),
                "hits": round(penalized_transfers[w].value()),
                "solve_time": solution_time,
                "n_transfers": round(
                    lpSum([transfer_out[p, w] for p in players]).value()
                ),
                "chip_used": chip_used,
            }
    overall_summary = (
        f"\n" f"{'':{'='}^80}\n" f"{horizon} weeks total xP = {total_xp:.2f}"
    )
    summary_of_actions.append(overall_summary)

    return picks_df, summary_of_actions, next_gw_dict


def get_historical_picks(team_id, next_gw, merged_data):
    initial_squad = get_initial_squad(team_id, next_gw)
    picks_df = (
        pd.DataFrame(initial_squad)
        .drop("position", axis=1)
        .rename({"element": "id"}, axis=1)
    )
    picks_df["week"] = next_gw
    picks_df = picks_df.merge(
        merged_data[["web_name", f"xPts_{next_gw}"]],
        left_on="id",
        right_index=True,
    ).rename({f"xPts_{next_gw}": "predicted_xP"}, axis=1)
    summary = [picks_df]
    next_gw_dict = {
        "hits": 0,
        "itb": 0,
        "ft": 0,
        "solve_time": 0,
        "n_transfers": 0,
        "chip_used": None,
    }
    return picks_df, summary, next_gw_dict


def live_run(parameters):
    data = get_live_data(parameters["team_id"], parameters["horizon"])
    picks, summary, next_gw_dict = solve_multi_period_fpl(
        data=data,
        options=parameters,
    )
    logger.info(f"Solved in {next_gw_dict['solve_time']}")
    for s in summary:
        logger.info(s)
    return summary, picks
    # picks.to_csv("picks.csv", index=False, encoding="utf-8-sig")
    # with open("summary.txt", "w", encoding="utf-8") as f:
    #     f.write("\n".join(summary))


def backtest(options, title="Backtest Result"):
    # Arguments
    horizon = options.get("horizon", 5)
    team_id = options.get("team_id", None)
    player_history = options.get("player_history", False)

    # Pre season
    latest_elements_team, team_data, type_data, all_gws = get_fpl_base_data()
    latest_elements_team = latest_elements_team.drop("now_cost", axis=1)
    itb = 100
    initial_squad = []
    options["ft"] = 1
    total_predicted_xp = 0
    total_xp = 0

    result_gw = []
    result_xp = []
    result_predicted_xp = []
    result_solve_times = []

    for next_gw in all_gws["finished"]:
        gameweeks = [i for i in range(next_gw, next_gw + horizon)]
        logger.info(80 * "=")
        logger.info(
            f"Backtesting GW {next_gw}. ITB = {itb:.1f}. FT = {options['ft']}. {gameweeks}"
        )
        logger.info(80 * "=")
        elements_team = get_backtest_data(latest_elements_team, next_gw)
        if elements_team.empty:
            logger.warning(f"No data from GW {next_gw}")
        else:
            pred_pts_data = get_pred_pts_data(gameweeks)
            pred_pts_data = resolve_fpl_names(
                pred_pts_data, elements_team[["web_name", "short_name"]]
            )
            for p in initial_squad:
                name = elements_team.loc[elements_team["id"] == p, "web_name"].item()
                team = elements_team.loc[elements_team["id"] == p, "short_name"].item()
                if pred_pts_data.loc[
                    (pred_pts_data["FPL name"] == name)
                    & (pred_pts_data["Team"] == team)
                ].empty:
                    pred_pts_data.loc[len(pred_pts_data)] = [name, team, None, None] + [
                        0 for i in range(len(pred_pts_data.columns) - 4)
                    ]
            merged_data = elements_team.merge(
                pred_pts_data,
                left_on=["web_name", "short_name"],
                right_on=["FPL name", "Team"],
            ).set_index("id")
            merged_data["sell_price"] = merged_data["now_cost"]
            # merged_data[f"xPts_{next_gw}"] = merged_data["xP"] # peek actual xp

            if player_history:
                picks_df, summary, next_gw_dict = get_historical_picks(
                    team_id, next_gw, merged_data
                )

            else:
                picks_df, summary, next_gw_dict = solve_multi_period_fpl(
                    {
                        "merged_data": merged_data,
                        "team_data": team_data,
                        "type_data": type_data,
                        "gameweeks": gameweeks,
                        "initial_squad": initial_squad,
                        "itb": itb,
                    },
                    options,
                )

            logger.info(summary[0])
            picks_df.to_csv("picks.csv", index=False, encoding="utf-8-sig")

            squad = picks_df.loc[picks_df["week"] == next_gw]
            squad = squad.merge(
                merged_data[["xP"]], left_on="id", right_index=True
            ).set_index("id")
            predicted_xp = (
                squad["predicted_xP"] * squad["multiplier"]
            ).sum() - next_gw_dict["hits"] * 4
            actual_xp = (squad["xP"] * squad["multiplier"]).sum() - next_gw_dict[
                "hits"
            ] * 4
            total_predicted_xp += predicted_xp
            total_xp += actual_xp
            logger.info(
                f"Predicted xP = {predicted_xp:.2f}. ({total_predicted_xp:.2f} overall)"
            )
            logger.info(f"Actual xP = {actual_xp:.2f}. ({total_xp:.2f} overall)")

            if not player_history:
                if next_gw_dict["chip_used"] not in ("fh", "wc") and next_gw != 1:
                    assert next_gw_dict["ft"] == min(
                        2, max(1, options["ft"] - next_gw_dict["n_transfers"] + 1)
                    )
                else:
                    assert next_gw_dict["ft"] == options["ft"]

            itb = next_gw_dict["itb"]
            options["ft"] = next_gw_dict["ft"]
            initial_squad = squad.index.to_list()

        result_gw.append(next_gw)
        result_xp.append(total_xp)
        result_predicted_xp.append(total_predicted_xp)
        result_solve_times.append(next_gw_dict["solve_time"])
        logger.info(
            f"Avg solve time: {sum(result_solve_times) / len(result_solve_times):.1f}"
        )

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(result_gw, result_xp, label="Actual xP")
    for index in range(len(result_gw)):
        ax.text(result_gw[index], result_xp[index], f"{result_xp[index]:.1f}", size=12)
    for index in range(len(result_gw)):
        ax.text(
            result_gw[index],
            result_predicted_xp[index],
            f"{result_predicted_xp[index]:.1f}",
            size=12,
        )

    plt.plot(result_gw, result_predicted_xp, linewidth=2.0, label="Predicted xP")
    plt.title(title)
    plt.legend()
    plt.savefig(f"[{total_predicted_xp:.1f}, {total_xp:.1f}] {title}.png")
    # plt.show()


if __name__ == "__main__":
    # players = {"Me": 3531385, "Hazard": 1195527, "FPL Raptor": 5431, "Donald": 307190}
    options = {
        "team_id": 3531385,
        "ft": 1,
        "horizon": 3,
        "tr_horizon": 2,
        "wc_on": 8,
        "timeout": 60,
        "log": False,
        "decay": 0.5,
    }
    logging.basicConfig(level=logging.INFO)

    live_run(options)

    # options["player_history"] = True
    # for p, id in players.items():
    #     options["team_id"] = id
    #     backtest(options, p)
