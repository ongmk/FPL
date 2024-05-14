from pulp import (
    LpProblem,
    LpMinimize,
    LpBinary,
    LpVariable,
    LpContinuous,
    LpInteger,
    lpSum,
)
import itertools
from dataclasses import dataclass
from fpl.pipelines.optimization_pipeline.fpl_api import FplData
import pandas as pd


@dataclass
class LpData:
    squad: dict
    squad_fh: dict
    transfer_in: dict
    transfer_out: dict
    use_fh: dict
    use_wc: dict
    use_bb: dict
    captain: dict
    vicecap: dict
    lineup: dict
    bench: dict
    buy_price: dict
    sell_price: dict
    points_player_week: dict
    free_transfers: dict
    penalized_transfers: dict
    in_the_bank: dict

    gameweeks: list[int]
    next_gw: int
    transfer_gws: list[int]
    players: list[int]
    order: list[int]
    price_modified_players: list[int]

    merged_data: pd.DataFrame
    type_data: pd.DataFrame


def construct_lp(
    fpl_data: FplData, parameters: dict, lp_file_path: str
) -> tuple[LpProblem, LpData]:
    # Arguments
    ft = parameters["ft"]
    decay = parameters["decay"]
    horizon = parameters["horizon"]
    tr_horizon = parameters["tr_horizon"]
    wc_on = parameters["wc_on"]
    bb_on = parameters["bb_on"]
    fh_on = parameters["fh_on"]
    ft_bonus = parameters["ft_bonus"]
    itb_bonus = parameters["itb_bonus"]
    bench_weights = parameters["bench_weights"]

    # Data
    merged_data = fpl_data.merged_data
    keys = [k for k in merged_data.columns.to_list() if "xPts_" in k]
    merged_data["total_ev"] = merged_data[keys].sum(axis=1)
    merged_data = merged_data.sort_values(by=["total_ev"], ascending=[False])
    team_data = fpl_data.team_data
    type_data = fpl_data.type_data
    gameweeks = fpl_data.gameweeks
    initial_squad = fpl_data.initial_squad
    itb = fpl_data.itb
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
        + ft_bonus * free_transfers[w]
        + itb_bonus * in_the_bank[w]
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
    model.writeLP(lp_file_path)

    return model, LpData(
        squad=squad,
        squad_fh=squad_fh,
        transfer_in=transfer_in,
        transfer_out=transfer_out,
        use_fh=use_fh,
        use_wc=use_wc,
        use_bb=use_bb,
        captain=captain,
        vicecap=vicecap,
        lineup=lineup,
        bench=bench,
        buy_price=buy_price,
        sell_price=sell_price,
        points_player_week=points_player_week,
        free_transfers=free_transfers,
        penalized_transfers=penalized_transfers,
        in_the_bank=in_the_bank,
        gameweeks=gameweeks,
        next_gw=next_gw,
        transfer_gws=transfer_gws,
        players=players,
        order=order,
        price_modified_players=price_modified_players,
        merged_data=merged_data,
        type_data=type_data,
    )
