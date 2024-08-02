from dataclasses import dataclass

from pulp import LpVariable, lpSum


@dataclass
class LpParams:
    next_gw: int
    transfer_gws: int
    threshold_gw: int

    ft: int
    horizon: int
    wc_on: int
    bb_on: int
    fh_on: int

    wc_limit: int
    bb_limit: int
    fh_limit: int

    decay: float
    ft_bonus: float
    itb_bonus: float
    bench_weights: dict[int, float]


@dataclass
class LpKeys:
    element_types: list[int]
    teams: list[str]
    players: list[int]
    price_modified_players: list[int]
    all_gws: list[int]
    player_all_gws: list[tuple[int, int]]
    player_gameweeks: list[tuple[int, int]]
    order: list[int]
    player_gameweeks_order: list[tuple[int, int, int]]
    price_modified_players_gameweeks: list[tuple[int, int]]
    player_type: dict[int, int]
    sell_price: dict[int, float]
    buy_price: dict[int, float]


@dataclass
class LpVariables:
    squad: dict[tuple[int, int], LpVariable]
    squad_fh: dict[tuple[int, int], LpVariable]
    lineup: dict[tuple[int, int], LpVariable]
    captain: dict[tuple[int, int], LpVariable]
    vicecap: dict[tuple[int, int], LpVariable]
    bench: dict[tuple[int, int, int], LpVariable]
    transfer_in: dict[tuple[int, int], LpVariable]
    transfer_out: dict[tuple[int, int], LpVariable]
    transfer_out_first: dict[tuple[int, int], LpVariable]
    transfer_out_regular: dict[tuple[int, int], LpVariable]
    in_the_bank: dict[int, LpVariable]
    free_transfers: dict[int, LpVariable]
    penalized_transfers: dict[int, LpVariable]
    aux: dict[int, LpVariable]
    use_wc: dict[int, LpVariable]
    use_bb: dict[int, LpVariable]
    use_fh: dict[int, LpVariable]


@dataclass
class VariableSums:
    lineup_type_count: dict[tuple[int], lpSum]
    squad_type_count: dict[tuple[int], lpSum]
    squad_fh_type_count: dict[tuple[int], lpSum]
    sold_amount: dict[int, lpSum]
    fh_sell_price: dict[int, lpSum]
    bought_amount: dict[int, lpSum]
    points_player_week: dict[tuple[int], lpSum]
    squad_count: dict[int, lpSum]
    squad_fh_count: dict[int, lpSum]
    number_of_transfers: dict[int, lpSum]
    transfer_diff: dict[int, lpSum]
