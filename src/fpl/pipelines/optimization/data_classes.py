from dataclasses import dataclass

from pulp import LpVariable, lpSum
from pydantic import BaseModel

from fpl.utils import PydanticDataFrame


@dataclass
class LpParams:
    next_gw: int
    transfer_gws: int
    threshold_gw: int

    horizon: int
    wildcard_week: int
    bench_boost_week: int
    free_hit_week: int
    triple_captain_week: int

    decay: float
    free_transfer_bonus: float
    in_the_bank_bonus: float
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
    squad_free_hit: dict[tuple[int, int], LpVariable]
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
    use_wildcard: dict[int, LpVariable]
    use_bench_boost: dict[int, LpVariable]
    use_free_hit: dict[int, LpVariable]
    use_triple_captain: dict[tuple[int, int], LpVariable]


@dataclass
class VariableSums:
    lineup_type_count: dict[tuple[int], lpSum]
    squad_type_count: dict[tuple[int], lpSum]
    squad_free_hit_type_count: dict[tuple[int], lpSum]
    sold_amount: dict[int, lpSum]
    free_hit_sell_price: dict[int, lpSum]
    bought_amount: dict[int, lpSum]
    points_player_week: dict[tuple[int], lpSum]
    squad_count: dict[int, lpSum]
    squad_free_hit_count: dict[int, lpSum]
    number_of_transfers: dict[int, lpSum]
    transfer_diff: dict[int, lpSum]
    use_triple_captain_week: dict[int, lpSum]


class LpData(BaseModel):
    merged_data: PydanticDataFrame
    team_data: PydanticDataFrame
    type_data: PydanticDataFrame
    gameweeks: list[int]
    initial_squad: list[int]
    team_name: str
    in_the_bank: float
    free_transfers: int
    current_season: str

    class Config:
        arbitrary_types_allowed = True
