from dataclasses import dataclass
from typing import Any, Literal, Optional

import pandas as pd
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
    gameweeks_plus: list[int]
    player_gameweeks_plus: list[tuple[int, int]]
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
    in_the_bank: dict[int, LpVariable]  # after transfers
    free_transfers: dict[int, LpVariable]  # after transfers
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


@dataclass
class StartingParams:
    gameweek: int
    free_transfers: int
    in_the_bank: float


@dataclass
class GwResults:
    gameweek: int
    transfer_data: list[dict[str, Any]]
    captain: int
    vicecap: int
    lineup: list[int]
    bench: dict[int, int]
    chip_used: Optional[
        Literal["wildcard", "bench_boost", "free_hit", "triple_captain"]
    ]
    hits: int
    total_predicted_points: float
    total_actual_points: float
    free_transfers: int
    in_the_bank: float
    starting_params: StartingParams
    player_details: pd.DataFrame


TYPE_DATA = [
    {
        "id": 1,
        "singular_name_short": "GKP",
        "squad_select": 2,
        "squad_min_play": 1,
        "squad_max_play": 1,
    },
    {
        "id": 2,
        "singular_name_short": "DEF",
        "squad_select": 5,
        "squad_min_play": 3,
        "squad_max_play": 5,
    },
    {
        "id": 3,
        "singular_name_short": "MID",
        "squad_select": 5,
        "squad_min_play": 2,
        "squad_max_play": 5,
    },
    {
        "id": 4,
        "singular_name_short": "FWD",
        "squad_select": 3,
        "squad_min_play": 1,
        "squad_max_play": 3,
    },
]
