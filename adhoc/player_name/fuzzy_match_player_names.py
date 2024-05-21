import logging
import sqlite3

import pandas as pd
import yaml
from thefuzz import process
from tqdm import tqdm

logger = logging.getLogger(__name__)


def fuzzy_match_player_names(
    player_match_log: pd.DataFrame,
    fpl_data: pd.DataFrame,
    fbref2fpl_player_overrides: dict[str, str],
) -> None:
    fpl_data = (
        fpl_data[["full_name", "total_points"]]
        .groupby(["full_name"], as_index=False)
        .sum()
        .rename(columns={"full_name": "fpl_name"})
    )

    player_match_log = player_match_log[["link", "player"]].rename(
        columns={"player": "fbref_name", "link": "fbref_id"}
    )
    player_match_log["fbref_id"] = player_match_log["fbref_id"].str.extract(
        r"/players/([a-f0-9]+)/"
    )
    matched_df = player_match_log.drop_duplicates().copy()
    tqdm.pandas(desc=f"Fuzzy matching player names")
    matched_df["fpl_name"] = (
        matched_df["fbref_name"]
        .progress_apply(
            lambda fbref_name: process.extract(
                fbref_name, fpl_data["fpl_name"].tolist(), limit=1
            )
        )
        .explode()
    )
    name_only_mapping = {
        fbref_name: (fpl_name, 100)
        for fbref_name, fpl_name in fbref2fpl_player_overrides[
            "name_only_mapping"
        ].items()
    }
    id_name_mapping = {
        (fbref_id, fbref_name): (fpl_name, 100)
        for fbref_id, fbref_name, fpl_name in fbref2fpl_player_overrides[
            "id_name_mapping"
        ]
    }
    matched_df["fpl_name"] = (
        matched_df["fbref_name"].map(name_only_mapping).fillna(matched_df["fpl_name"])
    )

    matched_df["fpl_name"] = matched_df.apply(
        lambda row: id_name_mapping.get((row["fbref_id"], row["fbref_name"]), None),
        axis=1,
    ).fillna(matched_df["fpl_name"])

    matched_df["fuzzy_score"] = matched_df["fpl_name"].str[1]
    matched_df["fpl_name"] = matched_df["fpl_name"].str[0]

    matched_df = pd.merge(matched_df, fpl_data, on=["fpl_name"], how="outer")
    matched_df.loc[
        ((matched_df["fuzzy_score"] < 90) | (matched_df["fuzzy_score"].isna()))
        & (matched_df["total_points"] > 0),
        "review",
    ] = 1
    matched_df["duplicated"] = (
        matched_df["fpl_name"].duplicated(keep=False).map({True: 1, False: pd.NA})
    )
    matched_df.loc[
        (matched_df["fbref_name"].isna()) & (matched_df["total_points"] > 0),
        "missing_matchlogs",
    ] = 1
    if matched_df["review"].sum() > 0:
        logger.warning(
            f"{int(matched_df['review'].sum())}/{len(matched_df)} records in player name mappings needs review."
        )
    if matched_df["duplicated"].sum() > 0:
        logger.error(
            f"{int(matched_df['duplicated'].sum())} duplicated player name mappings."
        )
    if matched_df["missing_matchlogs"].sum() > 0:
        logger.error(
            f"There are missing FBRef matchlogs for {int(matched_df['missing_matchlogs'].sum())} players."
        )
    matched_df.to_csv("data/preprocess/player_name_mapping.csv", index=False)
    return None


if __name__ == "__main__":
    conn = sqlite3.connect("data/fpl.db")
    cur = conn.cursor()

    player_match_log = pd.read_sql(
        f"select * from raw_player_match_log",
        conn,
    )
    fpl_data = pd.read_sql(
        f"select * from raw_fpl_data",
        conn,
    )
    conn.close()

    with open(
        "adhoc/player_name/fbref2fpl_player_overrides.yml", "r", encoding="utf-8"
    ) as file:
        fbref2fpl_player_overrides = yaml.safe_load(file)

    fuzzy_match_player_names(player_match_log, fpl_data, fbref2fpl_player_overrides)
