import sqlite3

import numpy as np
import pandas as pd


def find_missing_integer(group, all_teams):
    missing_opponent = [
        team for team in all_teams if team not in group["opponent_team"].tolist()
    ]
    if len(missing_opponent) == 1:
        return missing_opponent[0]


def map_name_only(conn, cur, name_only_mapping):
    _filter = [old_name for old_name in name_only_mapping.keys()]
    _filter = ", ".join([f"'{item}'" for item in _filter])
    _filter = f"({_filter})"

    df = pd.read_sql(
        f"select distinct full_name from raw_fpl_data where full_name in {_filter}",
        conn,
    ).rename(columns={"full_name": "old_name"})
    df["new_name"] = df["old_name"].map(name_only_mapping)

    for idx, row in df.iterrows():
        old_name = row.old_name
        new_name = row.new_name
        query = f"UPDATE raw_fpl_data SET full_name = '{new_name}' WHERE full_name = '{old_name}'"
        cur.execute(query)
        if cur.rowcount > 0:
            print(f"{cur.rowcount} X {old_name} --> {new_name}")
        conn.commit()
    return None


def map_season_element(conn, cur, season_element_mapping):
    _filter = [old_name for _, _, old_name in season_element_mapping.keys()]
    _filter = ", ".join([f"'{item}'" for item in _filter])
    _filter = f"({_filter})"

    df = pd.read_sql(
        f"select distinct season, element, full_name from raw_fpl_data where full_name in {_filter}",
        conn,
    ).rename(columns={"full_name": "old_name"})
    df["new_name"] = df.apply(
        lambda row: season_element_mapping.get(
            (row["season"], row["element"], row["old_name"]), None
        ),
        axis=1,
    )

    for idx, row in df.iterrows():
        if row.new_name is None:
            continue
        season = row.season
        element = row.element
        old_name = row.old_name
        new_name = row.new_name
        query = f"""
        UPDATE raw_fpl_data SET full_name = '{new_name}'
        WHERE full_name = '{old_name}' and season = '{season}' and element = {element}
        """
        cur.execute(query)
        if cur.rowcount > 0:
            print(f"{cur.rowcount} X {season=}, {element=}, {old_name=} --> {new_name}")
        conn.commit()
    return None


if __name__ == "__main__":
    conn = sqlite3.connect("data/fpl.db")
    cur = conn.cursor()

    name_only_mapping = {
        "Lukasz Fabianski": "Łukasz Fabiański",
        "Seamus Coleman": "Séamus Coleman",
        "Martin Dubravka": "Martin Dúbravka",
        "Olu Aina": "Ola Aina",
        "Alexandre Moreno Lopera": "Álex Moreno Lopera",
        "Joe Ayodele-Aribo": "Joe Aribo",
        "Yegor Yarmoliuk": "Yehor Yarmoliuk",
        "Radu Dragusin": "Radu Drăgușin",
        "Olayinka Fredrick Oladotun Ladapo": "Freddie Ladapo",
    }
    map_name_only(conn, cur, name_only_mapping)

    # season_element_mapping = {
    #     ("2021-2022", 496, "Aaron Ramsey"): "Aaron James Ramsey",
    #     ("2023-2024", 675, "Aaron Ramsey"): "Aaron James Ramsey",
    #     ("2018-2019", 105, "Danny Ward"): "Daniel Carl Ward",
    # }
    # map_season_element(conn, cur, season_element_mapping)

    conn.close()
