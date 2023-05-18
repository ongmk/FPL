import sqlite3
import pandas as pd
from kedro.pipeline import Pipeline, node, pipeline


def create_table(sql):
    conn = sqlite3.connect("fpl.db")
    cursor = conn.cursor()
    sql_file = open(sql).read()
    cursor.executescript(sql_file)
    conn.commit()
    conn.close()


def copy_fpl_data(fpl_data_csv):
    return fpl_data_csv


def create_pipeline():
    return Pipeline(
        [
            node(
                func=copy_fpl_data,
                inputs="FPL_DATA_CSV",
                outputs="FPL_DATA",
                name="copy_fpl_data_node",
            )
        ]
    )


if __name__ == "__main__":
    sqls = [
        # "create_team_match_log.sql",
        # "create_player_match_log.sql",
        # "create_match_odds.sql"
    ]

    for sql in sqls:
        create_table(sql)

    # fpl_data_to_sql()
