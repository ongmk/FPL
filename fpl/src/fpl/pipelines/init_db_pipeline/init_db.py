import sqlite3
import pandas as pd


def create_table(sql):
    conn = sqlite3.connect("fpl.db")
    cursor = conn.cursor()
    sql_file = open(sql).read()
    cursor.executescript(sql_file)
    conn.commit()
    conn.close()


def fpl_data_to_sql():
    conn = sqlite3.connect("../data/fpl.db")
    df = pd.read_csv(
        "../data/backtest_data/merged_seasons.csv", index_col=0, dtype={"team_x": str}
    )
    df.to_sql("01_FPL_DATA", conn, if_exists="replace", index=False)
    conn.close()


if __name__ == "__main__":
    sqls = [
        # "create_team_match_log.sql",
        # "create_player_match_log.sql",
        # "create_match_odds.sql"
    ]

    for sql in sqls:
        create_table(sql)

    fpl_data_to_sql()
