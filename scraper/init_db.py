import sqlite3


def create_table(sql):
    conn = sqlite3.connect("fpl.db")
    cursor = conn.cursor()
    sql_file = open(sql).read()
    cursor.executescript(sql_file)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    sqls = [
        # "create_team_match_log.sql",
        # "create_player_match_log.sql",
        # "create_match_odds.sql"
    ]

    for sql in sqls:
        create_table(sql)
