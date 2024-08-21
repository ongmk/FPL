import logging
import os
import sqlite3

import pandas as pd
from kedro.pipeline import Pipeline, node

logger = logging.getLogger(__name__)


def execute_sql(sql):
    conn = sqlite3.connect("./data/fpl.db")
    cursor = conn.cursor()
    sql_file = open(sql).read()
    cursor.executescript(sql_file)
    conn.commit()
    conn.close()


def create_db_tables():
    sqls = [
        "drop_tables.sql",
        "create_tables.sql",
    ]

    for sql in sqls:
        path = os.path.join("./src/fpl/pipelines/init_db_pipeline/", sql)
        execute_sql(path)
        logger.info(f"Executed SQL {sql}")
    return True


def copy_fpl_data_to_db(_):
    return pd.read_csv("data/fpl_history_backup.csv")


init_db_pipeline = Pipeline(
    [
        node(
            func=create_db_tables,
            inputs=None,
            outputs="done",
        ),
        node(
            func=copy_fpl_data_to_db,
            inputs="done",
            outputs="FPL_DATA",
        ),
    ]
)
