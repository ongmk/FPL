import logging
import os
import sqlite3

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
        # "drop_tables.sql",
        "create_tables.sql",
    ]

    for sql in sqls:
        path = os.path.join("./src/fpl/pipelines/init_db_pipeline/", sql)
        execute_sql(path)
        logger.info(f"Executed SQL {sql}")
    return True


def create_pipeline():
    return Pipeline(
        [
            node(
                func=create_db_tables,
                inputs=None,
                outputs="done",
                name="create_db_tables_node",
            ),
        ]
    )
