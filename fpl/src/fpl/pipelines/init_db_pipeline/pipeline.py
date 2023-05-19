import sqlite3
from kedro.pipeline import Pipeline, node
import logging
import os

logger = logging.getLogger(__name__)


def create_table(sql):
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
        create_table(path)
        logger.info(f"Created table with {sql}")
    return True


def copy_fpl_data(fpl_data_csv):
    return fpl_data_csv


def create_pipeline():
    return Pipeline(
        [
            # node(
            #     func=create_db_tables,
            #     inputs=None,
            #     outputs="done",
            #     name="create_db_tables_node",
            # ),
            node(
                func=copy_fpl_data,
                inputs="FPL_DATA_CSV",
                outputs="FPL_DATA",
                name="copy_fpl_data_node",
            ),
        ]
    )
