import logging
import os
import sqlite3
import subprocess
from datetime import datetime
from sqlite3 import Connection
from typing import Any

import pandas as pd
from flatten_dict import flatten

logger = logging.getLogger(__name__)


def delete_from_db(table_name: str, keep_start_times: list[str], conn: Connection):
    cursor = conn.cursor()
    cursor.execute(
        f"DELETE FROM {table_name} WHERE start_time NOT IN ({','.join('?' * len(keep_start_times))})",
        keep_start_times,
    )
    conn.commit()
    return None


def delete_from_path(relative_path: str, keep_start_times: list[str]):
    file_list = os.listdir(relative_path)

    for file in file_list:
        timestamp = file.split("__")[0]

        if timestamp not in keep_start_times:
            file_path = os.path.join(relative_path, file)
            os.remove(file_path)
    return None


def delete_column(conn: Connection, table_name: str, empty_column: str):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    all_columns = [col_info[1] for col_info in cursor.fetchall()]
    remaining_columns = [col for col in all_columns if col != empty_column]

    cursor.execute(
        f"CREATE TABLE temp_table AS SELECT {','.join(remaining_columns)} FROM {table_name}"
    )
    cursor.execute(f"DROP TABLE {table_name}")
    cursor.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_table")
    cursor.execute("DROP TABLE temp_table")
    conn.commit()
    logger.info(f"Deleted column {empty_column} from table {table_name}")
    return None


def delete_empty_columns(table_name: str, conn: Connection):
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    row_count = cursor.fetchone()[0]
    if row_count == 0:
        return None

    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
    col_names = [desc[0] for desc in cursor.description if desc[0] != "keep"]

    for col_name in col_names:
        cursor.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NOT NULL"
        )
        count = cursor.fetchone()[0]
        if count == 0:
            delete_column(conn=conn, table_name=table_name, empty_column=col_name)

    return None


def ensure_metric_col_type(conn: Connection, metric_column: str):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info(experiment)")
    columns = cursor.fetchall()

    column_data = next((col for col in columns if col[1] == metric_column), None)
    if column_data:
        column_type = column_data[2]
        if column_type != "FLOAT":
            raise TypeError(
                f'Metric column {metric_column} is not a float in table "experiment"'
            )
        else:
            return metric_column
    else:
        logger.warning(f"Metric column {metric_column} not found. Revert to sort by id")
        return "id"


def update_model_best(conn: Connection, metric_column: str, top_n: int, maximize: bool):
    metric_column = ensure_metric_col_type(conn, metric_column)

    cursor = conn.cursor()

    # Set all model_best values to NULL
    cursor.execute(f"UPDATE experiment SET model_best = NULL")
    conn.commit()

    # Get distinct groups based on models, numerical_features, and categorical_features columns
    cursor.execute(
        f"SELECT DISTINCT models, numericalFeatures, categoricalFeatures FROM experiment"
    )
    groups = cursor.fetchall()

    # Get the top n rows based on the metric_column within each group
    order = "DESC" if maximize else "ASC"
    for group in groups:
        models, numericalFeatures, categoricalFeatures = group
        cursor.execute(
            f"""
            UPDATE experiment
            SET model_best = 1
            WHERE id IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        ORDER BY {metric_column} {order}
                    ) AS rank
                    FROM experiment
                    WHERE models = ? AND numericalFeatures = ? AND categoricalFeatures = ?
                )
                WHERE rank <= {top_n}
            )
        """,
            (models, numericalFeatures, categoricalFeatures),
        )
        conn.commit()
    return None


def run_housekeeping(parameters: dict[str, Any]):
    to_keep = parameters["to_keep"]
    metric = parameters["metric"]
    keep_model_best = parameters["keep_model_best"]
    maximize = True if parameters["strategy"] == "max" else False

    conn = sqlite3.connect("./data/fpl.db")
    cursor = conn.cursor()

    # Get all unique start_time values
    cursor.execute("SELECT start_time FROM experiment")
    unique_start_times = [row[0] for row in cursor.fetchall()]
    update_model_best(conn, metric, keep_model_best, maximize)

    unique_start_times.sort(reverse=True)
    recent_start_times = unique_start_times[:to_keep]
    cursor.execute(
        "SELECT start_time FROM experiment where keep == 1 or model_best == 1"
    )
    keep_start_times = [row[0] for row in cursor.fetchall()]
    keep_start_times = list(set(keep_start_times + recent_start_times))

    delete_from_db("experiment", keep_start_times, conn)
    delete_from_db("evaluation_result", keep_start_times, conn)
    delete_from_path("./data/evaluation", keep_start_times)
    delete_empty_columns("experiment", conn)

    conn.close()
    return None


def snake_to_camel(snake_str: str) -> str:
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def camel_reducer(k1: str, k2: str) -> str:
    k2 = snake_to_camel(k2)
    if k1 is None:
        return k2
    else:
        k1 = snake_to_camel(k1)
        return f"{k1}_{k2}"


def get_latest_git_commit_message():
    try:
        commit_message = (
            subprocess.check_output(["git", "log", "-1", "--pretty=%B"])
            .decode("utf-8")
            .strip()
        )
        return commit_message
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        return None


def init_experiment(parameters: dict[str, Any]) -> tuple[int, str, pd.DataFrame]:
    conn = sqlite3.connect("./data/fpl.db")
    query = "select COALESCE(max(id)  + 1 , 0) from experiment;"
    cursor = conn.execute(query)
    id = cursor.fetchone()[0]
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cursor.close()
    conn.close()

    record = flatten(parameters, reducer=camel_reducer)
    keys_to_remove = []
    for key, value in record.items():
        if isinstance(value, list):
            record[key] = ", ".join(str(x) for x in value)
        if value is None:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        record.pop(key)

    record["id"] = id
    record["start_time"] = start_time
    record["git_message"] = get_latest_git_commit_message()
    experiment_record = pd.DataFrame.from_records([record])

    return id, start_time, experiment_record
