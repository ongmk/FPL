import sqlite3
import os
from datetime import datetime
from flatten_dict import flatten
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def _delete_from_db(table_name, keep_start_times, conn):
    cursor = conn.cursor()
    cursor.execute(
        f"DELETE FROM {table_name} WHERE start_time NOT IN ({','.join('?' * len(keep_start_times))})",
        keep_start_times,
    )
    conn.commit()
    return None


def _delete_from_path(relative_path, keep_start_times):
    file_list = os.listdir(relative_path)

    for file in file_list:
        timestamp = file.split("__")[0]

        if timestamp not in keep_start_times:
            file_path = os.path.join(relative_path, file)
            os.remove(file_path)
    return None


def delete_column(cursor, table_name, empty_column):
    cursor.execute(f"PRAGMA table_info({table_name})")
    all_columns = [col_info[1] for col_info in cursor.fetchall()]
    remaining_columns = [col for col in all_columns if col != empty_column]

    cursor.execute(
        f"CREATE TABLE temp_table AS SELECT {','.join(remaining_columns)} FROM {table_name}"
    )
    cursor.execute(f"DROP TABLE {table_name}")
    cursor.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_table")
    cursor.execute("DROP TABLE temp_table")
    logger.info(f"Deleted column {empty_column} from table {table_name}")
    return None


def delete_empty_columns(cursor, table_name):
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 1")
    col_names = [desc[0] for desc in cursor.description if desc[0] != "keep"]

    for col_name in col_names:
        cursor.execute(
            f"SELECT COUNT(*) FROM {table_name} WHERE {col_name} IS NOT NULL"
        )
        count = cursor.fetchone()[0]
        if count == 0:
            delete_column(cursor=cursor, table_name=table_name, empty_column=col_name)

    return None


def run_housekeeping(parameters):
    to_keep = parameters["to_keep"]

    conn = sqlite3.connect("./data/fpl.db")
    cursor = conn.cursor()

    # Get all unique start_time values
    cursor.execute("SELECT start_time FROM experiment")
    unique_start_times = [row[0] for row in cursor.fetchall()]

    unique_start_times.sort(reverse=True)
    recent_start_times = unique_start_times[:to_keep]
    cursor.execute("SELECT start_time FROM experiment where keep == 1")
    keep_start_times = [row[0] for row in cursor.fetchall()]
    keep_start_times = list(set(keep_start_times + recent_start_times))

    _delete_from_db("experiment", keep_start_times, conn)
    _delete_from_db("evaluation_result", keep_start_times, conn)
    _delete_from_path("./data/evaluation", keep_start_times)
    delete_empty_columns(cursor, "experiment")

    conn.close()
    return None


def snake_to_camel(snake_str):
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def camel_reducer(k1, k2):
    k2 = snake_to_camel(k2)
    if k1 is None:
        return k2
    else:
        k1 = snake_to_camel(k1)
        return f"{k1}_{k2}"


def init_experiment(parameters):
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
    experiment_record = pd.DataFrame.from_records([record])

    return id, start_time, experiment_record
