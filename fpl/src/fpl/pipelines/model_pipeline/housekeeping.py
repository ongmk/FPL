import sqlite3
import os


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

    conn.close()
    return None
