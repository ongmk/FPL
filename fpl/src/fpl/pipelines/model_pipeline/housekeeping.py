import sqlite3
import os


def _delete_from_db(table_name, recent_start_times, conn):
    cursor = conn.cursor()
    cursor.execute(
        f"DELETE FROM {table_name} WHERE start_time NOT IN ({','.join('?' * len(recent_start_times))})",
        recent_start_times,
    )
    conn.commit()
    return None


def _delete_from_path(relative_path, recent_start_times):
    file_list = os.listdir(relative_path)

    for file in file_list:
        timestamp = file.split("__")[0]

        if timestamp not in recent_start_times:
            file_path = os.path.join(relative_path, file)
            os.remove(file_path)
    return None


def run_housekeeping(loss, parameters):
    to_keep = parameters["to_keep"]

    conn = sqlite3.connect("./data/fpl.db")
    cursor = conn.cursor()

    # Get all unique start_time values
    cursor.execute('SELECT DISTINCT start_time FROM "evaluation_result"')
    unique_start_times = [row[0] for row in cursor.fetchall()]

    # Sort start_time values in descending order and select the most recent ones
    unique_start_times.sort(reverse=True)
    recent_start_times = unique_start_times[:to_keep]

    _delete_from_db('"evaluation_result"', recent_start_times, conn)
    _delete_from_path("./data/evaluation", recent_start_times)

    conn.close()
    return loss
