import logging

import pandas as pd
from kedro.extras.datasets.pandas.sql_dataset import SQLTableDataSet
from sqlalchemy import inspect

logger = logging.getLogger(__name__)


def pandas_dtype_to_sqlite(dtype):
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    elif pd.api.types.is_float_dtype(dtype):
        return "REAL"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    else:
        return "TEXT"


def expand_table(table_name: str, data: pd.DataFrame, engine):
    table_info = engine.execute(f"PRAGMA table_info({table_name})").fetchall()

    existing_columns = [info[1] for info in table_info]
    missing_columns = set(data.columns) - set(existing_columns)

    for column in missing_columns:
        column_type = pandas_dtype_to_sqlite(data[column].dtype)
        engine.execute(f"ALTER TABLE {table_name} ADD COLUMN {column} {column_type}")
        logger.info(f"Added column {column} to table {table_name}")

    return None


class FlexibleSQLTableDataSet(SQLTableDataSet):
    def _save(self, data: pd.DataFrame) -> None:
        engine = self.engines[self._connection_str]  # type: ignore

        expand_table(self._load_args["table_name"], data, engine)

        data.to_sql(con=engine, **self._save_args)


class ReadOnlySQLTableDataSet(SQLTableDataSet):
    def _load(self) -> pd.DataFrame:
        engine = self.engines[self._connection_str]  # type:ignore
        inspector = inspect(engine)
        table_name = self._load_args["table_name"]
        if inspector.has_table(table_name):
            return pd.read_sql_table(con=engine, **self._load_args)
        else:
            logger.warning(f"Table {table_name} doesn't exist. Returning None.")
            return None


class ExperimentMetrics(FlexibleSQLTableDataSet):
    def _save(self, data: tuple[int, dict[str, float]]) -> None:
        new_table = self._load()
        id, metrics = data
        values = list(metrics.values())
        metrics = list(metrics.keys())
        missing_columns = list(set(metrics) - set(new_table.columns))
        new_table[missing_columns] = None
        new_table.loc[new_table["id"] == id, metrics] = values
        self._save_args["if_exists"] = "replace"
        super()._save(new_table)


if __name__ == "__main__":
    print(SQLTableDataSet)
    print(ExperimentMetrics)
    print(FlexibleSQLTableDataSet)
