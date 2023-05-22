from kedro_datasets.pandas import SQLTableDataSet
import pandas as pd
import logging

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


class FlexibleSQLTableDataSet(SQLTableDataSet):
    def _save(self, data: pd.DataFrame) -> None:
        engine = self.engines[self._connection_str]  # type: ignore

        table_name = self._load_args["table_name"]
        table_info = engine.execute(f"PRAGMA table_info({table_name})").fetchall()

        existing_columns = [info[1] for info in table_info]
        missing_columns = set(data.columns) - set(existing_columns)

        for column in missing_columns:
            column_type = pandas_dtype_to_sqlite(data[column].dtype)
            # SQLite does not have explicit data types, so we use 'TEXT' as a generic type
            engine.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column} {column_type}"
            )
            logger.info(f"Added column {column} to table {table_name}")

        data.to_sql(con=engine, **self._save_args)
