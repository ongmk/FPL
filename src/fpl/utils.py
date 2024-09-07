import importlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema
from typing_extensions import Annotated


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """Extract an object from a given path.

    Args:
        obj_path: Path to an object to be extracted, including the object name.
        default_obj_path: Default object path.

    Returns:
        Extracted object.

    Raises:
        AttributeError: When the object does not have the given named attribute.

    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f"Object '{obj_name}' cannot be loaded from '{obj_path}'.")
    return getattr(module_obj, obj_name)


def read_dict_as_df(x):
    if isinstance(x, dict):
        x = {int(k): v for k, v in x.items()}
        return pd.DataFrame.from_dict(x, orient="index")
    elif isinstance(x, pd.DataFrame):
        return x
    else:
        raise ValueError(f"Expected dict or DataFrame, got {type(x)}")


PydanticDataFrame = Annotated[
    pd.DataFrame,
    BeforeValidator(read_dict_as_df),
    PlainSerializer(lambda x: x.replace({np.nan: None}).to_dict(orient="index")),
    WithJsonSchema({"type": "object"}, mode="serialization"),
]


def backup_latest_n(current_file, n=5):
    if isinstance(current_file, str):
        current_file = Path(current_file)
    dir = current_file.parent

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(
        current_file,
        str(dir / f"{current_file.stem}_{current_time}.bak") + current_file.suffix,
    )

    files_with_timestamps = sorted(
        [f for f in dir.glob(f"{current_file.stem}*.bak*") if f.is_file()],
        key=lambda f: "".join(f.stem.split("_")[-2:]),
        reverse=True,
    )

    for old_file in files_with_timestamps[n:]:
        old_file.unlink()
