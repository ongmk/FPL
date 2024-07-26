import importlib
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


PydanticDataFrame = Annotated[
    pd.DataFrame,
    BeforeValidator(lambda x: pd.DataFrame(x) if isinstance(x, list) else x),
    PlainSerializer(
        lambda x: x.replace({np.nan: None}).to_dict(
            orient="records",
        )
    ),
    WithJsonSchema({"type": "object"}, mode="serialization"),
]
