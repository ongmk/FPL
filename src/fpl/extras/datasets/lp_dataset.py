import logging
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any, Dict

import fsspec
import pandas as pd
from kedro.extras.datasets.json.json_dataset import JSONDataSet
from kedro.io.core import (
    AbstractVersionedDataset,
    DatasetError,
    Version,
    get_filepath_str,
    get_protocol_and_path,
)
from pulp import LpProblem
from pydantic import BaseModel

from fpl.utils import load_obj

logger = logging.getLogger(__name__)


def get_full_class_path(obj):
    cls = obj.__class__
    module = cls.__module__
    return module + "." + cls.__qualname__


class PydanticDataset(JSONDataSet):

    def __init__(self, *args, **kwargs):
        kwargs["fs_args"] = dict(open_args_save=dict(encoding="utf-8"))
        kwargs["save_args"] = dict(ensure_ascii=False)
        super().__init__(*args, **kwargs)

    def _save(self, obj: BaseModel) -> None:
        data = obj.model_dump()
        data = {
            "class_path": get_full_class_path(obj),
            "data": data,
        }
        super()._save(data)

    def _load(self) -> pd.DataFrame:
        data = super()._load()
        class_path = data["class_path"]
        cls = load_obj(class_path)
        data = data["data"]
        return cls(**data)


if __name__ == "__main__":
    print(PydanticDataset)
