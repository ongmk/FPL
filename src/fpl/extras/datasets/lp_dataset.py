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


class PuLpDataset(AbstractVersionedDataset[str, str]):

    def __init__(
        self,
        filepath: str,
        version: Version = None,
        credentials: Dict[str, Any] = None,
        fs_args: Dict[str, Any] = None,
    ) -> None:
        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        protocol, path = get_protocol_and_path(filepath, version)
        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

    def _describe(self) -> Dict[str, Any]:
        return {
            "filepath": self._filepath,
            "protocol": self._protocol,
            "version": self._version,
        }

    def _load(self) -> str:
        load_path = get_filepath_str(self._get_load_path(), self._protocol)
        return load_path

    def _save(self, model: LpProblem) -> None:
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        model.writeLP(save_path)

    def _exists(self) -> bool:
        try:
            load_path = get_filepath_str(self._get_load_path(), self._protocol)
        except DatasetError:
            return False

        return self._fs.exists(load_path)

    def _release(self) -> None:
        super()._release()


if __name__ == "__main__":
    print(DataClassDataset)
    print(PuLpDataset)
