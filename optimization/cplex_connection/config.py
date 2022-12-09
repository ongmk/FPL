import json
import os
from configparser import ConfigParser
from pathlib import PurePath

from logzero import logger


class Config:
    _config_path = os.environ.get('CONFIG_PATH', '')
    if _config_path == '':
        _config_path_filename = 'optimization/cplex_connection/env_config.ini'
    else:
        _config_path_filename = PurePath(_config_path) / 'env_config.ini'

    _config = ConfigParser()
    _config.read(_config_path_filename)

    @classmethod
    def get(cls, section, key):
        return cls._config.get(section, key)

    @classmethod
    def get_boolean(cls, section, key):
        return cls._config.getboolean(section, key)

    @classmethod
    def get_int(cls, section, key):
        return cls._config.getint(section, key)

    @classmethod
    def get_float(cls, section, key):
        return cls._config.getfloat(section, key)

    @classmethod
    def get_list(cls, section, key):
        return cls._config.get(section, key).split(',')

    @classmethod
    def get_dict(cls, section, key):
        return json.loads(cls._config[section][key].replace('\'', '\"'))

    @classmethod
    def get_purepath(cls, section, key):
        return PurePath(cls._config[section][key])

    # TODO temp method for compatibility
    @classmethod
    def get_section(cls, section):
        return cls._config[section]

    @classmethod
    def set(cls, section, key, value):
        if section not in cls._config.keys():
            cls._config[section] = {}
        cls._config[section][key] = value
