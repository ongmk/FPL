import math
from collections import defaultdict
from typing import Any, Dict

from deepdiff import DeepDiff
from flatten_dict import flatten, unflatten
from hyperopt import hp
from hyperopt.pyll.base import scope


def tuple_list_type_check(search_space: Dict, parameters: Dict):
    diff_result = DeepDiff(search_space, parameters).to_dict()
    type_changes_dict: Dict = diff_result.get("type_changes")
    if type_changes_dict is None:
        return parameters

    for diff_key, diff_details in type_changes_dict.items():
        if diff_details["old_type"] is list and diff_details["new_type"] is tuple:
            old_value = diff_details["old_value"]
            key = diff_key.replace("root", "parameters", 1)
            exec(f"{key} = old_value")

    return parameters


def find_search_groups(hyperopt_param: dict) -> list[dict[str, Any]]:
    hyperopt_group_params = (
        hyperopt_param.pop("groups") if "groups" in hyperopt_param else {}
    )
    flat_param_dict = flatten(hyperopt_group_params, reducer="tuple")
    if len(flat_param_dict.keys()) > 1:
        raise ValueError(f"Cannot group by more than one key.")
    if len(flat_param_dict.keys()) == 0:
        return [{}]
    else:
        key = list(flat_param_dict.keys())[0]
        groups = list(flat_param_dict.values())[0]
        return [unflatten({key: group}, splitter="tuple") for group in groups]


def construct_search_space(param_dict: Dict, hyper_param_dict: Dict) -> Dict:
    flat_param_dict = flatten(param_dict, reducer="tuple")
    flat_hyper_param_dict = flatten(hyper_param_dict, reducer="tuple")
    search_space = dict()

    for param_key, value in flat_param_dict.items():
        if not _is_param_key_in_hyper_param_key(
            param_key, flat_hyper_param_dict.keys()
        ):
            search_space[param_key] = value
        else:
            hyper_param_details = {
                hyper_param_key: hyper_param_value
                for hyper_param_key, hyper_param_value in flat_hyper_param_dict.items()
                if is_subset(param_key, hyper_param_key)
            }
            search_space[param_key] = _build_hyper_param_expression(
                param_key[-1], _unflatten_hyper_param_details_dict(hyper_param_details)
            )

    return unflatten(search_space, splitter="tuple")


def _build_hyper_param_expression(param_name: str, hyper_param_details: Dict):
    if "method" in hyper_param_details:
        default_hyper_dict = defaultdict(lambda: None, hyper_param_details)
        low = default_hyper_dict["low"]
        high = default_hyper_dict["high"]
        q = default_hyper_dict["q"]
        mu = default_hyper_dict["mu"]
        sigma = default_hyper_dict["sigma"]
        _scope = default_hyper_dict["scope"]
        method = hyper_param_details["method"]

        if method == "uniform":
            search_space = hp.uniform(param_name, low, high)
        elif method == "loguniform":
            low = round(math.log10(low))
            high = round(math.log10(high))
            search_space = hp.loguniform(param_name, low, high)
        elif method == "normal":
            search_space = hp.normal(param_name, mu, sigma)
        elif method == "lognormal":
            search_space = hp.lognormal(param_name, mu, sigma)
        elif method == "quniform":
            search_space = hp.quniform(param_name, low, high, q)
        elif method == "qloguniform":
            low = round(math.log10(low))
            high = round(math.log10(high))
            search_space = hp.qloguniform(param_name, low, high, q)
        elif method == "qnormal":
            search_space = hp.qnormal(param_name, mu, sigma, q)
        elif method == "qlognormal":
            search_space = hp.qlognormal(param_name, mu, sigma, q)
        elif method == "choice":
            if "values" in hyper_param_details:
                value_list = hyper_param_details["values"]
                search_space = hp.choice(param_name, value_list)
            else:
                search_space = hp.choice(param_name, range(low, high))
        elif method == "constant":
            if "value" in hyper_param_details:
                search_space = hyper_param_details["value"]
        else:
            raise ValueError(f"{param_name}: {method} is not a valid searching method.")
        return scope.int(search_space) if _scope == "int" else search_space
    else:
        raise KeyError(f"No searching method found for {param_name}.")


def _unflatten_hyper_param_details_dict(flat_details_dict: Dict) -> Dict:
    unflatten_dict = dict()
    for key, value in flat_details_dict.items():
        unflatten_dict[key[-1]] = value

    return unflatten_dict


def _is_param_key_in_hyper_param_key(param_key: str, hyper_param_key_list) -> bool:
    for hyper_param_key in hyper_param_key_list:
        if is_subset(param_key, hyper_param_key):
            return True
    return False


def is_subset(inner_tuple, outer_tuple):
    return set(inner_tuple).issubset(set(outer_tuple))


def update_parameters(old_dict: dict, new_dict: dict) -> dict:
    flat_old = flatten(old_dict, reducer="tuple")
    flat_new = flatten(new_dict, reducer="tuple")
    flat_old.update(flat_new)
    updated_old_dict = unflatten(flat_old, splitter="tuple")
    return updated_old_dict
