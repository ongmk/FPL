import logging
from typing import Dict

import numpy as np
from hyperopt import Trials, anneal, atpe, rand, tpe
from hyperopt.pyll.base import Apply

from fpl.custom_run.hyperopt_helpers import construct_search_space
from fpl.custom_run.hyperopt_helpers.termination_policy import (
    BanditParam,
    PercentileStoppingParam,
    bandit_policy,
    percentile_stopping_policy,
)

logger = logging.getLogger(__name__)


def build_run_config(param_dict: Dict, hyperopt_param: Dict, target_dict: dict):
    max_trials = target_dict["max_trials"]
    strategy = target_dict["strategy"]
    target_name = target_dict["name"]
    algo = target_dict.get("algo", "tpe")
    early_stop_config = target_dict.get("early_termination", None)

    search_space = construct_search_space(param_dict, hyperopt_param)

    # Validation
    max_trials = _validate_max_trials(max_trials, search_space)
    _validate_strategy(strategy)
    _validate_target(target_name)

    hyperopt_algo, algo = _validate_algo(algo)

    early_stop_policy = _validate_early_termination_policy(early_stop_config)

    trials = Trials("")

    return {
        "space": search_space,
        "algo": hyperopt_algo,
        "max_evals": max_trials,
        "trials": trials,
        "early_stop_fn": early_stop_policy,
        "return_argmin": False,
    }, {
        "algo": algo,
        "target_name": target_name,
        "strategy": strategy,
        "uuid": hyperopt_param.get("uuid", True),
    }


def recursiveFindNodes(root, node_type="switch"):
    nodes = []
    if isinstance(root, (list, tuple)):
        for node in root:
            nodes.extend(recursiveFindNodes(node, node_type))
    elif isinstance(root, dict):
        for node in root.values():
            nodes.extend(recursiveFindNodes(node, node_type))
    elif isinstance(root, (Apply)):
        if root.name == node_type:
            nodes.append(root)

        for node in root.pos_args:
            if node.name == node_type:
                nodes.append(node)
        for _, node in root.named_args:
            if node.name == node_type:
                nodes.append(node)
    return nodes


def parameters(space):
    # Analyze the domain instance to find parameters
    parameters = {}
    if isinstance(space, dict):
        space = list(space.values())
    for node in recursiveFindNodes(space, "switch"):

        # Find the name of this parameter
        paramNode = node.pos_args[0]
        assert paramNode.name == "hyperopt_param"
        paramName = paramNode.pos_args[0].obj

        # Find all possible choices for this parameter
        values = [literal.obj for literal in node.pos_args[1:]]
        parameters[paramName] = np.array(range(len(values)))
    return parameters


def spacesize(space):
    # Compute the number of possible combinations
    params = parameters(space)
    return np.prod([len(values) for values in params.values()])


def _validate_max_trials(max_trials, space):
    space_size = spacesize(space)
    if max_trials > space_size:
        logger.warn(
            f"Max trials > possible combinations. Setting max trials to {space_size}"
        )
        return space_size
    if max_trials is None or max_trials < 1 or not isinstance(max_trials, int):
        raise ValueError("Please provide a valid number for max_trials")
    return max_trials


def _validate_strategy(strategy):
    if strategy is None or strategy not in ["min", "max"]:
        raise ValueError("Please provide a valid strategy")


def _validate_target(target_name):
    if target_name is None:
        raise ValueError("Please provide name of metric to be optimized")


def _validate_algo(algo):
    if algo is not None:
        if algo == "tpe":
            hyperopt_algo = tpe.suggest
        elif algo == "atpe":
            hyperopt_algo = atpe.suggest
        elif algo == "random":
            hyperopt_algo = rand.suggest
        elif algo == "anneal":
            hyperopt_algo = anneal.suggest
        else:
            raise ValueError(
                "Please provide a valid algorithm for the optimizer. Supported algorithms: tpe, atpe, random, anneal "
            )
    else:
        hyperopt_algo = tpe.suggest
        algo = "tpe"

    return hyperopt_algo, algo


def _validate_early_termination_policy(policy_dict: Dict):
    if policy_dict is None:
        return None

    policy = policy_dict.get("policy", None)

    if policy is None or policy not in ["bandit", "percentile_stopping"]:
        raise ValueError(f"{policy} is not a valid policy")

    delay_evaluation = policy_dict.get("delay_evaluation", None)
    evaluation_interval = policy_dict.get("evaluation_interval", None)

    if (
        delay_evaluation is None
        or not isinstance(delay_evaluation, int)
        or delay_evaluation < 0
    ):
        raise ValueError(
            f"{delay_evaluation} is not a valid delay evaluation. Delay evaluation must be a non-negative integer"
        )

    if (
        evaluation_interval is None
        or not isinstance(evaluation_interval, int)
        or evaluation_interval < 0
    ):
        raise ValueError(
            f"{evaluation_interval} is not a valid evaluation evaluation. Evaluation evaluation must be a non-negative integer"
        )

    if policy == "bandit":
        slack_amount = policy_dict.get("slack_amount", None)

        if (
            slack_amount is None
            or not isinstance(slack_amount, float)
            or slack_amount <= 0
        ):
            raise ValueError(
                f"{slack_amount} is not a valid slack amount. Slack amount must be a positive number"
            )

        BanditParam.init(
            evaluation_interval=evaluation_interval,
            delay_evaluation=delay_evaluation,
            slack_amount=slack_amount,
        )
        return bandit_policy
    elif policy == "percentile_stopping":
        percentile = policy_dict.get("percentile", None)

        if percentile is None or percentile < 0 or percentile > 100:
            raise ValueError(
                f"{percentile} is not a valid percentile. Percentile must be between 0-100"
            )

        PercentileStoppingParam.init(
            evaluation_interval=evaluation_interval,
            delay_evaluation=delay_evaluation,
            percentile=percentile,
        )
        return percentile_stopping_policy
