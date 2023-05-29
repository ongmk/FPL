import logging
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Dict

from kedro.framework.cli.utils import KedroCliError
from kedro.framework.session import KedroSession
from kedro.utils import load_obj
from textwrap import dedent
import yaml


def dict_nested_update(lhs: dict, rhs: dict):
    for k, v in rhs.items():
        if isinstance(v, dict):
            lhs[k] = dict_nested_update(lhs.get(k, {}) or {}, v)
        elif v is not None:
            lhs[k] = v
    return lhs


def read_base_params():
    with open("./conf/base/parameters.yml", "r") as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    return params


def custom_kedro_run(
    env="local",
    project_path=Path.cwd(),
    params={},
    run={
        "parallel": None,
        "runner": None,
        "is_async": None,
        "node_names": None,
        "to_nodes": None,
        "from_nodes": None,
        "from_inputs": None,
        "to_outputs": None,
        "load_version": None,
        "pipeline": None,
        "tag": None,
    },
):
    # trains would not actaully call the python file with args, have to save this
    run_args_cli = {
        "parallel": None,
        "runner": None,
        "is_async": None,
        "node_names": None,
        "to_nodes": None,
        "from_nodes": None,
        "from_inputs": None,
        "to_outputs": None,
        "load_version": None,
        "pipeline": None,
        "tag": None,
    }
    run_args_cli.update(run)

    param_dict = read_base_params()

    if "hyperopt" in param_dict:
        from hyperopt import STATUS_OK, fmin
        from src.fpl.custom_run.hyperopt_helpers import (
            build_run_config,
            tuple_list_type_check,
            update_parameters,
            find_search_groups,
        )

        def optimize(parameters: Dict):
            parameters = tuple_list_type_check(hyperopt_run_config["space"], parameters)

            trial_num = len(hyperopt_run_config["trials"].tids)
            print(f"Hyperopt trial {trial_num}/{hyperopt_run_config['max_evals']}")
            res = run_kedro_task(
                env=env,
                params=parameters,
                run_args_cli=run_args_cli,
                project_path=project_path,
            )
            return {
                "loss": res[target_name] if strategy == "min" else -res[target_name],
                "status": STATUS_OK,
                target_name: res[target_name],
            }

        hyperopt_param = param_dict.pop("hyperopt")
        hyperopt_run_groups = find_search_groups(hyperopt_param)
        for i, group_param in hyperopt_run_groups:
            base_param = update_parameters(param_dict, group_param)
            hyperopt_run_config, log_info = build_run_config(
                base_param, hyperopt_param[i], hyperopt_param["target"]
            )
            trials = hyperopt_run_config["trials"]
            target_name = log_info["target_name"]
            strategy = log_info["strategy"]
            algo = log_info["algo"]
            uuid_tag = log_info["uuid"]
            uuid_str = uuid.uuid4().hex

            hyperopt_run_config["fn"] = optimize
            fmin(**hyperopt_run_config, show_progressbar=False)

            print(f"Best parameters are: {trials.best_trial['misc']['vals']}")
            print(f"{target_name} = {trials.best_trial['result'][target_name]}")
        return None

    else:
        run_kedro_task(
            env=env,
            params=params,
            run_args_cli=run_args_cli,
            project_path=project_path,
        )
        return None


def run_kedro_task(env, params, run_args_cli, project_path):
    print(
        dedent(
            f"""\
            kedro_trains_run started with args:
            env: {env}
            extra_params: {params}
            run_args: {run_args_cli}"""
        )
    )
    # NOTE: workaround to overwrite partial keys in the parameters, kedro only replace object in first level
    old_context = KedroSession.create(project_path=project_path, env=env)

    with KedroSession.create(
        project_path=project_path,
        env=env,
        extra_params=dict_nested_update(old_context.load_context().params, params),
    ) as context:
        if run_args_cli["parallel"] and run_args_cli["runner"]:
            raise KedroCliError(
                "Both --parallel and --runner options cannot be used together. "
                "Please use either --parallel or --runner."
            )
        runner = run_args_cli["runner"] or "SequentialRunner"
        if run_args_cli["parallel"]:
            runner = "ParallelRunner"
        runner_class = load_obj(runner, "kedro.runner")

        run_args: dict = deepcopy(run_args_cli)
        run_args["runner"] = runner_class(is_async=run_args_cli["is_async"])
        run_args.pop("parallel")
        run_args.pop("is_async")
        run_args["load_versions"] = run_args.pop("load_version")
        run_args["pipeline_name"] = run_args.pop("pipeline")
        run_args["tags"] = run_args.pop("tag")

        res = context.run(**run_args)

        logging.getLogger(__name__).info(
            "Kedro context finishes with result:\n {}".format(res)
        )
        context.close()
        return res
