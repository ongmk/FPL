import inspect
from typing import Any


def has_named_parameter(cls, parameter_name):
    signature = inspect.signature(cls.__init__)
    return parameter_name in signature.parameters


def get_model_instance(
    model_id: str,
    available_models: dict[str, Any],
    model_params: dict[str, Any],
    n_jobs: int = None,
    random_state: int = None,
    verbose: bool = None,
) -> Any:
    if model_id not in available_models:
        raise ValueError(
            f"Invalid model name. Available models: {', '.join(available_models.keys())}"
        )
    if model_params is None:
        model_params = {}
    model = available_models[model_id]
    if has_named_parameter(model, "n_jobs"):
        model_params["n_jobs"] = n_jobs
    if has_named_parameter(model, "random_state"):
        model_params["random_state"] = random_state
    if has_named_parameter(model, "verbose"):
        model_params["verbose"] = verbose
    return model(**model_params)


def get_model_default_parameters(
    model_id: str, available_models: dict[str, Any]
) -> dict:
    model = get_model_instance(model_id, available_models, {})
    return model.get_params()
