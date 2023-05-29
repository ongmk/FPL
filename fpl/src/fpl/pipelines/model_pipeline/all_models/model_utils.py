from typing import Any


def get_model_instance(
    model_id: str,
    model_params: dict[str, Any],
    available_models: dict[str, Any],
) -> Any:
    if model_id in available_models:
        if model_params is not None:
            return available_models[model_id](**model_params)
        else:
            return available_models[model_id]()
    else:
        raise ValueError(
            f"Invalid model name. Available models: {', '.join(available_models.keys())}"
        )
