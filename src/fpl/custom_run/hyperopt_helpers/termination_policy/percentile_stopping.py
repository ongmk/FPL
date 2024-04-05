import logging

import numpy as np

from fpl.custom_run.hyperopt_helpers.termination_policy.BaseParam import BaseParam

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class PercentileStoppingParam(BaseParam):
    percentile: float

    @classmethod
    def init(cls, evaluation_interval: int, delay_evaluation: int, **policy_params):
        super().init(
            evaluation_interval=evaluation_interval, delay_evaluation=delay_evaluation
        )
        cls.percentile = policy_params["percentile"]


def percentile_stopping_policy(result, trial_count: int = 1):
    evaluation_interval = PercentileStoppingParam.evaluation_interval
    delay_evaluation = PercentileStoppingParam.delay_evaluation
    percentile = PercentileStoppingParam.percentile

    if (
        trial_count <= delay_evaluation
        or (trial_count - delay_evaluation) % evaluation_interval != 0
    ):
        return False, [trial_count + 1]

    metric_value = result.results[-1]["loss"]
    metric_history = [hist_result["loss"] for hist_result in result.results]

    threshold = np.percentile(metric_history, 100 - percentile)

    logger.info(f"Latest metric: {metric_value}")
    logger.info(f"{percentile}% metric: {threshold}")

    return metric_value > threshold, [trial_count + 1]
