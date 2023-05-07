from fpl.custom_run.hyperopt_helpers.termination_policy.BaseParam import BaseParam


class BanditParam(BaseParam):
    slack_amount: float

    @classmethod
    def init(cls, evaluation_interval: int, delay_evaluation: int, **policy_params):
        super().init(
            evaluation_interval=evaluation_interval, delay_evaluation=delay_evaluation
        )
        cls.slack_amount = policy_params["slack_amount"]


def bandit_policy(result, trial_count: int = 1):
    slack_amount = BanditParam.slack_amount
    evaluation_interval = BanditParam.evaluation_interval
    delay_evaluation = BanditParam.delay_evaluation
    metric_value = result.results[-1]["loss"]
    best_value = result.best_trial["result"]["loss"]

    if (
        trial_count <= delay_evaluation
        or (trial_count - delay_evaluation) % evaluation_interval != 0
    ):
        return False, [trial_count + 1]

    return (metric_value - best_value) > slack_amount, [trial_count + 1]
