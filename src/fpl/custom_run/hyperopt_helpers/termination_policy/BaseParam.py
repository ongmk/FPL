class BaseParam:
    evaluation_interval: int
    delay_evaluation: int

    @classmethod
    def init(cls, evaluation_interval: int, delay_evaluation: int, **policy_params):
        cls.evaluation_interval = evaluation_interval
        cls.delay_evaluation = delay_evaluation
