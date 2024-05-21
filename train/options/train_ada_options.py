from typing import Self
from train_options import TrainOptions


class TrainADAOptions(TrainOptions):
    learning_rate_D: float = 0.0001
    """ learning rate used for train """

    lambda_d1: float = 0.001
    """lambda for adversarial loss"""

    lambda_d2: float = 0.0002
    """lambda for adversarial loss"""

    lambda_d3: float = 0.0002
    """lambda for adversarial loss"""


def parse_args(*args, **kwargs) -> TrainADAOptions:
    return TrainADAOptions().parse_args(*args, **kwargs)


if __name__ == "__main__":
    from pprint import pprint
    args = parse_args()
    pprint(args)
    pprint(args.as_dict())
