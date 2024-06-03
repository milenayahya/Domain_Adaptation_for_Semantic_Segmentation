from .train_options import TrainOptions


class TrainFDAOptions(TrainOptions):
    """Training Options for FDA task"""

    fda_beta: float = 0.006
    """FDA's Beta value, used to determine how mich of the amplitude to 'transfer' from target to source image"""

    eta: float = 2
    """FDA's hyperparameter - used for robust entropy minimization
    
    It penalizes high entropy predictions more than the low entropy ones for η > 0.5 as shown in Fig. 3.
    @see [Paper 7]: "FDA: Fourier Domain Adaptation for Semantic Segmentation"
    """

    ent_loss_scaling: float = 1.6e-6
    """Entropy Loss Scaling hyperparameter"""

    switch_to_entropy_after_epoch: int = 20
    """Switch to entropy loss after this many epochs"""

    use_sst: bool = False
    """Whether to use SST loss or not"""


def parse_args(*args, **kwargs) -> TrainFDAOptions:
    return TrainFDAOptions().parse_args(*args, **kwargs)


if __name__ == "__main__":
    from pprint import pprint

    args = parse_args()
    pprint(args)
    pprint(args.as_dict())
