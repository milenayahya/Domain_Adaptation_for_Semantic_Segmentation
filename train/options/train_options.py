from pathlib import Path
from typing import Literal
from tap import Tap

try:
    from Domain_Adaptation_for_Semantic_Segmentation.Datasets import PROJECT_BASE_PATH
except ImportError:
    from Datasets import PROJECT_BASE_PATH # type: ignore

class TrainOptions(Tap):
    mode: Literal["train", "validate_with_best"] = "train"
    backbone: Literal["CatmodelSmall"] = "CatmodelSmall"
    pretrain_path: Path = Path(PROJECT_BASE_PATH) / "./STDCNet813M_73.91.tar"
    use_conv_last: bool = False
    num_epochs: int = 50
    """Number of epochs to train for """
    resume: bool = True
    """Resume from latest checkpoint"""
    use_best: bool = True
    """Use best model for final validation"""
    checkpoint_step: int = 1
    """How often to save checkpoints (epochs)"""
    validation_step: int = 5
    """How often to perform validation (epochs)"""
    batch_size: int = 4
    """Number of images in each batch"""
    learning_rate: float = 0.01
    """Learning rate """
    num_workers: int = 4
    """Number of workers"""
    num_classes: int = 19
    """Number of object classes (with void)"""
    cuda: str = "0"
    """GPU id used for training"""
    use_gpu: bool = True
    """Use GPU for training"""
    save_model_path: Path = Path(PROJECT_BASE_PATH) / Path("./results")
    """Path to save model checkpoints"""
    optimizer: Literal["rmsprop", "sgd", "adam"] = "adam"
    """Optimizer"""
    loss: Literal["crossentropy"] = "crossentropy"
    """Loss Function"""

    @staticmethod
    def default() -> "TrainOptions":
        return TrainOptions().from_dict({})

def parse_args(*args, **kwargs) -> TrainOptions:
    return TrainOptions().parse_args(*args, **kwargs)


if __name__ == "__main__":
    from pprint import pprint
    args = parse_args()
    pprint(args)
    pprint(args.as_dict())
