from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class CustomTrainer(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, **kwargs)
        self.num_epochs = 20