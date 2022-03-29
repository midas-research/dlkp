class KpDatasets:
    def __init__(self) -> None:
        pass

    def get_train_dataset(self):
        if "train" not in self.datasets:
            return None
        return self.datasets["train"]

    def get_eval_dataset(self):
        if "validation" not in self.datasets:
            return None
        return self.datasets["validation"]

    def get_test_dataset(self):
        if "test" not in self.datasets:
            return None
        return self.datasets["test"]
