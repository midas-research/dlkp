import os, sys
from dataclasses import dataclass, field
from typing import Optional
from datasets import ClassLabel, load_dataset
from . import KpDatasets


class KpGenerationDatasets(KpDatasets):
    def __init__(self, data_args_, tokenizer_) -> None:
        super().__init__()
        self.data_args = data_args_
        self.tokenizer = tokenizer_
        self.text_column_name = self.data_args.text_column_name
        self.keyphrases_column_name = self.data_args.keyphrases_column_name
        self.max_seq_length = self.data_args.max_seq_length
        self.max_keyphrases_length = self.data_args.max_keyphrases_length
        self.padding = "max_length" if self.data_args.pad_to_max_length else False
        self.datasets = None

    def load_kp_datasets(self):
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
            )
        else:
            data_files = {}
            if self.data_args.train_file is not None:
                data_files["train"] = self.data_args.train_file
                extension = self.data_args.train_file.split(".")[-1]

            if self.data_args.validation_file is not None:
                data_files["validation"] = self.data_args.validation_file
                extension = self.data_args.validation_file.split(".")[-1]
            if self.data_args.test_file is not None:
                data_files["test"] = self.data_args.test_file
                extension = self.data_args.test_file.split(".")[-1]
            raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                field="data",
            )
