import os, sys
from dataclasses import dataclass, field
from tkinter.messagebox import NO
from typing import Optional
from datasets import ClassLabel, load_dataset


class KPDatasets:
    def __init__(self) -> None:
        pass


class KpExtractionDatasets(KPDatasets):
    def __init__(self, data_args_, tokenizer_) -> None:
        super().__init__()
        self.data_args = data_args_
        self.tokenizer = tokenizer_
        self.text_column_name = self.data_args.text_column_name
        self.label_column_name = self.data_args.label_column_name
        self.datasets = None
        self.label_to_id = {"B": 0, "I": 1, "O": 2}
        self.id_to_label = {0: "B", 1: "I", 2: "O"}

    def load_kp_datasets(self):
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.datasets = load_dataset(
                self.data_args.dataset_name, self.data_args.dataset_config_name
            )
        else:
            data_files = {}
            if self.data_args.train_file is not None:
                data_files["train"] = self.data_args.train_file
                extension = self.data_args.train_file.split(".")[-1]
            elif self.data_args.validation_file is not None:
                data_files["validation"] = self.data_args.validation_file
                extension = self.data_args.validation_file.split(".")[-1]
            elif self.data_args.test_file is not None:
                data_files["test"] = self.data_args.test_file
                extension = self.data_args.test_file.split(".")[-1]
            self.datasets = load_dataset(extension, data_files=data_files)
        if "train" in self.datasets:
            column_names = self.datasets["train"].column_names
            features = self.datasets["train"].features
        elif "validation" in self.datasets:
            column_names = self.datasets["validation"].column_names
            features = self.datasets["validation"].features
        elif "test" in self.datasets:
            column_names = self.datasets["test"].column_names
            features = self.datasets["test"].features
        else:
            raise AssertionError(
                "neither train, validation nor test dataset is availabel"
            )

        if self.text_column_name is None:
            self.text_column_name = (
                "document" if "document" in column_names else column_names[1]
            )  # either document or 2nd column as text i/p

        assert self.text_column_name in column_names

        if self.label_column_name is None:
            self.label_column_name = (
                "doc_bio_tags" if "doc_bio_tags" in column_names else None
            )
            if len(column_names) > 2:
                self.label_column_name = column_names[2]

        if self.label_column_name is not None:
            assert self.label_column_name in column_names

    def get_train_dataset(self):
        pass

    def get_valid_dataset(self):
        pass

    def get_test_dataset(self):
        pass

    def tokenize_and_align_labels(self, examples):
        pass

    def get_keyphrases_from_tags(self, examples):
        pass
