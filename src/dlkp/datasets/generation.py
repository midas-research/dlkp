import os, sys, logging
from dataclasses import dataclass, field
from typing import Optional
from datasets import ClassLabel, load_dataset
from . import KpDatasets

logger = logging.getLogger(__name__)


class KpGenerationDatasets(KpDatasets):
    def __init__(self, data_args_, tokenizer_) -> None:
        super().__init__()
        self.data_args = data_args_
        self.tokenizer = tokenizer_
        self.text_column_name = self.data_args.text_column_name
        self.keyphrases_column_name = self.data_args.keyphrases_column_name
        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(
            self.data_args.max_seq_length, self.tokenizer.model_max_length
        )
        self.max_keyphrases_length = self.data_args.max_keyphrases_length
        self.padding = "max_length" if self.data_args.pad_to_max_length else False
        self.datasets = None
        self.truncation = True
        self.kp_sep_token = self.data_args.keyphrase_sep_token
        self.load_kp_datasets()

    def load_kp_datasets(self):
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.datasets = load_dataset(
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
            self.datasets = load_dataset(
                extension,
                data_files=data_files,
                field="data",
            )

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

        self.columns = column_names

        if self.text_column_name is None:
            self.text_column_name = (
                "document" if "document" in column_names else column_names[1]
            )  # either document or 2nd column as text i/p

        assert self.text_column_name in column_names

        if self.keyphrases_column_name is None:
            self.keyphrases_column_name = (
                "keyphrases" if "keyphrases" in column_names else None
            )
            if len(column_names) > 2:
                self.keyphrases_column_name = column_names[2]

        if self.keyphrases_column_name is not None:
            assert self.keyphrases_column_name in column_names

    @staticmethod
    def prepare_text_input(txt):
        return " ".join(txt)

    @staticmethod
    def prepare_one2many_target(keyphrase_list, sep_token):
        sep_token = " " + sep_token + " "
        return sep_token.join(keyphrase_list)

    def preapre_inputs_and_target(self, examples):
        # TODO give option to preapare based on one2one option
        # print("len of ex", len(examples[self.text_column_name]))
        input_text = self.prepare_text_input(examples[self.text_column_name])

        target_text = self.prepare_one2many_target(
            examples[self.keyphrases_column_name], self.kp_sep_token
        )
        # assert len(input_text) == len(target_text)

        # print(input_text)

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=self.truncation,
        )

        with self.tokenizer.as_target_tokenizer():
            targets = self.tokenizer(
                target_text,
                max_length=self.max_keyphrases_length,
                padding=self.padding,
                truncation=self.truncation,
            )

        if self.padding and self.data_args.ignore_pad_token_for_loss:
            targets["input_ids"] = [
                (t if t != self.tokenizer.pad_token_id else -100)
                for t in targets["input_ids"]
            ]

        inputs["labels"] = targets["input_ids"]
        # print("inputs", inputs)
        return inputs

    def get_train_inputs(self):
        if "train" not in self.datasets:
            return None
        if self.data_args.max_train_samples is not None:
            self.datasets["train"] = self.datasets["train"].select(
                range(self.data_args.max_train_samples)
            )
        return self.prepare_split_inputs("train")

    def get_eval_inputs(self):
        if "validation" not in self.datasets:
            return None
        if self.data_args.max_eval_samples is not None:
            self.datasets["validation"] = self.datasets["validation"].select(
                range(self.data_args.max_eval_samples)
            )
        return self.prepare_split_inputs("validation")

    def get_test_inputs(self):
        if "test" not in self.datasets:
            return None
        if self.data_args.max_test_samples is not None:
            self.datasets["test"] = self.datasets["test"].select(
                range(self.data_args.max_test_samples)
            )
        return self.prepare_split_inputs("test")

    def prepare_split_inputs(self, split_name):
        # TODO test remove other columns feature
        self.datasets[split_name] = self.datasets[split_name].map(
            self.preapre_inputs_and_target,
            # batched=True,
            # remove_columns=self.columns,
            num_proc=self.data_args.preprocessing_num_workers,
        )

        return self.datasets[split_name]
