import os, sys, logging
from dataclasses import dataclass, field
from typing import Optional
from datasets import ClassLabel, load_dataset, Dataset
from . import KPDatasets

logger = logging.getLogger(__name__)


class KGDatasets(KPDatasets):
    def __init__(self, data_args_, tokenizer_) -> None:
        super().__init__()
        self.data_args = data_args_
        self.tokenizer = tokenizer_
        self.text_column_name = self.data_args.text_column_name
        self.keyphrases_column_name = self.data_args.keyphrases_column_name
        if self.data_args.max_seq_length is None:
            self.data_args.max_seq_length = self.tokenizer.model_max_length
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
        self.preprocess_function = self.data_args.preprocess_func
        self.kp_sep_token = self.data_args.keyphrase_sep_token
        self.load_kp_datasets()

    @staticmethod
    def load_kp_datasets_from_text(txt):
        return Dataset.from_dict({"document": txt})

    def load_kp_datasets(self):
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.data_args.cache_dir,
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
                extension, data_files=data_files, cache_dir=self.data_args.cache_dir
            )

        if self.preprocess_function:
            self.datasets = self.datasets.map(
                self.preprocess_function,
                num_proc=self.data_args.preprocessing_num_workers,
            )
            print("preprocess done. New datasets labels", self.datasets)

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

    def pre_process_keyphrases(self, text_ids, kp_list):
        kp_order_list = []
        kp_set = set(kp_list)
        text = self.tokenizer.decode(
            text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        text = text.lower()
        for kp in kp_set:
            kp = kp.strip()
            kp_index = text.find(kp.lower())
            kp_order_list.append((kp_index, kp))

        if self.data_args.cat_sequence:
            kp_order_list.sort()
        present_kp, absent_kp = [], []

        for kp_index, kp in kp_order_list:
            if kp_index < 0:
                absent_kp.append(kp)
            else:
                present_kp.append(kp)
        return present_kp, absent_kp

    def preapre_inputs_and_target(self, examples):
        # TODO give option to preapare based on one2one option

        input_text = self.prepare_text_input(examples[self.text_column_name])
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_seq_length,
            padding=self.padding,
            truncation=self.truncation,
        )

        if self.data_args.cat_sequence or self.data_args.present_keyphrase_only:
            # get present and absent kps, present will be ordered if cat_sequence = True
            present_kp, absent_kp = self.pre_process_keyphrases(
                text_ids=inputs["input_ids"],
                kp_list=examples[self.keyphrases_column_name],
                # TODO (AD) need to do the hack here for considering absent kps for temp training
            )

            keyphrases = present_kp
            if self.data_args.cat_sequence:
                keyphrases += absent_kp

        else:
            keyphrases = examples[self.keyphrases_column_name]

        target_text = self.prepare_one2many_target(keyphrases, self.kp_sep_token)
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
