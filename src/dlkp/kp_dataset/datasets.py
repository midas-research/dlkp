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
        self.padding = "max_length" if self.data_args.pad_to_max_length else False
        self.set_labels()
        self.load_kp_datasets()

    def tokenize_and_align_labels(self):
        self.tokenized_datasets = self.datasets.map(
            self.tokenize_and_align_labels_,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            # cache_file_name= data_args.cache_file_name
        )

    def set_labels(self):
        self.label_to_id = {"B": 0, "I": 1, "O": 2}
        self.id_to_label = {0: "B", 1: "I", 2: "O"}
        self.num_labels = 3

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
        if "train" not in self.tokenized_datasets:
            return None
        return self.tokenized_datasets["train"]

    def get_eval_dataset(self):
        if "validation" not in self.tokenized_datasets:
            return None
        return self.tokenized_datasets["validation"]

    def get_test_dataset(self):
        if "test" not in self.tokenized_datasets:
            return None
        return self.tokenized_datasets["test"]

    def tokenize_and_align_labels_(self, examples):
        tokenized_inputs = self.tokenizer(
            examples[self.text_column_name],
            padding=self.padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        if self.label_column_name is None:
            return tokenized_inputs

        for i, label in enumerate(examples[self.label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                    # label_ids.append(2)  # to avoid error change -100 to 'O' tag i.e. 2 class
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(
                        self.label_to_id[label[word_idx]]
                        if self.data_args.label_all_tokens
                        else -100
                    )
                    # to avoid error change -100 to 'O' tag i.e. 2 class
                    # label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else 2)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    def get_keyphrases_from_tags(self, examples):
        pass
