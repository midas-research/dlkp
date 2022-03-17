from typing import List, Union
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
import os
import sys
import numpy as np

from ..datasets.extraction import KEDatasets
from .utils import KEDataArguments, KEModelArguments, KETrainingArguments
from .trainer import KpExtractionTrainer, CrfKpExtractionTrainer
from .models import AutoModelForKpExtraction, AutoCrfModelforKpExtraction
from .data_collators import DataCollatorForKpExtraction
from .train_eval_kp_tagger import train_eval_extraction_model


class KeyphraseTagger:
    def __init__(
        self, model_name_or_path
    ) -> None:  # TODO use this class in train and eval purpose as well
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.use_crf = self.config.use_crf if self.config.use_crf is not None else False
        self.id_to_label = self.config.id_to_label
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            add_prefix_space=True,
        )
        model_type = (
            AutoCrfModelforKpExtraction if self.use_crf else AutoModelForKpExtraction
        )

        self.model = model_type.from_pretrained(
            model_name_or_path,
            config=self.config,
        )
        self.data_collator = DataCollatorForKpExtraction(self.tokenizer)

        self.trainer = (
            CrfKpExtractionTrainer if self.use_crf else KpExtractionTrainer
        )(model=self.model, tokenizer=self.tokenizer, data_collator=self.data_collator)

    @classmethod
    def load(cls, model_name_or_path):
        return cls(model_name_or_path)

    def predict(self, texts: Union[List, str]):
        if isinstance(texts, str):
            texts = [texts]
        self.datasets = KEDatasets.load_kp_datasets_from_text(texts)
        # tokenize current datsets
        def tokenize_(txt):
            return KEDatasets.tokenize_text(
                txt["document"].split(), self.tokenizer, "max_length"
            )

        self.datasets = self.datasets.map(tokenize_)

        predictions, labels, metrics = self.trainer.predict(self.datasets)
        predictions = np.argmax(predictions, axis=2)

        def extract_kp_from_tags_(examples, idx):
            ids = examples["input_ids"]
            special_tok_mask = examples["special_tokens_mask"]
            tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
            tags = [
                self.id_to_label[str(p)]
                for (p, m) in zip(predictions[idx], special_tok_mask)
                if m == 0
            ]  # TODO remove str(p)
            assert len(tokens) == len(
                tags
            ), "number of tags (={}) in prediction and tokens(={}) are not same for {}th".format(
                len(tags), len(tokens), idx
            )
            token_ids = self.tokenizer.convert_tokens_to_ids(
                tokens
            )  # needed so that we can use batch decode directly and not mess up with convert tokens to string algorithm
            all_kps = KEDatasets.extract_kp_from_tags(token_ids, tags)

            extracted_kps = self.tokenizer.batch_decode(
                all_kps,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            examples["extracted_keyphrase"] = extracted_kps

            return examples

        self.datasets = self.datasets.map(extract_kp_from_tags_, with_indices=True)

        return self.datasets["extracted_keyphrase"]

    @staticmethod
    def train_and_eval(model_args, data_args, training_args):
        return train_eval_extraction_model(
            model_args=model_args, data_args=data_args, training_args=training_args
        )

    @staticmethod
    def train_and_eval_cli():
        parser = HfArgumentParser(
            (KEModelArguments, KEDataArguments, KETrainingArguments)
        )

        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        return train_eval_extraction_model(
            model_args=model_args, data_args=data_args, training_args=training_args
        )
