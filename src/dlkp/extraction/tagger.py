from typing import List, str, Union
import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
    BertForTokenClassification,
)
import numpy as np
from .crf_models import AutoCRFforTokenClassification
from .crf_trainer import CRF_Trainer

from ..datasets.extraction import KpExtractionDatasets


class KeyphraseTagger:
    def __init__(
        self, model_name_or_path
    ) -> None:  # TODO use this class in train and eval purpose as well
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.use_crf = self.config.use_crf if self.config.use_crf is not None else False
        self.id_to_label = {
            0: "B",
            1: "I",
            2: "O",
        }  # TODO take this from config file in future
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            add_prefix_space=True,
        )
        model_type = (
            AutoCRFforTokenClassification
            if self.use_CRF
            else AutoModelForTokenClassification
        )

        self.model = model_type.from_pretrained(
            model_name_or_path,
            config=self.config,
        )
        self.data_collator = DataCollatorForTokenClassification(self.tokenizer)

        self.trainer = (CRF_Trainer if self.use_crf else Trainer)(
            model=self.model, tokenizer=self.tokenizer, data_collator=self.data_collator
        )

    @classmethod
    def load(cls, model_name_or_path):
        return cls(model_name_or_path)

    def predict(self, texts: Union[List, str]):
        if isinstance(texts, str):
            texts = [texts]
        self.datasets = KpExtractionDatasets.load_kp_datasets_from_text(texts)
        # tokenize current datsets
        def tokenize_(txt):
            return KpExtractionDatasets.tokenize_text(txt, self.tokenizer, "max_length")

        self.datasets = self.datasets.map(tokenize_)

        predictions, labels, metrics = self.trainer.predict(self.datasets)
        predictions = np.argmax(predictions, axis=2)

        def extract_kp_from_tags_(examples, idx):
            ids = examples["input_ids"]
            atn_mask = examples["special_tokens_mask"]
            tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
            tags = [
                self.id_to_label[p]
                for (p, m) in zip(predictions[idx], atn_mask)
                if m == 0
            ]
            assert len(tokens) == len(
                tags
            ), "number of tags (={}) in prediction and tokens(={}) are not same for {}th".format(
                len(tags), len(tokens), idx
            )
            token_ids = self.tokenizer.convert_tokens_to_ids(
                tokens
            )  # needed so that we can use batch decode directly and not mess up with convert tokens to string algorithm
            all_kps = KpExtractionDatasets.extract_kp_from_tags(token_ids, tags)

            extracted_kps = self.tokenizer.batch_decode(
                all_kps,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            examples["extracted_keyphrase"] = extracted_kps

            return examples

        self.datasets = self.datasets.map(extract_kp_from_tags_, with_indices=True)

        return self.datasets["extracted_keyphrase"]
