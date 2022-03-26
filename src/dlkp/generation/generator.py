from typing import List, Union
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from .data_collators import DataCollatorForSeq2SeqKpGneration
from .train_eval_generator import train_eval_generator
from ..datasets.generation import KpGenerationDatasets
from .utils import KGDataArguments, KGModelArguments, KGTrainingArguments
from .models import AutoSeq2SeqModelForKpGeneration
from .trainers import KpGenerationTrainer


class KeyphraseGenerator:
    def __init__(self, model_name_or_path) -> None:
        # Config
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=True,
            add_prefix_space=True,
        )
        # Model
        self.model = AutoSeq2SeqModelForKpGeneration.from_pretrained(
            model_name_or_path,
            config=self.config,
        )
        # Data collator
        self.data_collator = DataCollatorForSeq2SeqKpGneration(self.tokenizer)
        # Trainer
        self.trainer = KpGenerationTrainer(
            model=self.model, tokenizer=self.tokenizer, data_collator=self.data_collator
        )

    @classmethod
    def load(cls, model_name_or_path):
        return cls(model_name_or_path)

    def generate(self, texts):
        if isinstance(texts, str):
            texts = [texts]
