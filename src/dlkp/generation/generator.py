import sys, os
from typing import List, Union
import transformers
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser
from .data_collators import DataCollatorForSeq2SeqKpGneration
from .train_eval_generator import train_and_eval_generation_model
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
        self.datasets = None

    @classmethod
    def load(cls, model_name_or_path):
        return cls(model_name_or_path)

    def generate(
        self,
        texts,
        num_return_sequences=1,
        max_length=50,
        num_beams=3,
        output_seq_score=False,
    ):

        assert (
            num_beams >= num_return_sequences
        ), "num_beams>=num_return_sequences for generation to work"
        if isinstance(texts, str):
            texts = [texts]

        self.datasets = KpGenerationDatasets.load_kp_datasets_from_text(texts)

        model_input = self.tokenizer(
            self.datasets["document"], padding=True, truncation=True
        )

        gen_out = self.trainer.predict(
            model_input,
            num_return_sequences=num_return_sequences,
            max_length=max_length,
            num_beams=num_beams,
            output_scores=output_seq_score,
        )

        generated_seq = self.tokenizer.batch_decode(
            gen_out.predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return generated_seq

    @staticmethod
    def train_and_eval(model_args, data_args, training_args):
        return train_and_eval_generation_model(
            model_args=model_args, data_args=data_args, training_args=training_args
        )

    @staticmethod
    def train_and_eval_cli():
        parser = HfArgumentParser(
            (KGModelArguments, KGDataArguments, KGTrainingArguments)
        )

        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(
                json_file=os.path.abspath(sys.argv[1])
            )
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        return train_and_eval_generation_model(
            model_args=model_args, data_args=data_args, training_args=training_args
        )
