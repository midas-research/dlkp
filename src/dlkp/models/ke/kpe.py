# run_kpe.py
# all long docu,emt modesl realted to KP
#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for token classification.
"""
# You can also adapt this script on your own token classification task and datasets. Pointers for this are left as
# comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

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
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from .transformer.crf_models import (
    BERT_CRFforTokenClassification,
    AutoCRFforTokenClassification,
)
from .transformer.token_classification_models import (
    LongformerForTokenClassification,
)
from .crf.crf_trainer import CRF_Trainer

# from extraction_utils import ModelArguments, DataTrainingArguments
from ...kp_metrics.metrics import compute_metrics
from ...kp_dataset.datasets import KpExtractionDatasets

logger = logging.getLogger(__name__)


CRF_MODEL_DICT = {
    "bert": BERT_CRFforTokenClassification,
    "auto": AutoCRFforTokenClassification,
    # "longformer": Longformer_CRFforTokenClassification,
}
TOKEN_MODEL_DICT = {
    "bert": BertForTokenClassification,
    "auto": AutoModelForTokenClassification
    # "longformer": LongformerForTokenClassification,
    # "reformer": ReformerForTokenClassification,
}

MODEL_DICT = {"crf": CRF_MODEL_DICT, "token": TOKEN_MODEL_DICT}


TRAINER_DICT = {
    "crf": CRF_Trainer,
    "token": Trainer,
}


def run_extraction_model(model_args, data_args, training_args):

    # See all possible arguments in src/transformers/training_args.py

    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)
    # logger.set_global_logging_level(logging.INFO)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    # data
    logging.info("loading kp dataset")
    dataset = KpExtractionDatasets(data_args, tokenizer)
    num_labels = dataset.num_labels
    logging.info("tokeenize and allign laebls")
    dataset.tokenize_and_align_labels()
    train_dataset = dataset.get_train_dataset()
    eval_dataset = dataset.get_eval_dataset()
    test_dataset = dataset.get_test_dataset()

    # config
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    config.use_CRF = model_args.use_CRF

    # model
    model = (
        AutoCRFforTokenClassification
        if model_args.use_CRF
        else AutoModelForTokenClassification
    )
    model = model.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Initialize our Trainer
    trainer = TRAINER_DICT["crf" if model_args.use_CRF else "token"](
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(
                os.path.join(training_args.output_dir, "trainer_state.json")
            )

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate()
        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results_KPE.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in results.items():
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        assert test_dataset is not None, "test data is none"
        predictions, labels, metrics = trainer.predict(test_dataset)
        # if model_args.use_CRF is False:
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        predicted_labels = [
            [dataset.id_to_label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [dataset.id_to_label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        output_test_results_file = os.path.join(
            training_args.output_dir, "test_results.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        output_test_predictions_file = os.path.join(
            training_args.output_dir, "test_predictions.csv"
        )
        output_test_predictions_BIO_file = os.path.join(
            training_args.output_dir, "test_predictions_BIO.txt"
        )
        if trainer.is_world_process_zero():
            predicted_kps = dataset.get_extracted_keyphrases(
                predicted_labels=predicted_labels
            )
            df = pd.DataFrame.from_dict({"extractive_keyphrase": predicted_kps})
            df.to_csv(output_test_predictions_file, index=False)

            # get BIO tag files

            with open(output_test_predictions_BIO_file, "w") as writer:
                for prediction in predicted_labels:
                    writer.write(" ".join(prediction) + "\n")

    return results
