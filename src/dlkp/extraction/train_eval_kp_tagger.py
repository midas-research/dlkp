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
    AutoTokenizer,
    set_seed,
    HfArgumentParser,
    RobertaConfig,
    BertConfig,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from .trainer import KpExtractionTrainer, CrfKpExtractionTrainer
from .models import (
    AutoModelForKpExtraction,
    BertCrfModelForKpExtraction,
    RobertaCrfForKpExtraction,
)
from .utils import KEDataArguments, KEModelArguments, KETrainingArguments
from .data_collators import DataCollatorForKpExtraction
from ..metrics.metrics import compute_metrics, compute_kp_level_metrics
from ..datasets.extraction import KEDatasets

logger = logging.getLogger(__name__)


def train_eval_extraction_model(model_args, data_args, training_args):

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

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
    )
    pad_token_none = tokenizer.pad_token == None
    if pad_token_none:
        tokenizer.pad_token = tokenizer.eos_token

    # load keyphrase data
    logging.info("loading kp dataset")
    dataset = KEDatasets(data_args, tokenizer)

    num_labels = dataset.num_labels
    logging.info("tokenize and align laebls")
    dataset.tokenize_and_align_labels()
    # get all data splits, non existance split will be None
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

    config.use_crf = model_args.use_crf
    config.label2id = dataset.label_to_id
    config.id2label = dataset.id_to_label
    if pad_token_none:
        config.pad_token_id = config.eos_token_id

    # model
    model_type = AutoModelForKpExtraction
    if model_args.use_crf:
        if isinstance(config, RobertaConfig):
            model_type = RobertaCrfForKpExtraction
        else:
            model_type = BertCrfModelForKpExtraction

    model = model_type.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Data collator
    data_collator = DataCollatorForKpExtraction(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Initialize our Trainer
    trainer = KpExtractionTrainer(
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
        predictions, labels, metrics = trainer.predict(
            eval_dataset, metric_key_prefix="eval"
        )
        predictions = np.exp(predictions)
        predicted_labels = np.argmax(predictions, axis=2)
        label_score = np.amax(predictions, axis=2) / np.sum(predictions, axis=2)
        output_eval_file = os.path.join(
            training_args.output_dir, "eval_results_KPE.txt"
        )
        if trainer.is_world_process_zero():
            predicted_kps, confidence_scores = dataset.get_extracted_keyphrases(
                predicted_labels=predicted_labels,
                split_name="validation",
                label_score=label_score,
                score_method=training_args.score_aggregation_method,
            )
            original_kps = dataset.get_original_keyphrases(split_name="validation")

            kp_level_metrics = compute_kp_level_metrics(
                predictions=predicted_kps, originals=original_kps, do_stem=True
            )
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

                logger.info("Keyphrase level metrics\n")
                writer.write("Keyphrase level metrics\n")

                for key, value in sorted(kp_level_metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

                total_keyphrases = sum([len(x) for x in confidence_scores])
                total_confidence_scores = sum([sum(x) for x in confidence_scores])
                avg_confidence_scores = total_confidence_scores / total_keyphrases
                total_examples = len(predicted_kps)

                avg_predicted_kps = total_keyphrases / total_examples

                logger.info(
                    "average confidence score: {}\n".format(avg_confidence_scores)
                )
                logger.info(
                    "average number of keyphrases predicted: {}\n".format(
                        avg_predicted_kps
                    )
                )
                writer.write(
                    "average confidence score: {}\n".format(avg_confidence_scores)
                )
                writer.write(
                    "average number of keyphrases predicted: {}\n".format(
                        avg_predicted_kps
                    )
                )

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        assert test_dataset is not None, "test data is none"
        predictions, labels, metrics = trainer.predict(test_dataset)
        predictions = np.exp(predictions)
        predicted_labels = np.argmax(predictions, axis=2)
        label_score = np.amax(predictions, axis=2) / np.sum(predictions, axis=2)

        output_test_results_file = os.path.join(
            training_args.output_dir, "test_results.txt"
        )

        output_test_predictions_file = os.path.join(
            training_args.output_dir, "test_predictions.csv"
        )
        output_test_predictions_BIO_file = os.path.join(
            training_args.output_dir, "test_predictions_BIO.txt"
        )
        if trainer.is_world_process_zero():
            predicted_kps, confidence_scores = dataset.get_extracted_keyphrases(
                predicted_labels=predicted_labels,
                split_name="test",
                label_score=label_score,
                score_method=training_args.score_aggregation_method,
            )
            original_kps = dataset.get_original_keyphrases(split_name="test")

            kp_level_metrics = compute_kp_level_metrics(
                predictions=predicted_kps, originals=original_kps, do_stem=True
            )
            df = pd.DataFrame.from_dict(
                {
                    "extracted_keyphrase": predicted_kps,
                    "original_keyphrases": original_kps,
                    "confidence_scores": confidence_scores,
                }
            )
            df.to_csv(output_test_predictions_file, index=False)

            results["extracted_keyprases"] = predicted_kps
            with open(output_test_results_file, "w") as writer:
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

                logger.info("Keyphrase level metrics\n")
                writer.write("Keyphrase level metrics\n")

                for key, value in sorted(kp_level_metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

                total_keyphrases = sum([len(x) for x in confidence_scores])
                total_confidence_scores = sum([sum(x) for x in confidence_scores])
                avg_confidence_scores = total_confidence_scores / total_keyphrases
                total_examples = len(predicted_kps)

                avg_predicted_kps = total_keyphrases / total_examples

                logger.info(
                    "average confidence score: {}\n".format(avg_confidence_scores)
                )
                logger.info(
                    "average number of keyphrases predicted: {}\n".format(
                        avg_predicted_kps
                    )
                )
                writer.write(
                    "average confidence score: {}\n".format(avg_confidence_scores)
                )
                writer.write(
                    "average number of keyphrases predicted: {}\n".format(
                        avg_predicted_kps
                    )
                )

    return results
