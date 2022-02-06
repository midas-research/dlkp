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
from datasets import ClassLabel, load_dataset, load_metric

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

from dlkp.models.ke.transformer.crf_models import (
    BERT_CRFforTokenClassification,
    AutoCRFforTokenClassification,
)
from dlkp.models.ke.transformer.token_classification_models import (
    LongformerForTokenClassification,
)
from dlkp.models.ke.crf.crf_trainer import CRF_Trainer

# from extraction_utils import ModelArguments, DataTrainingArguments
from dlkp.kp_metrics.metrics import compute_metrics

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


def run_kpe(model_args, data_args, training_args):

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

    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        datasets = load_dataset(extension, data_files=data_files)
    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = (
        "document" if "document" in column_names else column_names[1]
    )  # either document or 2nd column as text i/p
    label_column_name = (
        "doc_bio_tags" if "doc_bio_tags" in column_names else column_names[2]
    )  # either doc_bio_tags column should be available or 3 rd columns will be considered as tag

    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        # No need to convert the labels since they are already ints.
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(
            datasets["train"][label_column_name]
            if training_args.do_train
            else datasets["validation"][label_column_name]
        )
        label_to_id = {l: i for i, l in enumerate(label_list)}
    label_to_id = {"B": 0, "I": 1, "O": 2}
    num_labels = len(label_list)
    print("label to id", label_to_id)
    id2tag = {}
    for k in label_to_id.keys():
        id2tag[label_to_id[k]] = k
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    config.use_CRF = model_args.use_CRF  ##CR replace from arguments
    config.use_BiLSTM = model_args.use_BiLSTM
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
    )
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
    # model.freeze_encoder_layer()
    print("model")
    # print(model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        config.pad_token_id = config.eos_token_id

    # Tokenizer check: this script requires a fast tokenizer.
    # if not isinstance(tokenizer, PreTrainedTokenizerFast):
    #     raise ValueError(
    #         "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
    #         "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
    #         "requirement"
    #     )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )
        labels = []
        for i, label in enumerate(examples[label_column_name]):
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
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(
                        label_to_id[label[word_idx]]
                        if data_args.label_all_tokens
                        else -100
                    )
                    # to avoid error change -100 to 'O' tag i.e. 2 class
                    # label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else 2)
                previous_word_idx = word_idx

            labels.append(label_ids)
        if data_args.task_name == "guided":
            tokenized_inputs["guide_embed"] = examples["guide_embed"]
        tokenized_inputs["labels"] = labels
        # tokenized_inputs['paper_id']= examples['paper_id']
        # tokenized_inputs['extractive_keyphrases']= examples['extractive_keyphrases']

        return tokenized_inputs

    tokenized_datasets = datasets.map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        # cache_file_name= data_args.cache_file_name
    )

    # Data collator
    data_collator = DataCollatorForTokenClassification(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Initialize our Trainer

    trainer = TRAINER_DICT[data_args.task_name](
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"] if training_args.do_train else None,
        eval_dataset=tokenized_datasets["validation"]
        if training_args.do_eval
        else None,
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

        test_dataset = tokenized_datasets["test"]
        predictions, labels, metrics = trainer.predict(test_dataset)
        # if model_args.use_CRF is False:
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
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

        # Save predictions
        def get_kp_from_BIO(examples, i):
            # kps= []
            # for i in range(len(prediction)):
            ids = examples["input_ids"]
            # print(examples.keys())

            # print(tags)
            def mmkp(tag_):
                current_kps = []
                ckp = []
                prev_tag = None
                for j, tag in enumerate(tag_):
                    id = ids[j]

                    if tag == "O" and len(ckp) > 0:

                        current_kps.append(ckp)
                        ckp = []
                    elif tag == "B":
                        # print(ckp, tag)
                        if (
                            tokenizer.convert_ids_to_tokens(id).startswith("##")
                            or prev_tag == "B"
                        ):
                            ckp.append(id)
                        else:
                            if len(ckp) > 0:
                                current_kps.append(ckp)
                                ckp = []

                            ckp.append(id)
                            # print(ckp, id)

                    elif tag == "I" and len(ckp) > 0:
                        ckp.append(id)
                    prev_tag = tag
                decoded_kps = []
                if len(ckp) > 0:
                    current_kps.append(ckp)
                if len(current_kps) > 0:
                    decoded_kps = tokenizer.batch_decode(
                        current_kps,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    # print(decoded_kps)
                return decoded_kps

            tags = true_predictions[i]
            decoded_kps = mmkp(tags)

            ttgs = true_labels[i]
            eekp = mmkp(ttgs)

            # examples['kp_predicted']= decoded_kps
            examples["kp_predicted"] = list(dict.fromkeys(decoded_kps))
            examples["eekp"] = list(dict.fromkeys(eekp))
            # examples['eekp']= eekp
            # else:
            #     examples['kp_predicted']= ['<dummy_kp>']
            examples["id"] = i
            return examples

        import pandas as pd

        output_test_predictions_file = os.path.join(
            training_args.output_dir, "test_predictions.csv"
        )
        output_test_predictions_BIO_file = os.path.join(
            training_args.output_dir, "test_predictions_BIO.txt"
        )
        if trainer.is_world_process_zero():
            print(test_dataset, len(test_dataset["paper_id"]))
            ppid = test_dataset["paper_id"]
            # ekp= test_dataset['extractive_keyphrases']

            test_dataset = test_dataset.map(
                get_kp_from_BIO,
                num_proc=data_args.preprocessing_num_workers,
                with_indices=True,
            )
            #  input_columns= ['paper_id','input_ids','extractive_keyphrases']
            print(test_dataset, " agian")
            df = pd.DataFrame.from_dict(
                {
                    "id": ppid,
                    "extractive_keyphrase": test_dataset["eekp"],
                    "keyphrases": test_dataset["kp_predicted"],
                }
            )
            df.to_csv(output_test_predictions_file, index=False)

            # get BIO tag files

            with open(output_test_predictions_BIO_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")

    return results
