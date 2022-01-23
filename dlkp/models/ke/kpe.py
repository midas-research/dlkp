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


logger = logging.getLogger(__name__)
# from models.long_doc_kp_models import LONG_DOC_KP_MODELS

# KPE_MODELS_DICT={
#     'others': AutoModelForTokenClassification,
#     "longformer":
#     'reformer' :
#     'crf_longformer' :
#     'crf_bert': BERT_CRFforTokenClassification
# }

CRF_MODEL_DICT = {
    "bert": BERT_CRFforTokenClassification,
    "longformer": Longformer_CRFforTokenClassification,
}
TOKEN_MODEL_DICT = {
    "bert": BertForTokenClassification,
    "longformer": LongformerForTokenClassification,
    "reformer": ReformerForTokenClassification,
}

MODEL_DICT = {"crf": CRF_MODEL_DICT, "simple": TOKEN_MODEL_DICT}

# KPE_MODELS_DICT = KPE_MODELS_DICT | LONG_DOC_KP_MODELS


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_family_name: str = field(
        metadata={
            "help": "name of the family of model, bert, longformer, reformer etc."
        }
    )
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_CRF: bool = field(
        default=False,
        metadata={"help": "wether to use CRF on top of the classifier"},
    )
    use_BiLSTM: bool = field(
        default=False,
        metadata={"help": "use BiLSTM in sequence classification"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(
        default="simple", metadata={"help": "The name of the task simple, crf"}
    )

    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={
            "help": "Whether to return all the entity levels during evaluation or just the overall ones."
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    cache_file_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Provide the name of a path for the cache file. It is used to store the results of the computation instead of the automatically generated cache file name."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


# def main():
TRAINER_DICT = {
    "crf": CRF_Trainer,
    "simple": Trainer,
}


def main_run_kpe(model_args, data_args, training_args):

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

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
    logger.setLevel(
        logging.INFO if is_main_process(training_args.local_rank) else logging.WARN
    )

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

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    ## get dataset in here
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
        extension = data_args.train_file.split(".")[-1]
        datasets = load_dataset(
            extension, data_files=data_files
        )  ##CR get dataset in here
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["validation"].column_names
        features = datasets["validation"].features
    text_column_name = "text" if "text" in column_names else column_names[0]
    label_column_name = "BIO_tags" if "BIO_tags" in column_names else column_names[1]

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
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
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)
    print(label_to_id)
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
    config.use_BiLSTM = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
    )
    model = MODEL_DICT[data_args.task_name][
        model_args.model_family_name
    ].from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    model.freeze_encoder_layer()
    print("model")
    print(model)
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
                    # label_ids.append(-100)
                    label_ids.append(
                        2
                    )  # to avoid error change -100 to 'O' tag i.e. 2 class
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
        tokenized_inputs["labels"] = labels
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

    from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score

    def compute_metrics(p):
        predictions, labels = p
        # print(predictions.shape, labels.shape)
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

        # results = metric.compute(predictions=true_predictions, references=true_labels)
        results = {}
        results["overall_precision"] = precision_score(true_labels, true_predictions)
        results["overall_recall"] = recall_score(true_labels, true_predictions)
        results["overall_f1"] = f1_score(true_labels, true_predictions)
        results["overall_accuracy"] = accuracy_score(true_labels, true_predictions)
        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

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

    def get_kp_from_BIO(examples, prediction):
        kps = []
        for i in range(len(prediction)):
            ids = examples["input_ids"][i]

            tags = prediction[i]
            # print(tags)
            current_kps = []
            ckp = []
            for i, tag in enumerate(tags):
                id = ids[i]

                if tag == "O" and len(ckp) > 0:

                    current_kps.append(ckp)
                    ckp = []
                elif tag == "B":
                    # print(ckp, tag)
                    if tokenizer.convert_ids_to_tokens(id).startswith("##"):
                        ckp.append(id)
                    else:
                        if len(ckp) > 0:
                            current_kps.append(ckp)
                            ckp = []

                        ckp.append(id)
                        # print(ckp, id)

                elif tag == "I" and len(ckp) > 0:
                    ckp.append(id)
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
            kps.append(decoded_kps)

        # examples['predicted_kp']= kps
        return kps

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

        output_test_results_file = os.path.join(
            training_args.output_dir, "test_results.txt"
        )
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        # Save predictions
        output_test_predictions_file = os.path.join(
            training_args.output_dir, "test_predictions.txt"
        )
        if trainer.is_world_process_zero():
            # # test_dataset['predicted_tags']= true_predictions
            # # test_dataset=test_dataset.map(get_kp_from_BIO,batched=True,
            # num_proc=data_args.preprocessing_num_workers,
            # load_from_cache_file=not data_args.overwrite_cache,
            #  )
            gen_kps = get_kp_from_BIO(test_dataset, true_predictions)
            with open(output_test_predictions_file, "w") as writer:
                for prediction in gen_kps:
                    writer.write(" <kp_sep> ".join(prediction) + "\n")

    return results