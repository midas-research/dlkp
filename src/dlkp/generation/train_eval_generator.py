import logging
import os
import sys
from dataclasses import dataclass, field
import datasets
from datasets import load_dataset, load_metric

import transformers

from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
)
from transformers.trainer_utils import (
    EvalPrediction,
    get_last_checkpoint,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from .utils import KGDataArguments, KGModelArguments, KGTrainingArguments
from ..datasets.generation import KGDatasets
from .models import AutoSeq2SeqModelForKpGeneration
from .data_collators import DataCollatorForSeq2SeqKpGneration
from .trainers import KpGenerationTrainer
from ..metrics.metrics import compute_kp_level_metrics

check_min_version("4.17.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/question-answering/requirements.txt",
)

logger = logging.getLogger(__name__)


def train_and_eval_generation_model(model_args, data_args, training_args):

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

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
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # tokenizer
    # TODO add special token kp_sep
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.add_tokens(data_args.keyphrase_sep_token)
    # datasets
    raw_datasets = KGDatasets(data_args, tokenizer)
    train_dataset = raw_datasets.get_train_inputs()
    eval_dataset = raw_datasets.get_eval_inputs()
    test_dataset = raw_datasets.get_test_inputs()

    # Model
    model = AutoSeq2SeqModelForKpGeneration.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined"
        )

    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = DataCollatorForSeq2SeqKpGneration(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metrics
    metric = load_metric("sacrebleu")

    def compute_metrics(p: EvalPrediction):
        predictions = tokenizer.batch_decode(
            p.predictions,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        labels = [
            [x for x in label if x != label_pad_token_id] for label in p.label_ids
        ]
        originals = tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        # get keyphrases list from string
        if data_args.task_type == "one2many":
            predictions = [
                pred.split(data_args.keyphrase_sep_token) for pred in predictions
            ]
            originals = [
                orig.split(data_args.keyphrase_sep_token) for orig in originals
            ]

        return compute_kp_level_metrics(predictions, originals)

    # Initialize our Trainer
    trainer = KpGenerationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        # post_process_function=post_processing_function,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_keyphrases_length
    )
    num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval"
        )
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        results = trainer.predict(test_dataset)
        metrics = results.metrics
        pre = results.predictions
        decod = tokenizer.batch_decode(
            pre,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        trainer.save_metrics("predict", metrics)
