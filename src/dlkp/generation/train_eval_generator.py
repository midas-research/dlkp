from curses import raw
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
from ..datasets.generation import KpGenerationDatasets
from .models import AutoSeq2SeqModelForKpGeneration
from .data_collators import DataCollatorForSeq2SeqKpGneration
from .trainers import KpGenerationTrainer

check_min_version("4.17.0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/question-answering/requirements.txt",
)

logger = logging.getLogger(__name__)


def train_eval_tagger(model_args, data_args, training_args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    # parser = HfArgumentParser((KGModelArguments, KGDataArguments, KGTrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(
    #         json_file=os.path.abspath(sys.argv[1])
    #     )
    # else:
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
    raw_datasets = KpGenerationDatasets(data_args, tokenizer)
    train_dataset = raw_datasets.get_train_inputs()
    eval_dataset = raw_datasets.get_eval_dataset()
    test_dataset = raw_datasets.get_test_dataset()

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
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Post-processing:
    # def post_processing_function(
    #     examples: datasets.Dataset,
    #     features: datasets.Dataset,
    #     outputs: EvalLoopOutput,
    #     stage="eval",
    # ):
    #     # Decode the predicted tokens.
    #     preds = outputs.predictions
    #     if isinstance(preds, tuple):
    #         preds = preds[0]
    #     decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    #     # Build a map example to its corresponding features.
    #     example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    #     feature_per_example = {
    #         example_id_to_index[feature["example_id"]]: i
    #         for i, feature in enumerate(features)
    #     }
    #     predictions = {}
    #     # Let's loop over all the examples!
    #     for example_index, example in enumerate(examples):
    #         # This is the index of the feature associated to the current example.
    #         feature_index = feature_per_example[example_index]
    #         predictions[example["id"]] = decoded_preds[feature_index]

    #     # Format the result to the format the metric expects.
    #     if data_args.version_2_with_negative:
    #         formatted_predictions = [
    #             {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
    #             for k, v in predictions.items()
    #         ]
    #     else:
    #         formatted_predictions = [
    #             {"id": k, "prediction_text": v} for k, v in predictions.items()
    #         ]

    #     references = [{"id": ex["id"], "answers": ex[answer_column]} for ex in examples]
    #     return EvalPrediction(predictions=formatted_predictions, label_ids=references)

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
        else data_args.val_max_answer_length
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
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
