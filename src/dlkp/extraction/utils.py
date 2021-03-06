from dataclasses import dataclass, field
from typing import Optional, Callable
from transformers import TrainingArguments


@dataclass
class KETrainingArguments(TrainingArguments):
    return_keyphrase_level_metrics: bool = field(
        default=True,
        metadata={
            "help": "Whether to return keyphrase level metrics during evaluation or just the BIO tag level."
        },
    )

    score_aggregation_method: bool = field(
        default="avg",
        metadata={
            "help": "which method among avg, max and first to use while calculating confidence score of a keyphrase. None indicates not to calculate this s ore"
        },
    )


@dataclass
class KEModelArguments:
    """
    Arguments for model/config/tokenizer used for fine-tuning a keyphrase extraction model.
    """

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
            "help": "Cache directory for storing the pretrained models downloaded from huggingface.co"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_crf: bool = field(
        default=False,
        metadata={"help": "whether to use CRF head"},
    )


@dataclass
class KEDataArguments:
    """
    Arguments for training and evaluation data
    """

    preprocess_func: Optional[Callable] = field(
        default=None,
        metadata={
            "help": "a function to preprocess the dataset, which take a dataset object as input and return two columns text_column_name and label_column_name"
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default="extraction",
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
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

    text_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict on (a csv or JSON file)."
        },
    )
    train_data_percent: Optional[int] = field(
        default=100,
        metadata={"help": "percentage of training data to be used for training"},
    )
    valid_data_percent: Optional[int] = field(
        default=0,
        metadata={
            "help": "percentage of training data to be used for validation purpose"
        },
    )
    test_data_percent: Optional[int] = field(
        default=0,
        metadata={"help": "percentage of training data to be used for testing purpose"},
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    label_all_tokens: bool = field(
        default=True,
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

    cache_file_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Provide the name of a path for the cache file. It is used to store the results of the computation instead of the automatically generated cache file name."
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Provide the name of a path for the cache dir. It is used to store the results of the computation."
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
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
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`test_file` should be a csv or a json file."

        assert (
            self.train_data_percent + self.test_data_percent + self.valid_data_percent
            == 100
        )
