"""
sample huggingface script for dataset library to be used for processing the dataset while downloading
it from huggingface. This script is used for keyphrase extraction and generation from long documents.
"""

import json

import datasets

# _SPLIT = ['train', 'test', 'valid']
_CITATION = """\
"""

_DESCRIPTION = """\

"""

_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here

_URLS = {"test": "test.jsonl", "train": "train.jsonl", "valid": "valid.jsonl"}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class TestLDKP(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.1")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="extraction",
            version=VERSION,
            description="This part of my dataset covers extraction",
        ),
        datasets.BuilderConfig(
            name="generation",
            version=VERSION,
            description="This part of my dataset covers generation",
        ),
        datasets.BuilderConfig(
            name="raw",
            version=VERSION,
            description="This part of my dataset covers the raw dataset",
        ),
        datasets.BuilderConfig(
            name="ldkp_generation",
            version=VERSION,
            description="This part of my dataset covers abstract only",
        ),
        datasets.BuilderConfig(
            name="ldkp_extraction",
            version=VERSION,
            description="This part of my dataset covers abstract only",
        ),
    ]

    DEFAULT_CONFIG_NAME = "extraction"

    def _info(self):
        if (
            self.config.name == "extraction" or self.config.name == "ldkp_extraction"
        ):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "document": datasets.features.Sequence(datasets.Value("string")),
                    "doc_bio_tags": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                }
            )
        elif self.config.name == "generation" or self.config.name == "ldkp_generation":
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "document": datasets.features.Sequence(datasets.Value("string")),
                    "extractive_keyphrases": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "abstractive_keyphrases": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                }
            )
        else:
            features = datasets.Features(
                {
                    "id": datasets.Value("int64"),
                    "document": datasets.features.Sequence(datasets.Value("string")),
                    "doc_bio_tags": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "extractive_keyphrases": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "abstractive_keyphrases": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                    "other_metadata": datasets.features.Sequence(
                        {
                            "text": datasets.features.Sequence(
                                datasets.Value("string")
                            ),
                            "bio_tags": datasets.features.Sequence(
                                datasets.Value("string")
                            ),
                        }
                    ),
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        data_dir = dl_manager.download_and_extract(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={"filepath": data_dir["test"], "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir["valid"],
                    "split": "valid",
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, split):
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                if self.config.name == "extraction":
                    # Yields examples as (key, example) tuples
                    yield key, {
                        "id": data["paper_id"],
                        "document": data["document"],
                        "doc_bio_tags": data["doc_bio_tags"],
                    }
                elif self.config.name == "ldkp_extraction":
                    yield key, {
                        "id": data["paper_id"],
                        "document": data["document"] + data["other_metadata"]["text"],
                        "doc_bio_tags": data["document_tags"]
                        + data["other_metadata"]["bio_tags"],
                    }
                elif self.config.name == "ldkp_generation":
                    yield key, {
                        "id": data["paper_id"],
                        "document": data["document"] + data["other_metadata"]["text"],
                        "extractive_keyphrases": data["extractive_keyphrases"],
                        "abstractive_keyphrases": data["abstractive_keyphrases"],
                    }
                elif self.config.name == "generation":
                    yield key, {
                        "id": data["paper_id"],
                        "document": data["document"],
                        "extractive_keyphrases": data["extractive_keyphrases"],
                        "abstractive_keyphrases": data["abstractive_keyphrases"],
                    }
                else:
                    yield key, {
                        "id": data["paper_id"],
                        "document": data["document"],
                        "doc_bio_tags": data["doc_bio_tags"],
                        "extractive_keyphrases": data["extractive_keyphrases"],
                        "abstractive_keyphrases": data["abstractive_keyphrases"],
                        "other_metadata": data["other_metadata"],
                    }
