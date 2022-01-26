"""
sample huggingface script for dataset library to be used for processing the dataset while downloading
it from huggingface. For example - https://huggingface.co/datasets/midas/kptimes/blob/main/kptimes.py
"""
import json
import datasets

# _SPLIT = ['train', 'test', 'valid']
_CITATION = """\
@inproceedings{gallina2019kptimes,
  title={KPTimes: A Large-Scale Dataset for Keyphrase Generation on News Documents},
  author={Gallina, Ygor and Boudin, Florian and Daille, B{\'e}atrice},
  booktitle={Proceedings of the 12th International Conference on Natural Language Generation},
  pages={130--135},
  year={2019}
}
"""

_DESCRIPTION = """\

"""

_HOMEPAGE = "https://github.com/ygorg/KPTimes"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "Apache License 2.0"

# TODO: Add link to the official dataset URLs here

_URLS = {"test": "test.jsonl", "train": "train.jsonl", "valid": "valid.jsonl"}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class KPTimes(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("0.0.1")

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
    ]

    DEFAULT_CONFIG_NAME = "extraction"

    def _info(self):
        if (
            self.config.name == "extraction"
        ):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "document": datasets.features.Sequence(datasets.Value("string")),
                    "doc_bio_tags": datasets.features.Sequence(
                        datasets.Value("string")
                    ),
                }
            )
        elif self.config.name == "generation":
            features = datasets.Features(
                {
                    "id": datasets.Value("string"),
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
                    "id": datasets.Value("string"),
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
                            "id": datasets.Value("string"),
                            "categories": [datasets.Value("string")],
                            "date": datasets.Value("string"),
                            "title": datasets.Value("string"),
                            "abstract": datasets.Value("string"),
                            "keyword": datasets.Value("string"),
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
                        "id": data.get("paper_id"),
                        "document": data["document"],
                        "doc_bio_tags": data.get("doc_bio_tags"),
                    }
                elif self.config.name == "generation":
                    yield key, {
                        "id": data.get("paper_id"),
                        "document": data["document"],
                        "extractive_keyphrases": data.get("extractive_keyphrases"),
                        "abstractive_keyphrases": data.get("abstractive_keyphrases"),
                    }
                else:
                    yield key, {
                        "id": data.get("paper_id"),
                        "document": data["document"],
                        "doc_bio_tags": data.get("doc_bio_tags"),
                        "extractive_keyphrases": data.get("extractive_keyphrases"),
                        "abstractive_keyphrases": data.get("abstractive_keyphrases"),
                        "other_metadata": data["other_metadata"],
                    }
