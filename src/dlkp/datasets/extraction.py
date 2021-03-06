import logging
from datasets import load_dataset, Dataset
from . import KPDatasets

logger = logging.getLogger(__name__)


class KEDatasets(KPDatasets):
    def __init__(self, data_args_, tokenizer_) -> None:
        super().__init__()
        self.data_args = data_args_
        self.tokenizer = tokenizer_
        self.text_column_name = (
            self.data_args.text_column_name if self.data_args is not None else None
        )
        self.label_column_name = (
            self.data_args.label_column_name if self.data_args is not None else None
        )
        if self.data_args.max_seq_length is None:
            self.data_args.max_seq_length = self.tokenizer.model_max_length
        if self.data_args.max_seq_length > self.tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({self.data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({self.tokenizer.model_max_length}). Using max_seq_length={self.tokenizer.model_max_length}."
            )
        self.max_seq_length = min(
            self.data_args.max_seq_length, self.tokenizer.model_max_length
        )
        self.padding = "max_length" if self.data_args.pad_to_max_length else False
        self.datasets = None
        self.preprocess_function = self.data_args.preprocess_func
        self.set_labels()
        self.load_kp_datasets()

    def tokenize_and_align_labels(self):
        self.datasets = self.datasets.map(
            self.tokenize_and_align_labels_,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
            # cache_file_name= data_args.cache_file_name
        )

    def set_labels(self):
        self.label_to_id = {"B": 0, "I": 1, "O": 2}
        self.id_to_label = {0: "B", 1: "I", 2: "O"}
        self.num_labels = 3

    @staticmethod
    def load_kp_datasets_from_text(txt):
        return Dataset.from_dict({"document": txt})

    def load_kp_datasets(self):
        if self.data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            self.datasets = load_dataset(
                self.data_args.dataset_name,
                self.data_args.dataset_config_name,
                cache_dir=self.data_args.cache_dir,
            )
        else:
            data_files = {}
            if self.data_args.train_file is not None:
                data_files["train"] = self.data_args.train_file
                extension = self.data_args.train_file.split(".")[-1]
            elif self.data_args.validation_file is not None:
                data_files["validation"] = self.data_args.validation_file
                extension = self.data_args.validation_file.split(".")[-1]
            elif self.data_args.test_file is not None:
                data_files["test"] = self.data_args.test_file
                extension = self.data_args.test_file.split(".")[-1]
            self.datasets = load_dataset(
                extension, data_files=data_files, cache_dir=self.data_args.cache_dir
            )
        if self.preprocess_function:
            self.datasets = self.datasets.map(
                self.preprocess_function,
                num_proc=self.data_args.preprocessing_num_workers,
            )
            print("preprocess done new daasets labels", self.datasets)
        if "train" in self.datasets:
            column_names = self.datasets["train"].column_names
            features = self.datasets["train"].features
        elif "validation" in self.datasets:
            column_names = self.datasets["validation"].column_names
            features = self.datasets["validation"].features
        elif "test" in self.datasets:
            column_names = self.datasets["test"].column_names
            features = self.datasets["test"].features
        else:
            raise AssertionError(
                "neither train, validation nor test dataset is availabel"
            )

        if self.text_column_name is None:
            self.text_column_name = (
                "document" if "document" in column_names else column_names[1]
            )  # either document or 2nd column as text i/p

        assert self.text_column_name in column_names

        if self.label_column_name is None:
            self.label_column_name = (
                "doc_bio_tags" if "doc_bio_tags" in column_names else None
            )
            if len(column_names) > 2:
                self.label_column_name = column_names[2]

        if self.label_column_name is not None:
            assert self.label_column_name in column_names

    def get_train_dataset(self):
        # TODO tokenize and allign data from here
        if "train" not in self.datasets:
            return None
        return self.datasets["train"]

    def get_eval_dataset(self):
        if "validation" not in self.datasets:
            return None
        return self.datasets["validation"]

    def get_test_dataset(self):
        if "test" not in self.datasets:
            return None
        return self.datasets["test"]

    @staticmethod
    def tokenize_text(txt, tokenizer, padding, max_seq_len):
        tokenized_text = tokenizer(
            txt,
            max_length=max_seq_len,
            padding=padding,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
            return_special_tokens_mask=True,
        )
        return tokenized_text

    def tokenize_and_align_labels_(self, examples):
        tokenized_inputs = self.tokenize_text(
            examples[self.text_column_name],
            tokenizer=self.tokenizer,
            padding=self.padding,
            max_seq_len=self.max_seq_length,
        )
        labels = []
        if self.label_column_name is None:
            return tokenized_inputs

        for i, label in enumerate(examples[self.label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)

                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(self.label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on the label_all_tokens flag.
                else:
                    # only IOB2 scheme since decoding is for IOB1 only
                    # TODO (AD) add IOB2 encoding and decoding
                    label_ids.append(
                        (
                            self.label_to_id["I"]
                            if label[word_idx] in ["B", "I"]
                            else self.label_to_id[label[word_idx]]
                        )
                        if self.data_args.label_all_tokens
                        else -100
                    )
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    def get_extracted_keyphrases(
        self, predicted_labels, split_name="test", label_score=None, score_method=None
    ):
        """
        takes predicted labels as input and out put extracted keyphrase.
        threee type of score_method is available 'avg', 'max' and first.
        In 'avg' we take an airthimatic avergae of score of all the tags, in 'max' method maximum score among all the tags and in 'first' score of first tag is used to calculate the confidence score of whole keyphrase
        """
        assert self.datasets[split_name].num_rows == len(
            predicted_labels
        ), "number of rows in original dataset and predicted labels are not same"
        if score_method:
            # assert (
            #     label_score
            # ), "label score is not provided to calculate confidence score"
            assert len(predicted_labels) == len(
                label_score
            ), "len of predicted label is not same as of len of label score"

        self.predicted_labels = predicted_labels
        self.label_score = label_score
        self.score_method = score_method
        self.datasets[split_name] = self.datasets[split_name].map(
            self.get_extracted_keyphrases_,
            num_proc=self.data_args.preprocessing_num_workers,
            with_indices=True,
        )
        self.predicted_labels = None
        self.label_score = None
        self.score_method = None
        if "confidence_score" in self.datasets[split_name].features:
            return (
                self.datasets[split_name]["extracted_keyphrase"],
                self.datasets[split_name]["confidence_score"],
            )
        return self.datasets[split_name]["extracted_keyphrase"], None

    def get_extracted_keyphrases_(self, examples, idx):
        ids = examples["input_ids"]
        special_tok_mask = examples["special_tokens_mask"]
        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        tags = [
            self.id_to_label[p]
            for (p, m) in zip(self.predicted_labels[idx], special_tok_mask)
            if m == 0
        ]
        scores = None
        if self.score_method:
            scores = [
                scr
                for (scr, m) in zip(self.label_score[idx], special_tok_mask)
                if m == 0
            ]
        assert len(tokens) == len(
            tags
        ), "number of tags (={}) in prediction and tokens(={}) are not same for {}th".format(
            len(tags), len(tokens), idx
        )
        token_ids = self.tokenizer.convert_tokens_to_ids(
            tokens
        )  # needed so that we can use batch decode directly and not mess up with convert tokens to string algorithm
        extracted_kps, confidence_scores = self.extract_kp_from_tags(
            token_ids,
            tags,
            tokenizer=self.tokenizer,
            scores=scores,
            score_method=self.score_method,
        )
        examples["extracted_keyphrase"] = extracted_kps
        examples["confidence_score"] = []
        if confidence_scores:
            assert len(extracted_kps) == len(
                confidence_scores
            ), "len of scores and kps are not same"
            examples["confidence_score"] = confidence_scores

        return examples

    def get_original_keyphrases(self, split_name="test"):
        assert (
            "labels" in self.datasets[split_name].features
        ), "truth labels are not present"
        self.datasets[split_name] = self.datasets[split_name].map(
            self.get_original_keyphrases_,
            num_proc=self.data_args.preprocessing_num_workers,
            with_indices=True,
        )
        return self.datasets[split_name]["original_keyphrase"]

    def get_original_keyphrases_(self, examples, idx):
        ids = examples["input_ids"]
        special_tok_mask = examples["special_tokens_mask"]
        labels = examples["labels"]
        tokens = self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
        tags = [
            self.id_to_label[p] for (p, m) in zip(labels, special_tok_mask) if m == 0
        ]
        assert len(tokens) == len(
            tags
        ), "number of tags (={}) in prediction and tokens(={}) are not same for {}th".format(
            len(tags), len(tokens), idx
        )
        token_ids = self.tokenizer.convert_tokens_to_ids(
            tokens
        )  # needed so that we can use batch decode directly and not mess up with convert tokens to string algorithm
        original_kps, _ = self.extract_kp_from_tags(
            token_ids, tags, tokenizer=self.tokenizer
        )

        examples["original_keyphrase"] = original_kps

        return examples

    @staticmethod
    def calculate_confidence_score(scores=None, score_method=None):
        assert scores and score_method
        if score_method == "avg":
            return float(sum(scores) / len(scores))
        elif score_method == "first":
            return scores[0]
        elif score_method == "max":
            return max(scores)

    @staticmethod
    def extract_kp_from_tags(
        token_ids, tags, tokenizer, scores=None, score_method=None
    ):
        if score_method:
            assert len(tags) == len(
                scores
            ), "Score is not none and len of score is not equal to tags"
        all_kps = []
        all_kps_score = []
        current_kp = []
        current_score = []

        for i, (id, tag) in enumerate(zip(token_ids, tags)):
            if tag == "O" and len(current_kp) > 0:  # current kp ends
                if score_method:
                    confidence_score = KEDatasets.calculate_confidence_score(
                        scores=current_score, score_method=score_method
                    )
                    current_score = []
                    all_kps_score.append(confidence_score)

                all_kps.append(current_kp)
                current_kp = []
            elif tag == "B":  # a new kp starts
                if len(current_kp) > 0:
                    if score_method:
                        confidence_score = KEDatasets.calculate_confidence_score(
                            scores=current_score, score_method=score_method
                        )
                        all_kps_score.append(confidence_score)
                    all_kps.append(current_kp)
                current_kp = []
                current_score = []
                current_kp.append(id)
                if score_method:
                    current_score.append(scores[i])
            elif tag == "I":  # it is part of current kp so just append
                current_kp.append(id)
                if score_method:
                    current_score.append(scores[i])
        if len(current_kp) > 0:  # check for the last KP in sequence
            all_kps.append(current_kp)
            if score_method:
                confidence_score = KEDatasets.calculate_confidence_score(
                    scores=current_score, score_method=score_method
                )
                all_kps_score.append(confidence_score)
        all_kps = tokenizer.batch_decode(
            all_kps,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        final_kps, final_score = [], []
        kps_set = {}
        for i, kp in enumerate(all_kps):
            if kp.lower() not in kps_set:
                final_kps.append(kp.lower())
                kps_set[kp.lower()] = -1
                if score_method:
                    kps_set[kp.lower()] = all_kps_score[i]
                    final_score.append(all_kps_score[i])

        if score_method:
            assert len(final_kps) == len(
                final_score
            ), "len of kps and score calculated is not same"
            return final_kps, final_score

        return final_kps, None
