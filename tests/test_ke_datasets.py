from transformers import AutoTokenizer
from dlkp.datasets.extraction import KEDatasets
from dlkp.extraction import KEDataArguments

# dataset from huggingface hub compatible with dlkp library
# https://huggingface.co/datasets/midas/inspec
dataset_name = "midas/inspec"

# tokenizer supported by the transformers library
tokenizer_name = "roberta-base"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    use_fast=True,
    add_prefix_space=True,
)

data_args = KEDataArguments(
    dataset_name=dataset_name,
    dataset_config_name="extraction",
    pad_to_max_length=True,
    overwrite_cache=True,
    label_all_tokens=True,
    preprocessing_num_workers=8,
    return_entity_level_metrics=True,
)

pretokenized_input_text = [
    'Impact', 'of', 'aviation', 'highway-in-the-sky', 'displays', 'on', 'pilot', 'situation', 'awareness',
    'Thirty-six', 'pilots', '-LRB-', '31', 'men', ',', '5', 'women', '-RRB-', 'were', 'tested', 'in', 'a',
    'flight', 'simulator', 'on', 'their', 'ability', 'to', 'intercept', 'a', 'pathway', 'depicted', 'on',
    'a', 'highway-in-the-sky', '-LRB-', 'HITS', '-RRB-', 'display', '.', 'While', 'intercepting', 'and',
    'flying', 'the', 'pathway', ',', 'pilots', 'were', 'required', 'to', 'watch', 'for', 'traffic',
    'outside', 'the', 'cockpit', '.', 'Additionally', ',', 'pilots', 'were', 'tested', 'on', 'their',
    'awareness', 'of', 'speed', ',', 'altitude', ',', 'and', 'heading', 'during', 'the', 'flight', '.',
    'Results', 'indicated', 'that', 'the', 'presence', 'of', 'a', 'flight', 'guidance', 'cue',
    'significantly', 'improved', 'flight', 'path', 'awareness', 'while', 'intercepting', 'the', 'pathway',
    ',', 'but', 'significant', 'practice', 'effects', 'suggest', 'that', 'a', 'guidance', 'cue', 'might',
    'be', 'unnecessary', 'if', 'pilots', 'are', 'given', 'proper', 'training', '.', 'The', 'amount',
    'of', 'time', 'spent', 'looking', 'outside', 'the', 'cockpit', 'while', 'using', 'the', 'HITS',
    'display', 'was', 'significantly', 'less', 'than', 'when', 'using', 'conventional', 'aircraft',
    'instruments', '.', 'Additionally', ',', 'awareness', 'of', 'flight', 'information', 'present',
    'on', 'the', 'HITS', 'display', 'was', 'poor', '.', 'Actual', 'or', 'potential', 'applications',
    'of', 'this', 'research', 'include', 'guidance', 'for', 'the', 'development', 'of', 'perspective',
    'flight', 'display', 'standards', 'and', 'as', 'a', 'basis', 'for', 'flight', 'training', 'requirements'
]

input_text = "In this work, we explore how to learn task-specific language models aimed towards learning rich " \
             "representation of keyphrases from text documents. We experiment with different masking strategies for " \
             "pre-training transformer language models (LMs) in discriminative as well as generative settings. In the " \
             "discriminative setting, we introduce a new pre-training objective - Keyphrase Boundary Infilling with " \
             "Replacement (KBIR), showing large gains in performance (upto 9.26 points in F1) over SOTA, when LM " \
             "pre-trained using KBIR is fine-tuned for the task of keyphrase extraction. In the generative setting, we " \
             "introduce a new pre-training setup for BART - KeyBART, that reproduces the keyphrases related to the " \
             "input text in the CatSeq format, instead of the denoised original input. This also led to gains in " \
             "performance (upto 4.33 points in F1@M) over SOTA for keyphrase generation. Additionally, we also " \
             "fine-tune the pre-trained language models on named entity recognition (NER), question answering (QA), " \
             "relation extraction (RE), abstractive summarization and achieve comparable performance with that of the " \
             "SOTA, showing that learning rich representation of keyphrases is indeed beneficial for many other " \
             "fundamental NLP tasks."

# get the dataset object
dataset = KEDatasets(data_args, tokenizer)


def test_get_train_dataset():
    # gets the dataset from huggingface hub
    train_dataset = dataset.get_train_dataset()
    example = [sample for sample in train_dataset][0]
    assert len([sample for sample in train_dataset]) == 1000
    assert list(example.keys()) == ['id', 'document', 'doc_bio_tags']


def test_get_eval_dataset():
    # gets the dataset from huggingface hub
    eval_dataset = dataset.get_eval_dataset()
    example = [sample for sample in eval_dataset][0]
    assert len([sample for sample in eval_dataset]) == 500
    assert list(example.keys()) == ['id', 'document', 'doc_bio_tags']


def test_get_test_dataset():
    # gets the dataset from huggingface hub
    test_dataset = dataset.get_test_dataset()
    example = [sample for sample in test_dataset][0]
    assert len([sample for sample in test_dataset]) == 500
    assert list(example.keys()) == ['id', 'document', 'doc_bio_tags']


def test_tokenize_and_align_labels():
    raise NotImplementedError()


def test_set_labels():
    assert dataset.label_to_id == {
        "B": 0,
        "I": 1,
        "O": 2
    }
    assert dataset.id_to_label == {
        0: "B",
        1: "I",
        2: "O"
    }
    assert dataset.num_labels == 3


def test_load_kp_datasets_from_text():
    data_instance = dataset.load_kp_datasets_from_text(input_text)
    assert list(data_instance.features.keys())[0] == "document"
    assert data_instance.num_rows == 1231


def test_load_kp_datasets():
    assert dataset.datasets is not None


def test_tokenize_text():
    tokenized_text = dataset.tokenize_text(pretokenized_input_text, tokenizer, padding=False)
    assert list(tokenized_text.keys()) == ['input_ids', 'attention_mask', 'special_tokens_mask']


def test_get_extracted_keyphrases():
    raise NotImplementedError()


def extract_kp_from_tags():
    raise NotImplementedError()
