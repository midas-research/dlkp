# dlkp
A transformers based deep learning library for keyphrase identification from text documents.

dlkp is:

* **A deep learning keyphrase extraction and generation library.** dlkp allows you to train and apply state-of-the-art 
  deep learning models for keyphrase extraction and generation from text documents.

* **Transformer based framework.** dlkp framework builds directly on [transformers](https://github.com/huggingface/transformers), 
  making it easy to train and evaluate your own transformer based keyphrase extraction and generation models and experiment with 
  new approaches using different contextual embeddings.

* **A dataset library for keyphrase extraction and generation.** dlkp has simple interfaces that allow you 
  to download several benchmark datasets in the domain of keyphrase extraction and generation from 
  [Huggingface Datasets](https://huggingface.co/docs/datasets/index) and readily use them in your training your models
  with the transformer library. It provides easy access to BIO tagged data for several datasets such as Inspec, NUS, 
  WWW, KDD, KP20K, LDKP and many more suitable for training your keyphrase extraction model as a sequence tagger.

* **An evaluation library for keyphrase extraction and generation.** dlkp implements several evaluation metrics for 
  evaluating keyphrase extraction and generation models and helps to generate evaluation reports of your models.
  

## State-of-the-Art Models

dlkp ships with state-of-the-art transformer models for the tasks of keyphrase extraction and generation 

* **Keyphrase Extraction**

| Language | Dataset | Model | Performance (F1) | Paper / Model Card
|  ---  | ----------- | ---------------- | ------------- | ------------- |
| | | | | |
| | | | | |
| | | | | |

* **Keyphrase Generation**

| Language | Dataset | Model | Performance (F1@M) | Paper / Model Card
|  ---  | ----------- | ---------------- | ------------- | ------------- |
| | | | | |
| | | | | |
| | | | | |
  
## Quick Start

### Requirements and Installation

The project is based on transformers>=4.6.0 and Python 3.6+. If you do not have Python 3.6, install it first. 
[Here is how for Ubuntu 16.04](https://vsupalov.com/developing-with-python3-6-on-ubuntu-16-04/).
Then, in your favorite virtual environment, simply do:

```
pip install dlkp
```

### Example Usage

#### Keyphrase Extraction

```python

from dlkp.models import KeyphraseTagger
from dlkp.extraction import (
    KEDataArguments,
    KEModelArguments,
    KETrainingArguments,
)

# sets the training arguments
training_args = KETrainingArguments(
    output_dir="../../outputs",
    learning_rate=4e-5,
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=100,
    logging_steps=100
)

# sets the model arguments
model_args = KEModelArguments(
    model_name_or_path="roberta-base",
    use_crf=False,
)

# sets the data arguments
# downloads the inspec dataset from huggingface hub
data_args = KEDataArguments(
    dataset_name="midas/inspec",
    dataset_config_name="extraction",
    pad_to_max_length=True,
    overwrite_cache=True,
    label_all_tokens=False,
    preprocessing_num_workers=8,
    return_entity_level_metrics=True,
)

# train the model
KeyphraseTagger.train_and_eval(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args
)

# load the model
tagger = KeyphraseTagger.load(
    model_name_or_path="../../outputs"
    )

# sample text
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

# extract keyphrases
keyphrases = tagger.predict(input_text)
print(keyphrases)

# [[' text documents', ' masking strategies', ' models', ' Keyphrase Boundary Infilling with Replacement', 'KBIR', ' KBIR', ' keyphrase extraction', ' BART', ' KeyBART', ' CatSeq', ' named entity recognition', ' question answering', ' relation extraction', ' abstractive summarization', ' NLP']]

```

#### Keyphrase Generation

## Tutorials

* [Loading Datasets](resources/docs/TUTORIAL_LOADING_DATASETS.md)

* [Training Models](resources/docs/TUTORIAL_TRAINING_MODELS.md)

* [Evaluating Models](resources/docs/TUTORIAL_EVALUATING_MODELS.md)

## Citing dlkp

## Contact

Please email your questions or comments to [Amardeep Kumar](https://ad6398.github.io) or [Debanjan Mahata](https://sites.google.com/a/ualr.edu/debanjan-mahata/)

## Contributing

Thanks for your interest in contributing! There are many ways to get involved;
start with our [contributor guidelines](CONTRIBUTING.md) and then
check these [open issues](https://github.com/midas-research/dlkp/issues) for specific tasks.


## [License](/LICENSE)

The MIT License (MIT)
