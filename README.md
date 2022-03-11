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

## Tutorials

## Citing dlkp

## Contact

Please email your questions or comments to [Amardeep Kumar](https://ad6398.github.io) or [Debanjan Mahata](https://sites.google.com/a/ualr.edu/debanjan-mahata/)

## Contributing

Thanks for your interest in contributing! There are many ways to get involved;
start with our [contributor guidelines](CONTRIBUTING.md) and then
check these [open issues](https://github.com/midas-research/dlkp/issues) for specific tasks.


## [License](/LICENSE)

The MIT License (MIT)
