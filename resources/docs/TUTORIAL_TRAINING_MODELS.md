# Training Models

## Keyphrase Extraction

* **Step 1** - Import the required modules

```python
from dlkp.models import KeyphraseTagger
from dlkp.extraction import (
    KEDataArguments,
    KEModelArguments,
    KETrainingArguments,
)
```

* **Step 2** - Initialize the data arguments.

```python
data_args = KEDataArguments(
    dataset_name="midas/inspec",
    dataset_config_name="extraction",
    pad_to_max_length=True,
    overwrite_cache=True,
    label_all_tokens=False,
    preprocessing_num_workers=8,
    return_entity_level_metrics=True,
)
```

This will automatically download the **inspec** corpus from the huggingface hub and prepare it for training a keyphrase 
extraction model by treating it as a document level token classification task. The datasets are already tagged following
the B,I,O scheme. For more information on available datasets please refer - [Datasets](/DATASETS.md)

* **Step 3** - Initialize the training arguments.

```python
training_args = KETrainingArguments(
    output_dir="{path_to_your_directory}",
    learning_rate=4e-5,
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1000
)
```

* **Step 4** - Initialize the model arguments.

```python
model_args = KEModelArguments(
    model_name_or_path="roberta-base",
    use_crf=True,
)
```

You can have a CRF head for training your keyphrase extractors. CRF has proved to be effective and generally have shown
to give better performance. Refer - [Keyphrase Extraction from Scholarly Articles as Sequence Labeling using
Contextualized Embeddings](https://arxiv.org/pdf/1910.08840.pdf)

* **Step 5** - Train and evaluate the model.

```python
KeyphraseTagger.train_and_eval(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args,
)
```

* **Step 6** - Visualize your training progress using tensorboard.

```commandline
tensorboard --logdir {path_to_your_log_dir}
```

* **Step 7** - Load the trained model for prediction
```python
tagger = KeyphraseTagger.load(
    model_name_or_path="{path_to_your_directory_where_model_is_saved}"
    )

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

keyphrases = tagger.predict(input_text)

print(keyphrases)
```

Output:
```commandline
[['text documents', 'masking strategies', 'models', 'Keyphrase Boundary Infilling with Replacement', 'KBIR', 'KeyBART', 
'CatSeq', 'named entity recognition', 'question answering', 'relation extraction', 'abstractive summarization', 'NLP']]
```

## Keyphrase Generation

* **Step 1** - Import the required modules
```python
from dlkp.models import KeyphraseGenerator
from dlkp.generation import KGTrainingArguments, KGModelArguments, KGDataArguments
```

* **Step 2** - Initialize the data arguments.

```python
data_args = KGDataArguments(
    dataset_name="midas/inspec",
    dataset_config_name="generation",
    text_column_name="document",
    keyphrases_column_name="extractive_keyphrases",
    n_best_size=5,
    num_beams=3,
    cat_sequence=True,
)
```

This will automatically download the **inspec** corpus from the huggingface hub and prepare it for training a keyphrase 
generation model.

* **Step 3** - Initialize the training arguments.

```python
training_args = KGTrainingArguments(
    output_dir="{path_to_your_directory}",
    predict_with_generate=True,
    learning_rate=4e-5,
    overwrite_output_dir=True,
    num_train_epochs=50,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    do_train=True,
    do_eval=True,
    do_predict=False,
    eval_steps=1000,
    logging_steps=1000
)
```
* **Step 4** - Initialize the model arguments.

```python
model_args = KGModelArguments(model_name_or_path="bloomberg/KeyBART")
```

* **Step 5** - Train and evaluate the model.

```python
 KeyphraseGenerator.train_and_eval(model_args, data_args, training_args)

```
* **Step 6** - Visualize your training progress using tensorboard.

```commandline
tensorboard --logdir {path_to_your_log_dir}
```

* **Step 7** - Load the trained model for prediction
```python
generator = KeyphraseGenerator.load(
    "{path_to_your_directory_where_model_is_saved}"
)
input_text = "Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned. Random decision forests correct for decision trees' habit of overfitting to their training set.Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees. However, data characteristics can affect their performance."

generator_out = generator.generate(input_text)
print(generator_out)
```
Output:
```commandline
['random decision forests [KP_SEP] ensemble learning method [KP_SEP] classification [KP_SEP] regression [KP_SEP] data characteristics']
```

for one2many keyphrase generation tasks, generated keyphrases are seprated by a seprator token (i.e. [KP_SEP]) as shown in above example. 


