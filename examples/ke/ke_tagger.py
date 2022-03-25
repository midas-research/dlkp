from dlkp.models import KeyphraseTagger
from dlkp.extraction import (
    KEDataArguments,
    KEModelArguments,
    KETrainingArguments,
)

training_args = KETrainingArguments(
    output_dir="/data/models/keyphrase/dlkp/inspec/kbir/finetuned-crf",
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
model_args = KEModelArguments(
    model_name_or_path="bloomberg/KBIR",
    use_crf=True,
)
data_args = KEDataArguments(
    dataset_name="midas/inspec",
    dataset_config_name="extraction",
    pad_to_max_length=True,
    overwrite_cache=True,
    label_all_tokens=False,
    preprocessing_num_workers=8,
    return_entity_level_metrics=True,
)
KeyphraseTagger.train_and_eval(
    model_args=model_args,
    data_args=data_args,
    training_args=training_args
)

tagger = KeyphraseTagger.load(
    model_name_or_path="/data/models/keyphrase/dlkp/inspec/kbir/finetuned-crf"
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
