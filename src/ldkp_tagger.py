from dlkp.models import KeyphraseTagger
from dlkp.extraction import (
    KEDataArguments,
    KEModelArguments,
    KETrainingArguments,
)

text_column_name = "text"
label_column_name = "bio_tags"


def preprocess_extraction(ex):
    """
    input: a dict of row of your dataset will be passed to this function. you can mainoulate your original data here.
    output : should return a dictionary which should contain fields equal to text_column_name and label_column_name

    """
    imp_sec_list = [
        "title",
        "abstract",
        "introduction",
        "conclusion",
        "discussion",
        "results",
        "related work",
        "background",
        "methodology",
    ]
    text = []
    bio_tags = []
    for sec, txt, tags in zip(ex["sections"], ex["sec_text"], ex["sec_bio_tags"]):
        if sec in imp_sec_list:
            text += txt
            bio_tags += tags
    assert len(text) == len(bio_tags)
    ex[text_column_name] = text
    ex[label_column_name] = bio_tags
    return ex


model_name = "allenai/longformer-base-4096"

print("Training: ", model_name)
training_args = KETrainingArguments(
    output_dir=f"/media/nas_mount/Debanjan/amardeep/long_ldkp/tagger/longformer-ldkp3k",
    learning_rate=4e-5,
    overwrite_output_dir=True,
    num_train_epochs=6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=2,
    do_train=True,
    do_eval=False,
    do_predict=False,
    evaluation_strategy="steps",
    save_steps=2000,
    eval_steps=2000,
    logging_steps=1000,
)

model_args = KEModelArguments(
    model_name_or_path=model_name,
    use_crf=False,
)

data_args = KEDataArguments(
    dataset_name="midas/ldkp3k",
    dataset_config_name="small",
    pad_to_max_length=True,
    overwrite_cache=True,
    preprocessing_num_workers=8,
    return_entity_level_metrics=True,
    preprocess_func=preprocess_extraction,
    text_column_name=text_column_name,
    label_column_name=label_column_name,
    cache_dir="/media/nas_mount/Debanjan/amardeep/long_ldkp/hf_cache",
)

KeyphraseTagger.train_and_eval(
    model_args=model_args, data_args=data_args, training_args=training_args
)
