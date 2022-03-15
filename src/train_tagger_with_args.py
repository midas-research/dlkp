from dlkp.models import KeyphraseTagger
from dlkp.extraction import (
    KpExtDataArguments,
    KpExtModelArguments,
    KpExtTrainingArguments,
)


training_args = KpExtTrainingArguments(
    output_dir="/media/nas_mount/Debanjan/amardeep/dlkp_out/inpec_debug_eval",  # todo
    learning_rate=3e-5,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # gradient_accumulation_steps=4,
    do_train=False,
    do_eval=False,
    do_predict=True,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=100,
    # lr_scheduler_type= 'cosine',
    # warmup_steps=200,
    logging_steps=100
    # weight_decay =0.001
)
model_args = KpExtModelArguments(
    model_name_or_path="/media/nas_mount/Debanjan/amardeep/dlkp_out/inpec_debug",
    use_crf=False,
)
data_args = KpExtDataArguments(
    # train_file="/media/nas_mount/Debanjan/amardeep/proc_data/kp20k/medium/conll/train.json",
    # validation_file="/media/nas_mount/Debanjan/amardeep/proc_data/kp20k/medium/conll/test.json",
    dataset_name="midas/inspec",
    dataset_config_name="extraction",
    pad_to_max_length=True,
    overwrite_cache=True,
    label_all_tokens=True,
    preprocessing_num_workers=8,
    # return_entity_level_metrics=True,
)
KeyphraseTagger.train_and_eval(
    model_args=model_args, data_args=data_args, training_args=training_args
)

# CUDA_VISIBLE_DEVICES=0 python run_auto_ke.py
