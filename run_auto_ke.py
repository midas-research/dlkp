from statistics import mode
from dlkp.models.ke.kpe import run_kpe
from dlkp.models.ke.extraction_utils import (
    DataTrainingArguments,
    ModelArguments,
    TrainingArguments,
)

training_args = TrainingArguments(
    output_dir="/media/nas_mount/Debanjan/amardeep/dlkp_out/inpec_crf_debug",  # todo
    learning_rate=3e-5,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    # gradient_accumulation_steps=4,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=100,
    # lr_scheduler_type= 'cosine',
    # warmup_steps=200,
    logging_steps=100
    # weight_decay =0.001
)
model_args = ModelArguments(model_name_or_path="roberta-base", use_CRF=True)
data_args = DataTrainingArguments(
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

run_kpe(model_args, data_args, training_args)

# CUDA_VISIBLE_DEVICES=0 python run_auto_ke.py
