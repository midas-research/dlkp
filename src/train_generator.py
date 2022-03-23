from dlkp.generation import KGTrainingArguments, KGModelArguments, KGDataArguments

from dlkp.generation.train_eval_generator import train_eval_tagger

model_args = KGModelArguments(model_name_or_path="t5-base")

data_args = KGDataArguments(
    dataset_name="midas/inspec",
    dataset_config_name="generation",
    text_column_name="document",
    keyphrases_column_name="extractive_keyphrases",
    n_best_size=5,
    num_beams=3,
)

training_args = KGTrainingArguments(
    # predict_with_generate =True,
    output_dir="/media/nas_mount/Debanjan/amardeep/dlkp_out/inspec_bart_gen",
    learning_rate=3e-5,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # gradient_accumulation_steps=4,
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluation_strategy="steps",
    save_steps=1000,
    eval_steps=10,
    # lr_scheduler_type= 'cosine',
    # warmup_steps=200,
    logging_steps=100
    # weight_decay =0.001
)

train_eval_tagger(model_args, data_args, training_args)