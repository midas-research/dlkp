from dlkp.generation import KGTrainingArguments, KGModelArguments, KGDataArguments

from dlkp.generation.train_eval_generator import train_eval_generator

model_args = KGModelArguments(
    model_name_or_path="/media/nas_mount/Debanjan/amardeep/dlkp_out/inspec_t5_gen"
)

data_args = KGDataArguments(
    dataset_name="midas/inspec",
    dataset_config_name="generation",
    text_column_name="document",
    keyphrases_column_name="extractive_keyphrases",
    max_test_samples=5,
    n_best_size=5,
    num_beams=3,
)

training_args = KGTrainingArguments(
    predict_with_generate=True,
    output_dir="/media/nas_mount/Debanjan/amardeep/dlkp_out/inspec_t5_gen_predict",
    learning_rate=3e-5,
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # gradient_accumulation_steps=4,
    do_train=False,
    do_eval=False,
    do_predict=True,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    # lr_scheduler_type= 'cosine',
    # warmup_steps=200,
    logging_steps=100
    # weight_decay =0.001
)

train_eval_generator(model_args, data_args, training_args)
