from dlkp.generation import KGTrainingArguments, KGModelArguments, KGDataArguments

from dlkp.generation.train_eval_generator import train_and_eval_generation_model

from dlkp.models import KeyphraseGenerator

model_args = KGModelArguments(model_name_or_path="google/bigbird-roberta-base")

data_args = KGDataArguments(
    dataset_name="midas/ldkp3k",
    dataset_config_name="generation",
    max_eval_samples=1000,
    n_best_size=5,
    num_beams=3,
    cat_sequence=True,
    # max_seq_length=100,
)

training_args = KGTrainingArguments(
    predict_with_generate=True,
    output_dir="/media/nas_mount/Debanjan/amardeep/dlkp_out/inspec_t5_gen_predict",
    learning_rate=3e-5,
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    # gradient_accumulation_steps=4,
    do_train=True,
    do_eval=False,
    do_predict=True,
    evaluation_strategy="steps",
    # save_steps=500,
    eval_steps=500,
    # lr_scheduler_type= 'cosine',
    # warmup_steps=200,
    logging_steps=100
    # weight_decay =0.001
)
# ppp = KeyphraseGenerator.train_and_eval(model_args, data_args, training_args)

# train_eval_generator(model_args, data_args, training_args)

#

gen = KeyphraseGenerator.load(
    "/media/nas_mount/Debanjan/amardeep/dlkp_out/inspec_t5_gen_predict"
)
text = "Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time. For classification tasks, the output of the random forest is the class selected by most trees. For regression tasks, the mean or average prediction of the individual trees is returned. Random decision forests correct for decision trees' habit of overfitting to their training set.Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees. However, data characteristics can affect their performance."

gen_out = gen.generate(text, num_return_sequences=2, output_seq_score=True)

print(gen_out)
