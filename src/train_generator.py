from dlkp.generation import KGTrainingArguments, KGModelArguments, KGDataArguments

from dlkp.generation.train_eval_generator import train_and_eval_generation_model

from dlkp.models import KeyphraseGenerator

model_args = KGModelArguments(model_name_or_path="t5-base")

data_args = KGDataArguments(
    dataset_name="midas/inspec",
    dataset_config_name="generation",
    text_column_name="document",
    keyphrases_column_name="extractive_keyphrases",
    max_test_samples=50,
    max_eval_samples=50,
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
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=50,
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
