from dlkp.generation import KGTrainingArguments, KGModelArguments, KGDataArguments
from dlkp.models import KeyphraseGenerator

model_args = KGModelArguments(model_name_or_path="bloomberg/KeyBART")

data_args = KGDataArguments(
    dataset_name="midas/inspec",
    dataset_config_name="generation",
    text_column_name="document",
    keyphrases_column_name="extractive_keyphrases",
    n_best_size=10,
    num_beams=50,
    max_seq_length=512,
)

training_args = KGTrainingArguments(
    predict_with_generate=True,
    output_dir="/data/models/keyphrase/dlkp/inspec/generation/bart-large/finetuned",
    learning_rate=4e-5,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    do_train=True,
    do_eval=True,
    do_predict=False,
    evaluation_strategy="steps",
    save_steps=500,
    eval_steps=500,
    logging_steps=500,
    metric_for_best_model="eval_F1@5",
    load_best_model_at_end=True,
)
gen = KeyphraseGenerator.train_and_eval(model_args, data_args, training_args)

gen = KeyphraseGenerator.load(
    "/data/models/keyphrase/dlkp/inspec/generation/bart-large/finetuned"
)

input_text = (
    "In this work, we explore how to learn task-specific language models aimed towards learning rich "
    "representation of keyphrases from text documents. We experiment with different masking strategies for "
    "pre-training transformer language models (LMs) in discriminative as well as generative settings. In the "
    "discriminative setting, we introduce a new pre-training objective - Keyphrase Boundary Infilling with "
    "Replacement (KBIR), showing large gains in performance (upto 9.26 points in F1) over SOTA, when LM "
    "pre-trained using KBIR is fine-tuned for the task of keyphrase extraction. In the generative setting, we "
    "introduce a new pre-training setup for BART - KeyBART, that reproduces the keyphrases related to the "
    "input text in the CatSeq format, instead of the denoised original input. This also led to gains in "
    "performance (upto 4.33 points in F1@M) over SOTA for keyphrase generation. Additionally, we also "
    "fine-tune the pre-trained language models on named entity recognition (NER), question answering (QA), "
    "relation extraction (RE), abstractive summarization and achieve comparable performance with that of the "
    "SOTA, showing that learning rich representation of keyphrases is indeed beneficial for many other "
    "fundamental NLP tasks."
)

gen_out = gen.generate(input_text, num_return_sequences=10, output_seq_score=True, num_beams=50)
print(gen_out)
print(set([kp.strip() for kp in gen_out[0][0].split("[KP_SEP]")]))
