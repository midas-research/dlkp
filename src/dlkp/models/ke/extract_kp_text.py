import os, json
from .extraction_utils import TrainingArguments, DataTrainingArguments, ModelArguments
from .kpe import run_extraction_model


def extract_from_text(
    text_list, model_name_or_path, use_CRF=False, output_dir="eval_output"
):
    # if output_dir is None:
    #     output_dir =
    # create a file and pass to extractor
    os.makedirs(output_dir, exist_ok=True)
    test_file = os.path.join(output_dir, "test_file.json")
    with open(test_file, "w", encoding="utf-8") as fp:
        for i, txt in enumerate(text_list):
            data = {}
            data["id"] = i
            data["document"] = txt.split()
            fp.write(json.dumps(data))
            fp.write("\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=4,
        do_train=False,
        do_eval=False,
        do_predict=True,
    )
    model_args = ModelArguments(model_name_or_path=model_name_or_path, use_CRF=use_CRF)
    data_args = DataTrainingArguments(
        test_file=test_file,
        pad_to_max_length=True,
        overwrite_cache=True,
        label_all_tokens=True,
        preprocessing_num_workers=8,
    )
    return run_extraction_model(model_args, data_args, training_args)
