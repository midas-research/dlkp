from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2, IOB1
import numpy as np


def compute_metrics(p):
    return_entity_level_metrics = False
    ignore_value = -100
    predictions, labels = p
    label_to_id = {"B": 0, "I": 1, "O": 2}
    id_to_label = ["B", "I", "O"]
    # if model_args.use_CRF is False:
    predictions = np.argmax(predictions, axis=2)
    # print(predictions.shape, labels.shape)

    # Remove ignored index (special tokens)
    true_predictions = [
        [id_to_label[p] for (p, l) in zip(prediction, label) if l != ignore_value]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id_to_label[l] for (p, l) in zip(prediction, label) if l != ignore_value]
        for prediction, label in zip(predictions, labels)
    ]

    # results = metric.compute(predictions=true_predictions, references=true_labels)
    results = {}
    # print("cal precisi")
    # mode="strict"
    results["overall_precision"] = precision_score(
        true_labels, true_predictions, scheme=IOB2
    )
    results["overall_recall"] = recall_score(true_labels, true_predictions, scheme=IOB2)
    # print("cal f1")
    results["overall_f1"] = f1_score(true_labels, true_predictions, scheme=IOB2)
    results["overall_accuracy"] = accuracy_score(true_labels, true_predictions)
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        # print("cal entity level mat")
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
