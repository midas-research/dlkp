from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2, IOB1
import numpy as np


def compute_metrics(
    predictions, labels, return_entity_level_metrics=True, ignore_value=-100
):
    # predictions, labels = p
    # print(predictions.shape, labels.shape)
    # if model_args.use_CRF is False:
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != ignore_value]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != ignore_value]
        for prediction, label in zip(predictions, labels)
    ]

    # results = metric.compute(predictions=true_predictions, references=true_labels)
    results = {}
    # print("cal precisi")
    results["overall_precision"] = precision_score(
        true_labels, true_predictions, mode="strict", scheme=IOB2
    )
    results["overall_recall"] = recall_score(
        true_labels, true_predictions, mode="strict", scheme=IOB2
    )
    # print("cal f1")
    results["overall_f1"] = f1_score(
        true_labels, true_predictions, mode="strict", scheme=IOB2
    )
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
