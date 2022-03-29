from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2, IOB1
import numpy as np

from nltk.stem.porter import PorterStemmer


def compute_metrics(p):
    return_entity_level_metrics = False
    ignore_value = -100
    predictions, labels = p
    label_to_id = {"B": 0, "I": 1, "O": 2}
    id_to_label = ["B", "I", "O"]
    # if model_args.use_crf is False:
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


def compute_kp_level_metrics(predictions, originals, do_stem=True):
    avg_metrics = {}
    assert len(predictions) == len(
        originals
    ), "len of pridicted examples and original examples is not same"
    total_ex = len(predictions)
    for predicted_kps, original_kps in zip(predictions, originals):
        metrics = compute_f1s(predicted_kps, original_kps, do_stem)
        for k in metrics:
            if k not in avg_metrics:
                avg_metrics[k] = 0.0
            avg_metrics[k] += metrics[k]

    if total_ex > 0:
        for k in avg_metrics:
            avg_metrics[k] /= total_ex

    return avg_metrics


def compute_f1s(predicted_kps, original_kps, do_stem=True):
    stemmer = PorterStemmer()
    if do_stem:
        predicted_kps = [
            " ".join([stemmer.stem(word) for word in keyphrase.split()])
            for keyphrase in predicted_kps
        ]
        original_kps = [
            " ".join([stemmer.stem(word) for word in keyphrase.split()])
            for keyphrase in original_kps
        ]

    predicted_kps = set(predicted_kps)
    original_kps = set(original_kps)

    correctly_matched = [1 if kp in original_kps else 0 for kp in predicted_kps]

    metrics = []
    for k in [5, 10, "m"]:
        metrics.append(calculate_f1_k(correctly_matched, original_kps, k))

    results = {}
    for metric in metrics:
        for key, value in metric.items():
            results[key] = value

    return results


def calculate_f1_k(correctly_matched, original_kps, k):
    m_name = k
    if isinstance(k, str):
        k = len(correctly_matched)
    metrics = {}

    metrics["P@{}".format(m_name)] = (
        float(sum(correctly_matched[:k])) / float(k) if k != 0 else 0.0
    )
    metrics["R@{}".format(m_name)] = (
        float(sum(correctly_matched[:k])) / float(len(original_kps))
        if len(original_kps) != 0
        else 0.0
    )
    if metrics["P@{}".format(m_name)] + metrics["R@{}".format(m_name)] > 0:
        metrics["F1@{}".format(m_name)] = (
            2
            * metrics["P@{}".format(m_name)]
            * metrics["R@{}".format(m_name)]
            / float(metrics["P@{}".format(m_name)] + metrics["R@{}".format(m_name)])
        )
    else:
        metrics["F1@{}".format(m_name)] = 0.0

    return metrics
