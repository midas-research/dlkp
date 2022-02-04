"""
Conditional random field utilis file 
"""
from typing import List, Tuple, Dict, Union, Optional

import torch
import math
import logging

# from allennlp.common.checks import ConfigurationError
# import allennlp.nn.util as util

VITERBI_DECODING = Tuple[List[int], float]  # a list of tags, and a viterbi score

logger = logging.get_logger(__name__)


def allowed_transitions(
    constraint_type: str, labels: Dict[int, str]
) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.
    # Parameters
    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : `Dict[int, str]`, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()
    # Returns
    `List[Tuple[int, int]]`
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [
        (start_tag, "START"),
        (end_tag, "END"),
    ]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(
                constraint_type, from_tag, from_entity, to_tag, to_entity
            ):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(
    constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str
):
    """
    Given a constraint type and strings `from_tag` and `to_tag` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.
    # Parameters
    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    from_tag : `str`, required
        The tag that the transition originates from. For example, if the
        label is `I-PER`, the `from_tag` is `I`.
    from_entity : `str`, required
        The entity corresponding to the `from_tag`. For example, if the
        label is `I-PER`, the `from_entity` is `PER`.
    to_tag : `str`, required
        The tag that the transition leads to. For example, if the
        label is `I-PER`, the `to_tag` is `I`.
    to_entity : `str`, required
        The entity corresponding to the `to_tag`. For example, if the
        label is `I-PER`, the `to_entity` is `PER`.
    # Returns
    `bool`
        Whether the transition is allowed under the given `constraint_type`.
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ("O", "B", "U")
        if to_tag == "END":
            return from_tag in ("O", "L", "U")
        return any(
            [
                # O can transition to O, B-* or U-*
                # L-x can transition to O, B-*, or U-*
                # U-x can transition to O, B-*, or U-*
                from_tag in ("O", "L", "U") and to_tag in ("O", "B", "U"),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ("B", "I")
                and to_tag in ("I", "L")
                and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ("O", "B")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or B-x
                to_tag in ("O", "B"),
                # Can only transition to I-x from B-x or I-x
                to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ("O", "I")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or I-x
                to_tag in ("O", "I"),
                # Can only transition to B-x from B-x or I-x, where
                # x is the same tag.
                to_tag == "B" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ("B", "S")
        if to_tag == "END":
            return from_tag in ("E", "S")
        return any(
            [
                # Can only transition to B or S from E or S.
                to_tag in ("B", "S") and from_tag in ("E", "S"),
                # Can only transition to M-x from B-x, where
                # x is the same tag.
                to_tag == "M" and from_tag in ("B", "M") and from_entity == to_entity,
                # Can only transition to E-x from B-x or M-x, where
                # x is the same tag.
                to_tag == "E" and from_tag in ("B", "M") and from_entity == to_entity,
            ]
        )
    else:
        print("error in constrint type")


def logsumexp(
    tensor: torch.Tensor, dim: int = -1, keepdim: bool = False
) -> torch.Tensor:
    """
    A numerically stable computation of logsumexp. This is mathematically equivalent to
    `tensor.exp().sum(dim, keep=keepdim).log()`.  This function is typically used for summing log
    probabilities.
    # Parameters
    tensor : `torch.FloatTensor`, required.
        A tensor of arbitrary size.
    dim : `int`, optional (default = `-1`)
        The dimension of the tensor to apply the logsumexp to.
    keepdim: `bool`, optional (default = `False`)
        Whether to retain a dimension of size one at the dimension we reduce over.
    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    tag_observations: Optional[List[int]] = None,
    allowed_start_transitions: torch.Tensor = None,
    allowed_end_transitions: torch.Tensor = None,
    top_k: int = None,
):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    # Parameters
    tag_sequence : `torch.Tensor`, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : `torch.Tensor`, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    tag_observations : `Optional[List[int]]`, optional, (default = `None`)
        A list of length `sequence_length` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labelings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.
    allowed_start_transitions : `torch.Tensor`, optional, (default = `None`)
        An optional tensor of shape (num_tags,) describing which tags the START token
        may transition *to*. If provided, additional transition constraints will be used for
        determining the start element of the sequence.
    allowed_end_transitions : `torch.Tensor`, optional, (default = `None`)
        An optional tensor of shape (num_tags,) describing which tags may transition *to* the
        end tag. If provided, additional transition constraints will be used for determining
        the end element of the sequence.
    top_k : `int`, optional, (default = `None`)
        Optional integer specifying how many of the top paths to return. For top_k>=1, returns
        a tuple of two lists: top_k_paths, top_k_scores, For top_k==None, returns a flattened
        tuple with just the top path and its score (not in lists, for backwards compatibility).
    # Returns
    viterbi_path : `List[int]`
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : `torch.Tensor`
        The score of the viterbi path.
    """
    if top_k is None:
        top_k = 1
        flatten_output = True
    elif top_k >= 1:
        flatten_output = False
    else:
        raise ValueError(
            f"top_k must be either None or an integer >=1. Instead received {top_k}"
        )

    sequence_length, num_tags = list(tag_sequence.size())

    has_start_end_restrictions = (
        allowed_end_transitions is not None or allowed_start_transitions is not None
    )

    if has_start_end_restrictions:

        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)

        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix

        # Start and end transitions are fully defined, but cannot transition between each other.

        allowed_start_transitions = torch.cat(
            [allowed_start_transitions, torch.tensor([-math.inf, -math.inf])]
        )
        allowed_end_transitions = torch.cat(
            [allowed_end_transitions, torch.tensor([-math.inf, -math.inf])]
        )

        # First define how we may transition FROM the start and end tags.
        new_transition_matrix[-2, :] = allowed_start_transitions
        # We cannot transition from the end tag to any tag.
        new_transition_matrix[-1, :] = -math.inf

        new_transition_matrix[:, -1] = allowed_end_transitions
        # We cannot transition to the start tag from any tag.
        new_transition_matrix[:, -2] = -math.inf

        transition_matrix = new_transition_matrix

    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ConfigurationError(
                "Observations were provided, but they were not the same length "
                "as the sequence. Found sequence of length: {} and evidence: {}".format(
                    sequence_length, tag_observations
                )
            )
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.0
        path_scores.append(one_hot.unsqueeze(0))
    else:
        path_scores.append(tag_sequence[0, :].unsqueeze(0))

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        # Best pairwise potential path score from the previous timestep.
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1 and observation != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning(
                    "The pairwise potential between tags you have passed as "
                    "observations is extremely unlikely. Double check your evidence "
                    "or transition potentials!"
                )
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.0
            path_scores.append(one_hot.unsqueeze(0))
        else:
            path_scores.append(tag_sequence[timestep, :] + scores)
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    path_scores_v = path_scores[-1].view(-1)
    max_k = min(path_scores_v.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()

        if has_start_end_restrictions:
            viterbi_path = viterbi_path[1:-1]

        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)

    if flatten_output:
        return viterbi_paths[0], viterbi_scores[0]

    return viterbi_paths, viterbi_scores
