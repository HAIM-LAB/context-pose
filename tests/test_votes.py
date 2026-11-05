import numpy as np

from asd_contextpose.votes import (
    annotator_ceiling,
    auc,
    binary_scores,
    decode,
    encode,
    majority,
    parse_votes,
    soft_target,
)


def test_parse_votes_maps_cells_and_unknowns():
    record = {
        "valence__B": "0.5:4",
        "valence__A": "1:5",
        "valence__C": "2:3",
        "arousal__A": "0:5",
        "arousal__B": "1:2",
        "arousal__C": None,
    }
    assert parse_votes(record, "valence") == [1, 0, None]  # annotators sorted
    assert parse_votes(record, "arousal") == [-1, 1, None]
    assert decode(encode([1, 0, None])) == [1, 0, None]


def test_soft_target_and_majority_ignore_unknown_votes():
    assert soft_target([1, 0, None], "valence") == 0.5
    assert majority([1, 0, None], "valence") is None  # tie
    assert majority([1, 1, -1], "valence") == 1
    assert majority([1, 0, -1], "valence") == 0  # neutral counts as not positive
    assert soft_target([-1, -1, 1], "arousal") == 1 / 3
    assert majority([-1, -1, 1], "arousal") == 0


def test_binary_scores_balance_the_classes():
    labels = np.array([0, 0, 0, 1])
    scores = binary_scores(labels, np.zeros(4, int))
    assert scores["accuracy"] == 0.75 and scores["balanced_accuracy"] == 0.5


def test_annotator_ceiling_scores_each_annotator_against_the_other_two():
    votes = [[1, 1, 1], [1, 1, -1], [-1, 1, None]]
    ceiling = annotator_ceiling(votes, "arousal")
    # Clip 1: three agreeing comparisons; clip 2: only annotator 3 (wrong);
    # clip 3: the unknown vote leaves no complete pair.
    assert ceiling["n"] == 4 and ceiling["accuracy"] == 0.75


def test_auc_ranks_and_ties():
    labels = np.array([0, 0, 1, 1])
    assert auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert auc(np.zeros(4), labels) == 0.5
    assert auc(np.array([0.1, 0.5, 0.5, 0.9]), labels) == 0.875
    assert auc(np.zeros(3), np.ones(3)) is None
