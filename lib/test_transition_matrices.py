import numpy as np

from .transition_matrices import tune_transition_matrix


def test_that_neutral_tuning_leaves_normalized_matrices_untouched():
    normalized_matrix = np.array([[np.sqrt(2), np.sqrt(2)], [0.0, 1.0]])

    tuned_matrix = tune_transition_matrix(normalized_matrix, 1, 1)

    assert np.all(np.equal(tuned_matrix, normalized_matrix))


def test_that_neutral_tuning_normalizes():
    unnormalized_matrix = np.array([[5.0, 0.0], [2.0, 0.0]])

    tuned_matrix = tune_transition_matrix(unnormalized_matrix, 1, 1)

    assert np.all(np.equal(tuned_matrix, np.array([[1.0, 0.0], [1.0, 0.0]])))
