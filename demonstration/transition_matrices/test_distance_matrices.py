import numpy as np

from demonstration.transition_matrices.compute_distance_matrices_adult import compute_distance_matrix


def test_compute_for_unit_matrix():
    transition_matrix = np.full(shape=(3, 3), fill_value=np.sqrt(1/3))

    expected_distance_matrix = np.full_like(transition_matrix, np.sqrt(1/3))
    np.fill_diagonal(expected_distance_matrix, 1.0)
    assert np.all(compute_distance_matrix(transition_matrix) == expected_distance_matrix)


def test_compute_for_simple_case():
    transition_matrix = np.full(shape=(3, 3), fill_value=0.5)

    expected_distance_matrix = np.full_like(transition_matrix, 0.5)
    np.fill_diagonal(expected_distance_matrix, 1.0)

    assert np.all(compute_distance_matrix(transition_matrix) == expected_distance_matrix)


def test_compute_for_complex_case():
    transition_matrix = np.array([
        [0.5, 0.5, 0.0],
        [0.1, 0.5, 0.5],
        [0.0, 0.0, 1.0]
    ])

    expected_distance_matrix = np.array([
        [1.0, 0.5, 0.25],
        [0.1, 1.0, 0.5],
        [0.0, 0.0, 1.0]
    ])

    assert np.all(compute_distance_matrix(transition_matrix) == expected_distance_matrix)
