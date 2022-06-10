import os
import pickle

import numpy as np

from demonstration.transition_matrices.final_matrices import load_transition_matrices
from demonstration.demonstration_data import ALL_CATEGORICAL_VALUES

PATH = os.path.dirname(os.path.abspath(__file__))


def compute_distance_matrix(transition_matrix: np.ndarray,unit=False) -> np.ndarray:
    assert transition_matrix.ndim == 2
    assert transition_matrix.shape[0] == transition_matrix.shape[1]

    # To avoid any side-effects, we make a copy and never modify the original array
    _transition_matrix = transition_matrix.copy()
    np.fill_diagonal(_transition_matrix, 0.0)
    distm=most_likely_path(_transition_matrix.copy(), _transition_matrix)
    if unit:
        distm=distm*((len(distm))/25)
        np.fill_diagonal(distm, 1.0)

    return distm


def most_likely_path(distance_matrix, transition_matrix, step=1):
    if step == distance_matrix.shape[0]:
        np.fill_diagonal(distance_matrix, 1.0)
        return distance_matrix
    else:
        next_distance_matrix = distance_matrix.copy()
        np.fill_diagonal(next_distance_matrix, 0.0)

        lower_distance_matrix = most_likely_path(
            np.dot(next_distance_matrix, transition_matrix),
            transition_matrix, step + 1)

    distance_matrix = np.maximum(distance_matrix, lower_distance_matrix)

    return distance_matrix


if __name__ == "__main__":
    transition_matrices = load_transition_matrices()

    distance_matrices = {}
    reflength = 0
    for i in ALL_CATEGORICAL_VALUES:
        reflength = np.maximum(reflength, len(i))

    for feature, transition_matrix in transition_matrices.items():
        distance_matrix = compute_distance_matrix(transition_matrix)

        distance_matrix = (distance_matrix * (len(distance_matrix) - 1)) / (reflength - 1)
        np.fill_diagonal(distance_matrix, 1.0)

        distance_matrices[feature] = distance_matrix

    print(distance_matrices)

    with open(os.path.join(PATH, "final_distance_matrices_adult.pickle"), "wb") as f:
        pickle.dump(distance_matrices, f)
