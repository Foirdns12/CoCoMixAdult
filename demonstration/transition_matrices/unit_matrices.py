import numpy as np

from demonstration.demonstration_data import ALL_CATEGORICAL_VALUES
from demonstration.transition_matrices.compute_distance_matrices import compute_distance_matrix


def get_unit_transition_matrices(norm=True):
    transition_matrices = {}
    for feature, values in ALL_CATEGORICAL_VALUES.items():
        transition_matrix = np.ones(shape=(len(values), len(values)), dtype=np.float)
        if norm:
            transition_matrix = transition_matrix[:] / np.sum(transition_matrix, axis=1)
        transition_matrices[feature] = transition_matrix

    return transition_matrices


def get_subspace_unit_transition_matrix(values, norm=True):
    transition_matrix = np.ones(shape=(len(values), len(values)), dtype=np.float)
    if norm:
        transition_matrix = transition_matrix[:] / np.sum(transition_matrix, axis=1)
    return transition_matrix


def get_unit_distance_matrices(norm=True):
    return {feature: compute_distance_matrix(transition_matrix,True)
            for feature, transition_matrix in get_unit_transition_matrices(norm).items()}


if __name__ == "__main__":
    print(get_unit_transition_matrices())
    print(get_unit_distance_matrices())
