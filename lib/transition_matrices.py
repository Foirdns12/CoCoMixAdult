import os
import time
from typing import Dict
import pickle

import numpy as np

PATH = os.path.dirname(os.path.abspath(__file__))


def compute_transition_matrix(samples, feature_idx, pdf, the_pdf=None,
                              categorical_values=None, quiet=False) -> np.ndarray:
    """Compute transition matrix for feature with index *feature_idx*.

    :param samples:
    :param feature_idx:
    :param pdf: Function to compute PDF.
    :param the_pdf: Pre-computed PDF.
    :param categorical_values: Exhaustive list of values the categorical variable can take on.
    :param quiet: If *True*, print verbose information about computation steps.
    :return: Transition matrix for the feature with index *feature_idx*
    """
    print(f"Compute transition matrix for feature {feature_idx}")
    start = time.time()

    def _print(text, *args, **kwargs):
        if not quiet:
            print(text, *args, **kwargs)

    if the_pdf is None:
        _print("Compute KDE of samples")
        the_pdf = pdf(samples)

    if categorical_values is None:
        values = np.unique(samples[:, feature_idx])
    else:
        values = categorical_values
        if not np.all([val in values for val in np.unique(samples[:, feature_idx])]):
            values_found = list(np.unique(samples[:, feature_idx]))
            for val in values_found:
                if val not in values:
                    print(f"Value {val} was not found in {values}")
            raise ValueError("Unknown values in samples!")

    _print(f"- There are {len(values)} unique values")
    values = np.array(values)

    _print(f"- Compute KDE for {len(values)} possible variable settings")
    next_state = samples.copy()
    next_state_pdfs = []

    for i, other in enumerate(values):
        _print(f"  {i + 1}/{len(values)}", end="")
        fname = os.path.join(PATH, f"feature_{feature_idx}_{i}.pickle")

        if os.path.exists(fname):
            with open(fname, "rb") as f:
                other_pdf = pickle.load(f)
        else:
            next_state[:, feature_idx] = other
            other_pdf = pdf(next_state)

            with open(fname, "wb") as f:
                pickle.dump(other_pdf, f)

        assert other_pdf.shape == the_pdf.shape
        next_state_pdfs.append(other_pdf)

    _print(f"\n- Fill transition matrix")
    transition_matrix = np.zeros((values.shape[0], values.shape[0]))
    for i, value in enumerate(values):
        value_mask = samples[:, feature_idx] == value
        assert value_mask.shape == the_pdf.shape

        _print(f"  ({i},{i})", end="")
        transition_matrix[i, i] = np.mean(the_pdf[value_mask])

        # compute off-diagonal elements
        for j, _ in enumerate(values):
            # Note: i = j can be computed in the very same way, but for sake of clarity
            #       we only handle the off-diagonal elements here
            if i != j:
                _print(f", ({i},{j})", end="")
                transition_matrix[i, j] = np.mean(next_state_pdfs[j][value_mask])

        _print("")

    print(f"Finished computing transition matrix for feature {feature_idx} in "
          f"{time.time() - start:0.2f} seconds.")

    return transition_matrix


def tune_transition_matrix(transition_matrix: np.ndarray, a: float, b: float) -> np.ndarray:
    """

    :param transition_matrix:
    :param a: Control the distribution of probability of the off-diagonal elements.
              The smaller a, the more level the off-diagonal probabilities.
    :param b: Control the balance between the diagonal and off-diagonal elements.
              The larger b, the higher the diagonal elements.
    :return: Normalized and tuned transition matrix.
    """
    for i in range(transition_matrix.shape[0]):
        row = transition_matrix[i, :]

        if np.mean(np.delete(row, i)) == 0.0:
            p_stay = 1.0
            p_switch = np.zeros_like(row)
        else:
            p_stay = -1.0 + (2.0 / (1.0 + np.exp(-b * row[i] / np.mean(np.delete(row, i)))))
            p_switch = (1.0 - p_stay) * np.power(row, a) / np.sum(np.power(np.delete(row, i), a))

        row = p_switch
        row[i] = p_stay

        transition_matrix[i, :] = row

    return transition_matrix


def estimate_transition_matrices(samples: np.ndarray, pdf, features, var_types,
                                 num_processes=1, categorical_values=None, exclude=None) -> Dict[str, np.ndarray]:
    """Estimate transition matrices based on the provided *samples* and *pdf.

    :param samples:
    :param pdf:
    :param features:
    :param var_types:
    :param num_processes: Number of CPU processes to use for computation
    :param categorical_values: If given, dictionary that contains the list of categorical values of each feature
    :param exclude: List of features to exclude
    :return: Dictionary of transition matrices with feature names as the keys.
    """
    exclude = exclude or []

    if num_processes < 1:
        raise ValueError("At least a single process is required.")

    if samples.shape[1] != len(features):
        raise ValueError("Array of samples contains more columns than features given.")

    if categorical_values is None:
        categorical_values = {feature: None for feature in features}

    transition_matrices = {}

    print("Compute KDE of samples")
    the_pdf = pdf(samples)

    if num_processes == 1:
        for feature_idx, (var_type, feature) in enumerate(zip(var_types, features)):
            if var_type != "c" and feature not in exclude:
                print(f"Estimate transition matrix for feature {feature} of type {var_type}")
                transition_matrix = compute_transition_matrix(samples, feature_idx, pdf, the_pdf,
                                                              categorical_values=categorical_values[feature])

                transition_matrices[feature] = transition_matrix

    else:
        import multiprocessing

        categorical_features = [feature for feature, var_type in zip(features, var_types)
                                if var_type != "c" and feature not in exclude]

        # schedule in descending order of number of feature values,
        # which is approximately proportional to the required computation time
        num_of_values = [len(np.unique(samples[:, i])) for i, feature in enumerate(features)
                         if feature in categorical_features]
        cat_feature_idx_by_value = np.argsort(-np.array(num_of_values))
        assert num_of_values[cat_feature_idx_by_value[0]] == np.max(num_of_values)

        worklist = [(samples, features.index(categorical_features[cat_feature_idx]), pdf, the_pdf,
                     categorical_values[categorical_features[cat_feature_idx]], False)
                    for cat_feature_idx in cat_feature_idx_by_value]

        for _samples, _feature_idx, _, _, _categorical_values, _ in worklist:
            all_values = list(np.unique(_samples[:, _feature_idx]))
            if not np.all([val in _categorical_values for val in all_values]):
                for val in all_values:
                    print(f"Value {val} is not in {_categorical_values} "
                          f"for feature {features[_feature_idx]}")
                raise ValueError("Incomplete categorical values")

        with multiprocessing.Pool(num_processes) as pool:
            results = pool.starmap(compute_transition_matrix, worklist)

        for result_idx, cat_feature_idx in enumerate(cat_feature_idx_by_value):
            feature = categorical_features[cat_feature_idx]
            transition_matrices[feature] = results[result_idx]

    return transition_matrices
