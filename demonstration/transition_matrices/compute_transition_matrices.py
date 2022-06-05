import os
import pickle

import numpy as np

from demonstration.demonstration_data import load_data, FEATURES, VAR_TYPES, ALL_CATEGORICAL_VALUES
from demonstration.density_estimation.compute_kde import kde, preprocess
from lib.transition_matrices import estimate_transition_matrices

PATH = os.path.dirname(os.path.abspath(__file__))

samples, _ = load_data(train=True)

if __name__ == '__main__':
    # preprocess categorical values
    longest_list = max((len(val) for val in ALL_CATEGORICAL_VALUES.values()))
    fake_samples = samples[:longest_list].copy()

    for feature_idx, feature in enumerate(FEATURES):
        if feature in ALL_CATEGORICAL_VALUES:
            col_value = [val for val in ALL_CATEGORICAL_VALUES[feature]]
            while len(col_value) < longest_list:
                col_value += [val for val in ALL_CATEGORICAL_VALUES[feature]]
            col_value = np.array(col_value[:longest_list])
            fake_samples[:, feature_idx] = col_value

    preprocessed_samples = preprocess(fake_samples)

    categorical_values = {}
    for feature_idx, feature in enumerate(FEATURES):
        if feature in ALL_CATEGORICAL_VALUES:
            categorical_values[feature] = list(preprocessed_samples[:, feature_idx][:len(ALL_CATEGORICAL_VALUES[feature])])

    for feature, p_values in categorical_values.items():
        print(feature)
        o_values = ALL_CATEGORICAL_VALUES[feature]
        print(len(o_values), len(p_values))
        assert len(o_values) == len(p_values)
        for o_val, p_val in zip(o_values, p_values):
            print("-", o_val, p_val)

    transition_matrices = estimate_transition_matrices(samples=preprocess(samples),
                                                       pdf=kde.pdf,
                                                       features=FEATURES,
                                                       var_types=VAR_TYPES,
                                                       num_processes=6,
                                                       categorical_values=categorical_values,
                                                       exclude=["geo_krs"])

    with open(os.path.join(PATH, "raw_transition_matrices.pickle"), "wb") as f:
        pickle.dump(transition_matrices, f)
