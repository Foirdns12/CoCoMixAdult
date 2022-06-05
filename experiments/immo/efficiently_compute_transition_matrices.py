import os
import pickle

import numpy as np

from data.immodata import load_data
from lib.kde import initialize_kde
from lib.transition_matrices import estimate_transition_matrices

PATH = os.path.dirname(os.path.abspath(__file__))

samples, targets, features = load_data(fillna=True)

var_types = np.array(["u", "u", "c", "c", "c", "u", "c", "u", "u", "u", "c", "u", "u", "u"])
bw = [0.001, 0.001, 0.023, 0.11, 0.4, 0.001, 0.0128, 1, 0.001, 0.001, 0.152, 0.001, 0.001, 0.001]
assert len(var_types) == len(features)

samples = samples[:10000]

categorical_values = [list(np.unique(samples[:, idx])) for idx, _ in enumerate(features)
                      if var_types[idx] != "c"]

kde, preprocess = initialize_kde(samples, var_types, categorical_values, bw)

if __name__ == '__main__':
    transition_matrices = estimate_transition_matrices(samples=preprocess(samples),
                                                       pdf=kde.pdf,
                                                       features=features,
                                                       var_types=var_types,
                                                       num_processes=4)

    with open(os.path.join(PATH, "transition_matrices.pickle"), "wb") as f:
        pickle.dump(transition_matrices, f)
