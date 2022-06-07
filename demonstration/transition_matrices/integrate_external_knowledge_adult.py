import os
import pickle
import numpy as np
from demonstration.demonstration_data import ALL_CATEGORICAL_VALUES

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, "tuned_transition_matrices_adult.pickle"), "rb") as f:
    transition_matrices = pickle.load(f)

for feature, values in ALL_CATEGORICAL_VALUES.items():
    try:
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)
    except KeyError:
        print("No transition matrix for", feature)

# Prevent transition to "no_information"

for feature, values in ALL_CATEGORICAL_VALUES.items():
    if feature not in transition_matrices:
        print(f"No transition matrix found for {feature}, skipping.")
        continue

    print(values)
    if "no_information" in values:
        idx = values.index("no_information")
        print("no_information index", idx)
        transition_matrices[feature][:idx, idx] = 0.0
        transition_matrices[feature][idx + 1:, idx] = 0.0

native_country = transition_matrices['native-country']
native_country[40,:]=1/40
print("")
for feature, values in ALL_CATEGORICAL_VALUES.items():
    print(feature)
    try:
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)
            assert np.sum(row) > 0.0
    except KeyError:
        print("No transition matrix for", feature)

print(transition_matrices)
with open(os.path.join(PATH, "final_transition_matrices_adult.pickle"), "wb") as f:
    pickle.dump(transition_matrices, f)
