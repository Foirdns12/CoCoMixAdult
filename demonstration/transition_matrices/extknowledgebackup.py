import os
import pickle

from demonstration.demonstration_data import ALL_CATEGORICAL_VALUES

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, "tuned_transition_matrices.pickle"), "rb") as f:
    transition_matrices = pickle.load(f)

for feature, values in ALL_CATEGORICAL_VALUES.items():
     for row, value in zip(transition_matrices[feature], values):
            print(value, row)

# Prevent transition to "no_information"

for feature, values in ALL_CATEGORICAL_VALUES.items():
    if feature not in transition_matrices:
        print(f"No transition matrix found for {feature}, skipping.")
        continue
    print(values)


for feature, values in ALL_CATEGORICAL_VALUES.items():
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)

