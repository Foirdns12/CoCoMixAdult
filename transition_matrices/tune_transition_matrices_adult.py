import os
import pickle

from demonstration.demonstration_data import ALL_CATEGORICAL_VALUES
from lib.transition_matrices import tune_transition_matrix

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, "raw_transition_matrices_adult.pickle"), "rb") as f:
    transition_matrices = pickle.load(f)

for feature, values in ALL_CATEGORICAL_VALUES.items():
    try:
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)
    except KeyError:
        print("No transition matrix for", feature)

transition_matrices["workclass"] = tune_transition_matrix(transition_matrices["workclass"], 8, 0.1)
transition_matrices["education"] = tune_transition_matrix(transition_matrices["education"], 8, 0.1)
transition_matrices["marital-status"] = tune_transition_matrix(transition_matrices["marital-status"], 8, 0.1)
transition_matrices["occupation"] = tune_transition_matrix(transition_matrices["occupation"], 8, 0.1)
transition_matrices["relationship"] = tune_transition_matrix(transition_matrices["relationship"], 8, 0.1)
transition_matrices["race"] = tune_transition_matrix(transition_matrices["race"], 8, 0.2)
transition_matrices["sex"] = tune_transition_matrix(transition_matrices["sex"], 8, 0.08)
transition_matrices["native-country"] = tune_transition_matrix(transition_matrices["native-country"], 8, 0.1)



for feature, values in ALL_CATEGORICAL_VALUES.items():
    try:
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)
    except KeyError:
        print("No transition matrix for", feature)

with open(os.path.join(PATH, "tuned_transition_matrices_adult.pickle"), "wb") as f:
    pickle.dump(transition_matrices, f)


