import os
import pickle

from demonstration.demonstration_data import ALL_CATEGORICAL_VALUES
from lib.transition_matrices import tune_transition_matrix

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, "raw_transition_matrices.pickle"), "rb") as f:
    transition_matrices = pickle.load(f)

for feature, values in ALL_CATEGORICAL_VALUES.items():
    try:
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)
    except KeyError:
        print("No transition matrix for", feature)

transition_matrices["obj_buildingType"] = tune_transition_matrix(transition_matrices["obj_buildingType"], 8, 0.1)
transition_matrices["obj_cellar"] = tune_transition_matrix(transition_matrices["obj_cellar"], 8, 0.1)
transition_matrices["obj_condition"] = tune_transition_matrix(transition_matrices["obj_condition"], 20, 0.1)
transition_matrices["obj_interiorQual"] = tune_transition_matrix(transition_matrices["obj_interiorQual"], 8, 0.2)
transition_matrices["obj_heatingType"] = tune_transition_matrix(transition_matrices["obj_heatingType"], 8, 0.1)
#transition_matrices["obj_newlyConst"] = tune_transition_matrix(transition_matrices["obj_newlyConst"], 8, 0.1)
transition_matrices["obj_regio1"] = tune_transition_matrix(transition_matrices["obj_regio1"], 8, 0.1)

for feature, values in ALL_CATEGORICAL_VALUES.items():
    try:
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)
    except KeyError:
        print("No transition matrix for", feature)

with open(os.path.join(PATH, "tuned_transition_matrices.pickle"), "wb") as f:
    pickle.dump(transition_matrices, f)

# Preis könnte hier helfen
