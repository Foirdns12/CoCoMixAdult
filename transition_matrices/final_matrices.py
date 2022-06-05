import os
import pickle

PATH = os.path.dirname(os.path.abspath(__file__))


def load_transition_matrices():

    with open(os.path.join(PATH, "final_transition_matrices_adult.pickle"), "rb") as f:
        transition_matrices = pickle.load(f)

    return transition_matrices


def load_distance_matrices():
    with open(os.path.join(PATH, "final_distance_matrices_adult.pickle"), "rb") as f:
        distance_matrices = pickle.load(f)

    return distance_matrices
