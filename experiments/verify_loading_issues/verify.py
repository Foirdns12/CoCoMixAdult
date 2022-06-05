import os

import pickle
import numpy as np
from demonstration.demonstration_immo_model import load_model, load_model_from_weights, inspect_confusion_matrix

PATH = os.path.dirname(os.path.abspath(__file__))

code = "20200521-150237"

m_full = load_model(code=code)
m_weights = load_model_from_weights(code=code)

val_cm_full = inspect_confusion_matrix(model=m_full)
val_cm_weights = inspect_confusion_matrix(model=m_weights)

#assert np.allclose(val_cm_full, val_cm_weights)


with open(os.path.join(PATH, "matrixloadval.pickle"), "rb") as f:
    ph_val_cm_full = pickle.load(f)

with open(os.path.join(PATH, "matrixloadfromweightsval.pickle"), "rb") as f:
    ph_val_cm_weights = pickle.load(f)

print("FULL")
print(val_cm_full)
print(ph_val_cm_full)


print("WEIGHTS")
print(val_cm_weights)
print(ph_val_cm_weights)

assert np.allclose(val_cm_full, ph_val_cm_full)
assert np.allclose(val_cm_weights, ph_val_cm_weights)
