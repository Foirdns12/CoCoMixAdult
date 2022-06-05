import numpy as np
from demonstration.demonstration_immo_model import load_model, load_model_from_weights, inspect_confusion_matrix

code = "20210214-133037"
# code = "20200519-164205"


def test_load_model():
    m = load_model(code=code)
    inspect_confusion_matrix(model=m)
    print(m.summary())


def test_load_model_from_weights():
    m = load_model_from_weights(code=code, layer_conf=[(0.0, 256), (0.1, 256)])
    inspect_confusion_matrix(model=m)
    print(m.summary())


def test_confusion_matrices_are_identical():
    m_full = load_model(code=code)
    m_weights = load_model_from_weights(code=code)

    val_cm_full = inspect_confusion_matrix(model=m_full)
    val_cm_weights = inspect_confusion_matrix(model=m_weights)

    assert np.allclose(val_cm_full, val_cm_weights)
