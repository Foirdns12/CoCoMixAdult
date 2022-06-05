from typing import Dict, Any, Callable

import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from model_instances import immo_model, adult_model

#model = immo_model.load_model(code="20210222-161559")
model = adult_model.load_model()

def equal(a, b):
    return a == b


def off_by_two(a, b):
    return abs(a - b) == 2


def make_correctness(model: Model, preprocess: Callable[[Dict[str, Any]], Any],
                     comparison: Callable[[int, int], bool] = equal) -> Callable[[Dict[str, Any], Dict[str, Any]], int]:
    def correctness(fact: Dict[str, Any], foil: Dict[str, Any]) -> int:
        input_fact = preprocess(fact)
        input_foil = preprocess(foil)

        fact_class = int(np.argmax(model.predict(input_fact)[0]))
        foil_class = int(np.argmax(model.predict(input_foil)[0]))

        return int(comparison(fact_class, foil_class))

    return correctness

def immo_correctness(fact: Dict[str, Any], foil: Dict[str, Any], **_: Any) -> int:
    p_fact = {k: np.array([v]) for k,v in fact.items()}
    fact_class=np.argmax(model.predict(p_fact)[0])
    p_foil = {k: np.array([v]) for k,v in foil.items()}
    foil_class=np.argmax(model.predict(p_foil)[0])
    return_val=0
    if abs(fact_class - foil_class) == 2:
        return_val=1
    return return_val


def adult_correctness(fact: Dict[str, Any], foil: Dict[str, Any], **_: Any) -> int:
    p_fact = {k: np.array([v]) for k,v in fact.items()}
    p_fact = pd.DataFrame.from_dict(p_fact)
    fact_class=np.argmax(model.predict_proba(p_fact)[0])
    p_foil = {k: np.array([v]) for k,v in foil.items()}
    p_foil = pd.DataFrame.from_dict(p_foil)
    foil_class=np.argmax(model.predict_proba(p_foil)[0])
    return_val=0
    if abs(fact_class - foil_class) == 1:
        return_val=1
    return return_val

