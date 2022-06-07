from typing import Any, Callable, Dict, List, Optional

import numpy as np

from demonstration.demonstration_data import ALL_CATEGORICAL_VALUES, CATEGORICAL, NUMERICAL, load_df
from demonstration.transition_matrices.compute_distance_matrices_adult import compute_distance_matrix
from demonstration.transition_matrices.final_matrices import load_distance_matrices
from demonstration.transition_matrices.unit_matrices import get_unit_distance_matrices, get_unit_transition_matrices
import pandas as pd

EPSILON = 1e-12


def make_cat_distance(categorical_distance_fn: Callable[[Dict[str, Any], Dict[str, Any], float], float], beta=1.0) \
        -> Callable[[Dict[str, Any], Dict[str, Any], Any], float]:
    def distance(fact: Dict[str, Any], foil: Dict[str, Any], **kwargs: Any) -> float:
        cat_fact = {feature: value for feature, value in fact.items() if feature in CATEGORICAL}
        cat_foil = {feature: value for feature, value in foil.items() if feature in CATEGORICAL}
        return categorical_distance_fn(cat_fact, cat_foil, beta)

    return distance


def make_distance(categorical_distance_fn: Callable[[Dict[str, Any], Dict[str, Any], float], float],
                  continuous_distance_fn: Callable[[np.ndarray, np.ndarray], float], beta=1.0) \
        -> Callable[[Dict[str, Any], Dict[str, Any], Any], float]:
    def distance(fact: Dict[str, Any], foil: Dict[str, Any], **kwargs: Any) -> float:
        cat_fact = {feature: value for feature, value in fact.items() if feature in CATEGORICAL}
        cat_foil = {feature: value for feature, value in foil.items() if feature in CATEGORICAL}
        cont_fact = np.array([value for feature, value in fact.items() if feature in NUMERICAL])
        cont_foil = np.array([value for feature, value in foil.items() if feature in NUMERICAL])
        return categorical_distance_fn(cat_fact, cat_foil, beta) + np.sum(continuous_distance_fn(cont_fact, cont_foil))

    return distance


def _categorical_distance(distance_matrices: Dict[str, np.ndarray]) \
        -> Callable[[Dict[str, Any], Dict[str, Any], float], float]:
    def compute_distance(fact: Dict[str, Any], foil: Dict[str, Any], beta: float = 0.5) -> float:
        distances = []

        for feature, values in ALL_CATEGORICAL_VALUES.items():
            dist = distance_matrices[feature]
            values = ALL_CATEGORICAL_VALUES[feature]
            idx1 = values.index(fact[feature])
            idx2 = values.index(foil[feature])

            distances.append((beta / (EPSILON + dist[idx1][idx2]) - beta))

        return sum(distances)

    return compute_distance


naive_distance = _categorical_distance(get_unit_distance_matrices())
weighted_distance = _categorical_distance(load_distance_matrices())


def wachter_distance(mad: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
    def compute_distance(fact: np.ndarray, foil: np.ndarray) -> float:
        return np.abs(fact.astype(np.float) - foil.astype(np.float)) / mad

    return compute_distance


def euclidean_distance() -> Callable[[np.ndarray, np.ndarray], float]:
    def compute_distance(fact: np.ndarray, foil: np.ndarray) -> float:
        return np.sqrt(np.power(fact.astype(np.float) - foil.astype(np.float), 2))

    return compute_distance


def create_reference_points():
    train_df = load_df(train=True).sample(n=10)
    fact_df = train_df[NUMERICAL + CATEGORICAL]
    return fact_df.to_dict(orient="records")


def min_distance_to_nearest_datapoint(distance: Callable[[Dict[str, Any], Dict[str, Any], Optional[Any]], float],
                                      reference_points: List[Dict[str, Any]]) \
        -> Callable[[Dict[str, Any], Dict[str, Any], Optional[Any]], float]:
    def min_distance(fact: Dict[str, Any], foil: Dict[str, Any], **kwargs: Any) -> float:
        print('start calc')
        print(reference_points)
        return min((distance(point, foil, **kwargs) for point in reference_points))

    return min_distance



if __name__ == "__main__":

    reference_points = create_reference_points()

    md = min_distance_to_nearest_datapoint(make_distance(categorical_distance_fn=naive_distance,
                                                         continuous_distance_fn=euclidean_distance()),
                                           reference_points=reference_points)

    assert md(reference_points[10], reference_points[11]) == 0.0


