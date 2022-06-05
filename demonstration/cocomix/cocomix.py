from typing import List, Callable, Tuple, Union, Dict

import numpy as np
import pandas as pd
import tensorflow as tf
from model_instances import adult_model
EPSILON = 1e-30

model = adult_model.load_model()
def make_decision_function(fact: np.ndarray,
                           target_class: int,
                           input_features: List[str],
                           var_types: List[str],
                           categorical_values: Dict[str, List[str]],
                           #model: tf.keras.models.Model,
                           model: model,
                           mad: np.ndarray,
                           pdf: Union[Callable[[np.ndarray], float], None],
                           distance_matrices: Dict[str, np.ndarray],
                           lambda_: float, mu: float, alpha: float, beta: float,
                           densitycut: float, densityaddloss: float, densityscaler: float) \
        -> Callable[[Tuple[Union[str, int, float]], bool], Tuple[float, Dict[str, float]]]:
    """

    :param fact: The fact to find the foil for.
    :param target_class: The target class of the foil.
    :param input_features: The names of the input features.
    :param var_types: The var_types of the input features.
    :param categorical_values: The possible values of the categorical variables.
    :param model: The model (algorithm assumes model that uses named feature_columns as input)
    :param mad: The MAD for all numerical variables
    :param pdf: A function that takes a single datapoint as the input and returns the (estimated) PDF.
    :param distance_matrices: Pre-computed matrices of distances between categorical variables
    :param lambda_: The pre-factor of the prediction term.
    :param mu: The pre-factor of the distance term.
    :param alpha: The pre-factor of the density loss term.
    :param beta: The pre-factor and constant of the categorical distance loss.
    :return:
    """
    #aenderungen
    p_fact = {feature: np.array([value]) for feature, value in zip(input_features, fact)}
    #p_fact = pd.DataFrame(fact)

    p_fact = pd.DataFrame.from_dict(p_fact)
    fact_prediction = model.predict(p_fact)[0]
    #num_classes =
    num_classes = 2
    target = np.zeros(num_classes)
    target[target_class] = 1.0

    feature_is_numerical = np.array(var_types) == "c"

    refvalue = pdf(np.array(fact))
    refscaled = refvalue ** 0.05
    refvalfin = (1 / (EPSILON + refscaled))
    correction = (densitycut / refvalfin) ** densityaddloss

    def compute_distance(feature: str, val1: str, val2: str) -> float:
        if val1 == val2:
            return 1.0  # so that distance loss is zero if values are equal (beta * 1.0 - beta)

        dist = distance_matrices[feature]
        values = categorical_values[feature]

        idx1 = values.index(val1)
        idx2 = values.index(val2)

        return 1 / (EPSILON + dist[idx1][idx2])

    def decision_function(*args: Union[str, int, float], print_: bool = False) -> Tuple[float, Dict[str, float]]:
        # prediction loss
        #aenderungen
        p_sample = {feature: np.array([value]) for feature, value in zip(input_features, args)}
        p_sample = pd.DataFrame.from_dict(p_sample)
        _prediction = model.predict_proba(p_sample)[0]
        prediction_loss = lambda_ * np.sqrt(np.sum(np.power(_prediction - target, 2)))

        sample = pd.DataFrame(args).to_numpy()[:, 0]
        # distance loss
        if mu > 0.0:
            distances = np.zeros_like(sample, dtype=np.float)

            # numerical distance term
            distances[feature_is_numerical] = np.abs(
                sample[feature_is_numerical].astype(np.float) - fact[feature_is_numerical].astype(np.float)) / mad

            # categorical distance term
            categorical_distances = np.array([compute_distance(feature, val1, val2) if var_type != "c" else None
                                              for feature, var_type, val1, val2
                                              in zip(input_features, var_types, fact, sample)], dtype=np.float)
            distances[~feature_is_numerical] = beta * categorical_distances[~feature_is_numerical] - beta
            distance_loss = mu * np.sum(distances)
        else:
            distance_loss = 0.0

        # density loss
        if alpha > 0.0 and pdf is not None:

            if refvalfin <= densitycut:
                densityscaled = pdf(sample) ** (densityscaler)
                density_loss = alpha * (1 / (EPSILON + densityscaled))
            else:
                densityscaled = pdf(sample) ** (densityscaler)
                density_loss = alpha * correction * (1 / (EPSILON + densityscaled))
        else:
            density_loss = 0.0

        if print_:
            print(f"Losses:\n"
                  f"- prediction {prediction_loss}\n"
                  f"- distance {distance_loss}\n"
                  f"- density: {density_loss}\n"
                  f"= total {prediction_loss + distance_loss + density_loss}")
            print(f"Prediction:\n"
                  f"- class {np.argmax(_prediction)} with probability {np.max(_prediction)}\n"
                  f"- target class {target_class} with probability {_prediction[target_class]}")

        full_loss = prediction_loss + distance_loss + density_loss

        log = {
            "full_loss": full_loss,
            "prediction_loss": prediction_loss,
            "distance_loss": distance_loss,
            "density_loss": density_loss,
            "predicted_class": np.argmax(_prediction),
            "predicted_class_confidence": np.max(_prediction),
            "target_confidence": float(_prediction[target_class]),
            "prediction": _prediction
        }

        return full_loss, log

    return decision_function
