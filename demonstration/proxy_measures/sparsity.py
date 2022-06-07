from typing import Dict, Any, List, Callable
import itertools


def make_sparsity(features: List[str]) -> Callable[[Dict[str, Any], Dict[str, Any]], int]:

    def sparsity(fact: Dict[str, Any], foil: Dict[str, Any], **_: Any) -> int:
        return sum((int(fact[feature] != foil[feature]) for feature in features))

    return sparsity


def make_categorical_sparsity(features: List[str],
                              var_types: List[str]) -> Callable[[Dict[str, Any], Dict[str, Any]], float]:

    sparsity = make_sparsity(features)

    def categorical_sparsity(fact: Dict[str, Any], foil: Dict[str, Any], **_: Any) -> float:
        _sparsity = sparsity(fact, foil)
        catspars = sum(
            (int(fact[feature] != foil[feature]) for (feature, var_type) in itertools.zip_longest(features, var_types)
             if var_type != 'c'))

        if _sparsity == 0:
            return 0.5
        else:
            return catspars / _sparsity

    return categorical_sparsity
