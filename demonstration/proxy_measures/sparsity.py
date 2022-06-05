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
        # das ist die kategorische sparsity im vergleich zur gesamten sparsity zur einstellung,
        # dass sich das ca im verhältnis der variablen hält. Wenn die normale sparsity null ist,
        # teilen wir aber durch null, deshalb habe ich den wert einfach auf 0.5 gesetzt für 0
        # (wenn die sparsity gesamt 0 ist, ist ja sowieso die kat sparsity null und der gesamte
        # wert null)
        catspars = sum(
            (int(fact[feature] != foil[feature]) for (feature, var_type) in itertools.zip_longest(features, var_types)
             if var_type != 'c'))

        if _sparsity == 0:
            return 0.5
        else:
            return catspars / _sparsity

    return categorical_sparsity
