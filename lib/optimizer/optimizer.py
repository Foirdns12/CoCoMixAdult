from typing import Callable, List, Tuple, Dict, Union
from lib.optimizer.parametrization import Parametrization
from demonstration.density_estimation.compute_kde import kde, preprocess
import pandas as pd

import numpy as np


class Optimizer:

    def __init__(self, parametrization: Parametrization,
                 mutation_rate: Union[float, None] = None,
                 mutation_rule: Union[str, None] = None):
        """Simple 1+1 optimizer.

        :param parametrization: The parameters, set to their initial values, as a `Parametrization` object.
        :param mutation_rate: The average fraction of parameters to mutate when generating a new candidate.
        :param mutation_rule: Name of a pre-defined mutation rule, e.g. '1/n' or '2/n', where n is the number
                              of mutable parameters of the *parametrization*.
        """
        if parametrization.dimension < 1:
            raise ValueError("Static parametrization, no parameters to mutate.")

        self.parametrization = parametrization
        self.parametrization.freeze()

        if mutation_rate is None and mutation_rule is None:
            raise ValueError("Either mutation_rate or mutation_rule has to be set.")

        if mutation_rule is not None:
            if mutation_rate is not None:
                raise ValueError("Only either mutation_rate or mutation_rule must be given.")

            if mutation_rule == "1/n":
                mutation_rate = 1/parametrization.dimension
            elif mutation_rule == "2/n":
                mutation_rate = 2/parametrization.dimension
            else:
                raise ValueError(f"Unknown mutation rule '{mutation_rule}'.")

        if not 0.0 < mutation_rate < 1.0:
            raise ValueError("Mutation rate has to be greater than 0 and less than 1.")

        self.mutation_rate = mutation_rate

    def minimize(self, decision_function: Callable, budget: int) -> Tuple[List, Dict[str, List]]:
        minimal_loss, log = decision_function(*self.parametrization.value)

        history = {k: [v] for k, v in log.items()}
        history.update({"step": [-1], "loss": [minimal_loss], "value": [self.parametrization.value], "pdf": [pdf(pd.DataFrame(self.parametrization.value).to_numpy()[:, 0])]})

        for step in range(budget):
            candidate = self.parametrization.clone()

            bits_to_mutate = self.sample(candidate.dimension)
            candidate.mutate(bits_to_mutate)
            candidate.freeze()

            candidate_loss, log = decision_function(*candidate.value)

            if candidate_loss < minimal_loss:
                self.parametrization = candidate

                minimal_loss = candidate_loss
                history["step"].append(step)
                history["loss"].append(minimal_loss)
                history["value"].append(candidate.value)
                history["pdf"].append(pdf(pd.DataFrame(candidate.value).to_numpy()[:, 0]))
                for k, v in log.items():
                    history[k].append(v)

        return self.parametrization.value, history

    def sample(self, length: int) -> List[bool]:
        bits_to_mutate = np.random.random_sample(length) < self.mutation_rate
        if np.all(~bits_to_mutate):
            # Flip at least one bit
            bits_to_mutate[np.random.randint(0, length)] = True
        return list(bits_to_mutate)


def pdf(sample):
    #print(sample)
    return kde.pdf(preprocess(sample.reshape(1, -1)))