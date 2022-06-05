from typing import Union, List

import numpy as np


def protected(method):
    """Decorator for Parameter methods that raises an error if the Parameter is frozen."""
    def wrapped(self, *args, **kwargs):
        if self.frozen:
            raise PermissionError("Parameter is frozen")
        else:
            return method(self, *args, **kwargs)

    return wrapped


class Parameter:

    def __init__(self):
        self.frozen = False

    def freeze(self):
        """Protect value from changing through mutation or direct assignment."""
        self.frozen = True

    @property
    def value(self):
        raise NotImplementedError

    @value.setter
    @protected
    def value(self, value):
        raise NotImplementedError

    @property
    def dimension(self):
        """Number of mutable parameters."""
        raise NotImplementedError

    @protected
    def mutate(self, bits_to_mutate: List[bool]):
        """Mutate the value.

        :param bits_to_mutate: List of size `dimension` that determines which parameter
                               should be mutated.
        """
        raise NotImplementedError

    def clone(self):
        child = self.__class__()
        return child


class Constant(Parameter):

    def __init__(self, value: Union[float, int, str]):
        super(Constant, self).__init__()
        self._value = value

    @property
    def value(self):
        return self._value

    @value.setter
    @protected
    def value(self, value):
        self._value = value

    @property
    def dimension(self):
        return 0

    def mutate(self, bits_to_mutate: List[bool]):
        raise AttributeError("Cannot mutate Constant")

    def clone(self):
        child = self.__class__(self.value)
        child.freeze()
        return child


class Scalar(Parameter):

    def __init__(self, initial_value: Union[float, int] = 0.0, sigma: float = 1.0):
        super(Scalar, self).__init__()

        self._value = initial_value
        self._bounds = (-np.inf, np.inf)
        self._cast_to_integer = False
        self.sigma = sigma

    @property
    def cast_to_integer(self):
        return self._cast_to_integer

    @cast_to_integer.setter
    @protected
    def cast_to_integer(self, value: bool):
        self._cast_to_integer = value

    @property
    def value(self):
        if self.cast_to_integer:
            return int(np.round(self._value))
        else:
            return self._value

    @value.setter
    @protected
    def value(self, value):
        value = self._constrain(value)
        self._value = value

    @property
    def dimension(self):
        return 1

    def set_bounds(self, lower_bound=None, upper_bound=None):
        if lower_bound is not None and upper_bound is not None:
            if lower_bound < upper_bound:
                self._bounds = (lower_bound, upper_bound)
            else:
                raise ValueError(f"Lower bound {lower_bound} must be strictly smaller than upper bound {upper_bound}.")

        if lower_bound is None:
            if upper_bound > self._bounds[0]:
                self._bounds = (self._bounds[0], upper_bound)
            else:
                raise ValueError(f"Upper bound {upper_bound} must be strictly larger than lower bound {lower_bound}.")

        if upper_bound is None:
            if lower_bound < self._bounds[1]:
                self._bounds = (lower_bound, self._bounds[1])
            else:
                raise ValueError(f"Lower bound {lower_bound} must be strictly smaller than upper bound {upper_bound}.")

        self.value = self._constrain(self.value)

    def _constrain(self, value):
        if self._bounds[0] <= value <= self._bounds[1]:
            return value
        elif value < self._bounds[0]:
            return self._bounds[0]
        else:
            return self._bounds[1]

    @protected
    def mutate(self, bits_to_mutate):
        if bits_to_mutate[0]:
            new_value = np.random.normal(loc=self.value, scale=self.sigma, size=1)
            self.value = float(new_value)

    def clone(self):
        child = self.__class__(self.value, self.sigma)
        child._bounds = self._bounds
        child.cast_to_integer = self.cast_to_integer
        return child


class Categorical(Parameter):

    def __init__(self, choices: list, transition_matrix: np.ndarray):
        super(Categorical, self).__init__()

        self.choices = choices
        self.transition_matrix = transition_matrix

        self._index = 0

    @property
    def value(self):
        return self.choices[self._index]

    @value.setter
    @protected
    def value(self, value):
        if value not in self.choices:
            raise ValueError(f"Unknown value {value}.")

        self._index = self.choices.index(value)

    @property
    def dimension(self):
        return 1

    @protected
    def mutate(self, bits_to_mutate):
        if bits_to_mutate[0]:
            probas = np.array(self.transition_matrix[self._index], dtype=float)
            probas /= np.sum(probas)

            new_index = np.random.choice(list(range(probas.size)), p=probas)
            self._index = int(new_index)

    def clone(self):
        child = self.__class__(self.choices, self.transition_matrix)
        child.value = self.value
        return child


class Parametrization(Parameter):

    def __init__(self, *parameters: Parameter):
        super(Parametrization, self).__init__()

        self._parameters = parameters

    def freeze(self):
        super().freeze()
        for p in self._parameters:
            p.freeze()

    @property
    def value(self):
        return [p.value for p in self._parameters]

    @property
    def dimension(self):
        return sum((p.dimension for p in self._parameters))

    @protected
    def mutate(self, bits_to_mutate):
        if len(bits_to_mutate) != self.dimension:
            raise ValueError("Bits to mutate must be exactly of length dimension.")

        bit_cnt = 0
        for p in self._parameters:
            if p.dimension < 1:
                continue

            p.mutate(bits_to_mutate[bit_cnt:bit_cnt + p.dimension])
            bit_cnt += p.dimension

        assert bit_cnt == self.dimension

    def clone(self):
        child_parameters = [p.clone() for p in self._parameters]
        child = self.__class__(*child_parameters)
        return child
