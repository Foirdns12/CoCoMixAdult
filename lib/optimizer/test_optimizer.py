from lib.optimizer.optimizer import Optimizer
from lib.optimizer.parametrization import Scalar, Parametrization


def test_simple_regression():
    def loss_fn(x):
        return x ** 2, {}

    parametrization = Parametrization(Scalar(10.0))

    optimizer = Optimizer(parametrization, mutation_rate=0.2)

    result, history = optimizer.minimize(loss_fn, budget=1000)

    print(history)

    assert abs(result[0]) < 0.01
