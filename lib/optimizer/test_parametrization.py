import pytest
import numpy as np

from lib.optimizer.parametrization import Constant, Scalar, Categorical, Parametrization


def test_constant():
    c = Constant(5.0)

    assert c.value == 5.0
    assert c.dimension == 0

    c.value = 8.0
    assert c.value == 8.0

    c.freeze()
    with pytest.raises(PermissionError):
        c.value = 10.0

    d = c.clone()

    assert d.value == 8.0

    with pytest.raises(PermissionError):
        d.value = 10.0


def test_scalar_basics():
    s = Scalar(1.0)

    assert s.value == 1.0
    assert s.dimension == 1

    s.value = 4.3
    assert s.value == 4.3

    s.freeze()
    with pytest.raises(PermissionError):
        s.value = 0.12

    t = s.clone()
    assert t.value == 4.3

    t.mutate(bits_to_mutate=[False])
    assert t.value == 4.3

    t.mutate(bits_to_mutate=[True])
    assert t.value != 4.3
    assert s.value == 4.3


def test_scalar_bounding():
    s = Scalar(1.0, sigma=10.0)

    s.set_bounds(0.0, 4.0)
    assert s.value == 1.0

    s.value = 8.0
    assert s.value == 4.0

    s.value = -12.2
    assert s.value == 0.0

    for _ in range(50):
        s.mutate(bits_to_mutate=[True])
        assert 0.0 <= s.value <= 4.0


def test_scalar_integer_casting():
    s = Scalar(3.7)

    assert s.value == 3.7

    s.cast_to_integer = True
    assert s.value == 4

    s.value = 3.4
    assert s.value == 3


def test_categorical():
    c = Categorical(choices=["first", "second"],
                    transition_matrix=np.array([[0.0, 1.0], [0.0, 1.0]]))

    assert c.value == "first"
    assert c.dimension == 1

    c.value = "second"
    assert c.value == "second"

    for _ in range(50):
        c.mutate(bits_to_mutate=[True])
        assert c.value == "second"

    c.value = "first"
    c.mutate(bits_to_mutate=[True])
    assert c.value == "second"

    d = c.clone()
    assert d.value == "second"


def test_parametrization():
    c = Categorical(choices=["first", "second"],
                    transition_matrix=np.array([[0.0, 1.0], [0.0, 1.0]]))
    s = Scalar(2.0)

    p = Parametrization(c, s)

    assert p.dimension == 2

    q = p.clone()
    assert q.dimension == 2

    q.mutate(bits_to_mutate=[True, False])
    assert q.value[0] == "second"
    assert q.value[1] == 2.0

    assert p.value[0] == "first"
    assert p.value[1] == 2.0
