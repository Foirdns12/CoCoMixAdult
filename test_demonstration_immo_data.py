import numpy as np

from data import immodata
from demonstration.demonstration_data import load_data, TARGET, load_df


def nan_equal(a, b):
    """Since NaN == NaN is always false, we need to test equality
       of arrays in a more sophisticated way.

       See https://stackoverflow.com/questions/10710328/comparing-numpy-arrays-containing-nan
       for details.
   """
    return np.all(((a == b) | ((a != a) & (b != b))))


def test_nan_equal():
    a = np.array([1.0, 2.0, None])
    b = np.array([1.0, 2.0, None])
    c = np.array([3.0, 2.0, None])
    d = np.array([5.0, 4.0, 2.0])
    e = np.array([5.0, 4.0, 2.0])
    f = np.array([2.0, 4.5, 10.0])

    assert nan_equal(a, b)
    assert not nan_equal(a, c)
    assert nan_equal(d, e)
    assert not nan_equal(b, d)
    assert not nan_equal(e, f)


def test_that_load_immo_is_reproducible():
    samples1, targets1 = immodata.load_data(fillna=False)
    samples2, targets2 = immodata.load_data(fillna=False)

    assert nan_equal(samples1, samples2)
    assert nan_equal(targets1, targets2)


def test_that_both_loading_approaches_are_identical():
    samples, targets = load_data(train=True)
    df = load_df(train=True)

    assert nan_equal(df.drop(columns=[TARGET]).to_numpy(), samples)
    assert nan_equal(df[TARGET].to_numpy(), targets)

    samples, targets = load_data(train=False)
    df = load_df(train=False)

    assert nan_equal(df.drop(columns=[TARGET]).to_numpy(), samples)
    assert nan_equal(df[TARGET].to_numpy(), targets)

