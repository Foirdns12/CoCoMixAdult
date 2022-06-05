import random
import numpy as np

from lib.cache_dict import CacheDict


def test_that_items_can_be_stored():
    d = CacheDict()

    d["itop"] = np.array([1, 2, 3])

    assert np.all(d["itop"] == np.array([1, 2, 3]))


def test_that_mean_is_computed():
    d = CacheDict()

    vals = []
    for i in range(10):
        val = np.random.random(20)
        d[i] = val
        vals.append(val)

    vals = np.array(vals).flatten()

    np.testing.assert_almost_equal(d.mean, np.mean(vals))
    assert d.size == 200
    np.testing.assert_almost_equal(d.sum, np.sum(vals))
    assert len(d) == 10


def test_that_nans_are_not_counted():
    d = CacheDict()

    vals = []
    for i in range(40):
        val = np.random.random(10)
        d[i] = val
        vals.append(val)

    vals = np.array(vals).flatten()

    for i in range(10):
        d[i + 40] = np.full(10, np.nan)

    np.testing.assert_almost_equal(d.mean, np.mean(vals))
    assert d.size == 400
    np.testing.assert_almost_equal(d.sum, np.sum(vals))
    assert len(d) == 50


def test_that_partial_nans_are_handled():
    d = CacheDict()

    vals = []
    for i in range(40):
        val = np.random.random(10)
        val[random.randint(0, 9)] = np.nan
        d[i] = val
        vals.append(val[~np.isnan(val)])

    vals = np.array(vals).flatten()

    for i in range(10):
        d[i + 40] = np.full(10, np.nan)

    np.testing.assert_almost_equal(d.mean, np.mean(vals))
    assert d.size == 360
    np.testing.assert_almost_equal(d.sum, np.sum(vals))
    assert len(d) == 50
