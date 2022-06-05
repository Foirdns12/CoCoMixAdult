import random

import numpy as np
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .kde import _create_preprocessor


def test_data_shape_in_preprocessing():
    data = np.random.random_sample((2, 100))

    with pytest.raises(ValueError):
        _create_preprocessor(data, "cc")


def test_scalar_preprocessing():
    data = np.random.random_sample((100, 1))

    preprocess = _create_preprocessor(data, "c")

    exp_mean = np.mean(data)
    exp_std = np.std(data)

    transformed_data = preprocess(data)

    assert transformed_data.shape == data.shape
    assert np.all(np.equal((data - exp_mean) / exp_std, transformed_data))


def test_categorical_preprocessing():
    categories = ["alpha", "beta", "gamma"]
    data = np.random.choice(categories, (100, 1), replace=True)

    preprocess = _create_preprocessor(data, "o", [categories])

    transformed_data = preprocess(data)

    assert transformed_data.shape == data.shape
    assert np.all(np.equal(transformed_data, np.array([[categories.index(value)] for value in data])))


def test_mixed_preprocessing():
    categories1 = ["alpha", "beta", "gamma"]
    categories2 = ["nobody", "somebody", "anybody", "everybody"]

    cat_data1 = np.random.choice(categories1, (100, 1), replace=True)
    cat_data2 = np.random.choice(categories2, (100, 1), replace=True)
    scalar_data = 2 * np.random.random_sample((100, 1))

    data = np.hstack([scalar_data,cat_data1, cat_data2]).astype(object)
    assert data.shape == (100, 3)

    preprocess = _create_preprocessor(data, "cou", [categories1, categories2])

    transformed_data = preprocess(data)

    exp_cat_data1 = np.array([[categories1.index(value)] for value in cat_data1])
    exp_cat_data2 = np.array([[categories2.index(value)] for value in cat_data2])
    exp_scalar_data = (scalar_data - np.mean(scalar_data)) / np.std(scalar_data)
    exp_data = np.hstack([exp_cat_data1, exp_scalar_data, exp_cat_data2])
    assert exp_data.shape == (100, 3)

    assert transformed_data.shape == data.shape
    assert np.all(np.equal(exp_data, transformed_data))


def test_value_order():
    categories = ['no_information', 'simple', 'normal', 'sophisticated', 'luxury']
    data = np.random.choice(categories, (100, 1), replace=True)

    with pytest.raises(ValueError):
        enc = OrdinalEncoder(categories=[categories])
        enc.fit(data)

    data = data.astype(object)

    random.seed(1104)
    for _ in range(10):
        random.shuffle(categories)

        enc = OrdinalEncoder(categories=[categories])

        enc.fit(data)

        assert enc.categories == [categories]
        assert np.all(enc.categories_ == np.array([categories]))

        assert np.all(enc.transform(data) == np.array([[categories.index(val)]
                                                       for val in data.flatten()]))


def test_value_order_in_column_transformer():
    categories = ['no_information', 'simple', 'normal', 'sophisticated', 'luxury']
    data = np.random.choice(categories, (100, 1), replace=True)

    categorical_transformer = Pipeline(steps=[('ordinal', OrdinalEncoder(categories=[categories]))])

    column_transformer = ColumnTransformer(
        transformers=[("categorical", categorical_transformer, [0])]
    )

    with pytest.raises(ValueError):
        column_transformer.fit(data)

    data = data.astype(object)

    column_transformer.fit(data)

    enc = column_transformer.transformers_[0][1].steps[0][1]

    assert enc.categories == [categories]
    assert np.all(enc.categories_ == np.array([categories]))

    assert np.all(column_transformer.transform(data) == np.array([[categories.index(val)]
                                                                  for val in data.flatten()]))


def test_value_order_in_preprocessor():
    categories = ['no_information', 'simple', 'normal', 'sophisticated', 'luxury']
    data = np.random.choice(categories, (100, 1), replace=True).astype(object)

    preprocess = _create_preprocessor(data, "o", [categories])

    for i, val in enumerate(categories):
        assert preprocess([[val]]) == [[i]]

    assert np.all(preprocess(data) == np.array([[categories.index(val)]
                                                for val in data.flatten()]))
