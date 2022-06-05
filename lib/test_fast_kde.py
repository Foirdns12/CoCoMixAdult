import time

import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from data.immo import load_data
from lib.kde import initialize_kde
from lib.fast_kde import FastKDEMultivariate

features = [
    "obj_lotArea",
    "obj_yearConstructed",
    "obj_regio1",
    "obj_heatingType",
    "obj_interiorQual"
]

bw = [
    25.0,
    10.0,
    0.1,
    0.1,
    0.1
]

var_types = ["c", "c", "u", "u", "o"]


def test_equality_and_speed():
    samples, _ = load_data(features, fillna="median")
    samples = samples[:2000]

    categorical_values = [list(np.unique(samples[:, i])) for i, var_type in enumerate(var_types)
                          if var_type != "c"]

    start = time.time()
    kde, preprocess = initialize_kde(samples, var_types,
                                     categorical_values=categorical_values,
                                     bw=bw,
                                     _KDE=KDEMultivariate,
                                     _scale_numerical_columns=False)
    print(f"Initialized MultiVariateKDE in {time.time() - start:0.4f} s")

    start = time.time()
    fast_kde, fast_preprocess = initialize_kde(samples, var_types,
                                               categorical_values=categorical_values,
                                               bw=bw,
                                               _KDE=FastKDEMultivariate,
                                               _scale_numerical_columns=False)
    print(f"Initialized FastMultiVariateKDE in {time.time() - start:0.4f} s")

    start = time.time()
    pdf = kde.pdf()
    print(f"Computed MultiVariateKDE in {time.time() - start:0.4f} s")

    start = time.time()
    fast_pdf = fast_kde.pdf()
    print(f"Computed FastMultiVariateKDE in {time.time() - start:0.4f} s")

    assert np.all(pdf == fast_pdf)


def test_nan_handling():
    samples, _ = load_data(features, fillna=False)
    samples = samples[:2000]

    categorical_values = [list(np.unique(samples[:, i])) for i, var_type in enumerate(var_types)
                          if var_type != "c"]

    fast_kde, fast_preprocess = initialize_kde(samples, var_types,
                                               categorical_values=categorical_values,
                                               bw=bw,
                                               _KDE=FastKDEMultivariate,
                                               _scale_numerical_columns=False,
                                               nan_as_cache_mean=True)

    start = time.time()
    fast_pdf = fast_kde.pdf()
    print(f"Computed FastMultiVariateKDE in {time.time() - start:0.4f} s")

    print(fast_pdf)

    assert not np.any(np.isnan(fast_pdf))

