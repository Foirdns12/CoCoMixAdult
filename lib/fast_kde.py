"""Multivariate KDE optimized for computational performance."""

import numpy as np
from statsmodels.nonparametric.kernel_density import KDEMultivariate, kernels

from lib.cache_dict import CacheDict

kernel_func = {
    "c": kernels.gaussian,
    "o": kernels.wang_ryzin,
    "u": kernels.aitchison_aitken
}


class FastKDEMultivariate(KDEMultivariate):

    def __init__(self, data, var_type, bw=None, defaults=None, nan_as_cache_mean=False):
        super(FastKDEMultivariate, self).__init__(data, var_type, bw, defaults)
        self.nan_as_cache_mean = nan_as_cache_mean

        self.kernel_funcs = [kernel_func[vtype] for vtype in var_type]
        self.cache = [CacheDict() for _ in range(len(self.var_type))]
        self._cache_bw = tuple(bw)
        self.rebuild_cache()

    def pdf(self, data_predict=None, bw=None):
        if data_predict is None:
            data_predict = data_predict or self.data

        if data_predict.shape[1] != self.k_vars:
            raise ValueError(
                f"Provided data has {data_predict.shape[0]} variables, expected {self.k_vars}.")

        bw = bw or self.bw

        if self._cache_bw != tuple(bw):
            self.rebuild_cache(bw)

        pdf_est = []
        for i in range(np.shape(data_predict)[0]):
            pdf_est.append(gpke(bw, data=self.data,
                                data_predict=data_predict[i, :],
                                var_type=self.var_type,
                                kernel_func=self.compute_kernel) / self.nobs)

        pdf_est = np.squeeze(pdf_est)
        return pdf_est

    def rebuild_cache(self, bw=None):
        bw = bw or self.bw
        self.cache = [CacheDict() for _ in range(len(self.var_type))]
        self.kernel_funcs = [kernel_func[vtype] for vtype in self.var_type]

        for i, vtype in enumerate(self.var_type):
            if vtype == "c":
                integer_rate = np.sum(self.data[:, i].astype(int) == self.data[:, i].astype(float)) / len(
                    self.data[:, i])
                if integer_rate < 0.95:
                    self.cache[i] = None

        self._cache_bw = tuple(bw)

    def compute_kernel(self, index, h, Xi, x):
        cache = self.cache[index]

        if self.nan_as_cache_mean and np.isnan(x):
            if cache:
                return cache.mean
            else:
                # TODO: How to deal with this case?
                return 0.1

        if self.var_type[index] == "c":
            if cache is not None:
                kernel_value = _numerical_kernel(h, Xi, x, self.kernel_funcs[index], self.cache[index])
            else:
                kernel_value = self.kernel_funcs[index](h, Xi, x)
        else:
            kernel_value = _categorical_kernel(h, Xi, x, self.kernel_funcs[index], self.cache[index])

        if self.nan_as_cache_mean:
            if cache:
                kernel_value[np.isnan(kernel_value)] = cache.mean
            else:
                # TODO: How to deal with this case?
                kernel_value[np.isnan(kernel_value)] = 0.0

        return kernel_value


def _categorical_kernel(h, Xi, x, func, cache):
    try:
        return cache[x]
    except KeyError:
        val = func(h, Xi, x)
        cache[x] = val
        return val


def _numerical_kernel(h, Xi, x, func, cache):
    if int(x) == x:
        try:
            return cache[x]
        except KeyError:
            val = func(h, Xi, x)
            cache[x] = val
            return val
    else:
        return func(h, Xi, x)


def gpke(bw, data, data_predict, var_type, kernel_func):
    Kval = np.empty(data.shape)
    for ii, vtype in enumerate(var_type):
        Kval[:, ii] = kernel_func(ii, bw[ii], data[:, ii], data_predict[ii])

    dens = Kval.prod(axis=1)

    is_continuous = np.array([c == 'c' for c in var_type])
    if np.any(is_continuous):
        dens = dens / np.prod(bw[is_continuous])

    return dens.sum(axis=0)
