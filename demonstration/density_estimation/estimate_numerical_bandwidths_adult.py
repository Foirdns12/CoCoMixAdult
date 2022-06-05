import numpy as np
from statsmodels.nonparametric.kde import KDEUnivariate

from demonstration.demonstration_data import load_data, FEATURES, VAR_TYPES


def estimate_bandwidth(column, bandwidths=None):
    bandwidths = bandwidths or ['scott', 'silverman', 'normal_reference']

    kde = KDEUnivariate(column)

    results = []

    for bw in bandwidths:
        kde.fit(bw=bw)
        results.append((kde.bw, kde.support, kde.density))

    return results


def plot_estimation_result(feature, column, results, plot_range=None):
    import matplotlib.pyplot as plt

    plot_range = plot_range or [np.min(column), np.max(column)]

    if np.all(column.astype(np.int) == column) and feature != "fnlwgt":
        n_bins = int(np.max(column) - np.min(column)) + 1
    else:
        n_bins = 10000

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.hist(column, bins=n_bins, label='Histogram from samples',
            zorder=5, edgecolor='k', density=True, alpha=0.5)

    for bw, support, density in results:
        if isinstance(bw, str):
            label = f'KDE from samples, bw = {bw}'
        else:
            label = f'KDE from samples, bw = {bw:0.5f}'
        ax.plot(support, density, '--', lw=2, zorder=10,
                label=label)
    ax.legend(loc='best')
    ax.set_xlim(plot_range)
    ax.grid(True, zorder=-5)

    plt.title(feature)
    plt.show()


if __name__ == "__main__":
    _plot_ranges = {
        "age": [17, 90],
        "fnlwgt": [1.e+04 , 1.5e+06],
        "education-num": [1, 16],
        "hours-per-week": [1, 99]
    }

    _bandwidths = {
        "age": [1,2,3,4],
        "fnlwgt": [2000,5000,8000,10000],
        "education-num": [0.5,0.8,1.0,1.2],
        "hours-per-week": [1,2,4,6]
    }

    samples, targets = load_data()

    for i, (feature, var_type) in enumerate(zip(FEATURES, VAR_TYPES)):
        if var_type == "c":
            column = samples[:, i].astype(np.float)
            column = column[~np.isnan(column)]

            results = estimate_bandwidth(column, bandwidths=_bandwidths.get(feature, None))
            plot_estimation_result(feature, column, results, plot_range=_plot_ranges[feature])
