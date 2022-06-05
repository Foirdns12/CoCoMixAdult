import numpy as np

from demonstration.demonstration_data import load_data, FEATURES, VAR_TYPES, ORDERED_CATEGORICAL_VALUES
from demonstration.density_estimation.compute_kde import initialize_kde
from lib.fast_kde import FastKDEMultivariate


def estimate_bandwidth(column, bandwidths, var_type, feature):
    column = column.reshape(-1, 1).astype(object)

    if var_type == "o":
        unique_values = ORDERED_CATEGORICAL_VALUES[feature]
    else:
        unique_values = np.unique(column)

    kde, preprocess = initialize_kde(data=column,
                                     var_types=[var_type],
                                     categorical_values=[unique_values],
                                     bw=[bandwidths[0]],
                                     _KDE=FastKDEMultivariate,
                                     _scale_numerical_columns=False
                                     )

    pp_column = preprocess(column)

    pp_values = [x[0] for x in preprocess([[val] for val in unique_values])]
    value_map = {enc_value: value for value, enc_value in zip(unique_values, pp_values)}
    print(value_map)

    results = []
    for bw in bandwidths:
        pdf = kde.pdf(pp_column, [bw])
        results.append((bw, pp_column, pdf, value_map))

    return results


def plot_estimation_result(feature, results):
    import matplotlib.pyplot as plt

    pp_column = results[0][1]
    value_map = results[0][3]

    n_bins = int(np.max(pp_column) - np.min(pp_column)) + 1
    bins = np.linspace(np.min(pp_column) - 0.5, np.max(pp_column) + 0.5, num=n_bins + 1)

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    ax.hist(pp_column, bins=bins, label='Histogram from samples',
            zorder=5, edgecolor='k', density=True, alpha=0.5)
    ax.set_xticks(list(value_map.keys()))
    ax.set_xticklabels(list(value_map.values()))

    for bw, support, density, _ in results:
        if isinstance(bw, str):
            label = f'KDE from samples, bw = {bw}'
        else:
            label = f'KDE from samples, bw = {bw:0.5f}'
        ax.plot(support, density, 'o', lw=2, zorder=10,
                label=label)
    ax.legend(loc='best')
    ax.grid(True, zorder=-5)

    plt.title(feature)
    plt.show()


if __name__ == "__main__":
    _bandwidths = {
        "u": [0.0, 0.01, 0.1, 0.5, 0.95],
        "o": [0.0, 0.1, 0.2, 0.3 ,0.5, 0.6]
    }

    samples, targets = load_data()

    for i, (feature, var_type) in enumerate(zip(FEATURES, VAR_TYPES)):

        if var_type != "c":
            column = samples[:, i]

            results = estimate_bandwidth(column,
                                         bandwidths=_bandwidths.get(var_type, None),
                                         var_type=var_type,
                                         feature=feature)
            plot_estimation_result(feature, results)
