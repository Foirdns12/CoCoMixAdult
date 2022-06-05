from demonstration.demonstration_data import load_data, VAR_TYPES, FEATURES, ORDERED_CATEGORICAL_VALUES, \
    UNORDERED_CATEGORICAL_VALUES

from demonstration.density_estimation.bandwidths import final_bandwidths
from lib.fast_kde import FastKDEMultivariate
from lib.kde import initialize_kde

samples, _ = load_data(train=True)
assert samples.shape[1] == len(final_bandwidths)

categorical_values = []
for i, (feature, var_type) in enumerate(zip(FEATURES, VAR_TYPES)):
    if var_type == "u":
        categorical_values.append(UNORDERED_CATEGORICAL_VALUES[feature])
    elif var_type == "o":
        categorical_values.append(ORDERED_CATEGORICAL_VALUES[feature])

kde, preprocess = initialize_kde(data=samples,
                                 var_types=VAR_TYPES,
                                 categorical_values=categorical_values,
                                 bw=final_bandwidths,
                                 _KDE=FastKDEMultivariate,
                                 _scale_numerical_columns=False,
                                 nan_as_cache_mean=True)

if __name__ == "__main__":
    import time
    print("Load data")
    test_samples, _ = load_data(train=False)
    print("Preprocess")
    p_samples = preprocess(test_samples)
    print(type(kde))
    print("Compute KDE for 100")
    start = time.time()
    kde.pdf(p_samples[:100])
    print("Took", time.time() - start)
    print("Compute KDE for 1000")
    start = time.time()
    kde.pdf(p_samples[:1000])
    print("Took", time.time() - start)
    print(f"Compute KDE for all {p_samples.shape}")
    start = time.time()
    kde.pdf(p_samples)
    print("Took", time.time() - start)
