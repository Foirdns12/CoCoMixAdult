import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariate, EstimatorSettings

from data.reduced_adult import load_data, FEATURES, VALUES

sns.set()

print(FEATURES)
var_type = np.array(["c", "u", "o", "u", "u", "u", "c", "c"])
assert len(var_type) == len(FEATURES)

X, Y, _ = load_data()

print("First 5 rows", X[:5])

Xt = X.transpose()
for type_, col in zip(var_type, range(Xt.shape[0])):
    print(f"Column {col} of type {type_}")
    if type_ != "c":
        print("- Prior to encoding:", Xt[col][:10])
        enc = OrdinalEncoder(categories=[VALUES[FEATURES[col]] + ["?"]])
        enc.fit(Xt[col].reshape(-1, 1))
        Xt[col] = enc.transform(Xt[col].reshape(-1, 1)).flatten().astype(np.int)
        print("- After encoding:", Xt[col][:10])
    else:
        print("- Prior to scaling:", Xt[col][:10])
        scaler = StandardScaler()
        Xt[col] = scaler.fit_transform(Xt[col].reshape(-1, 1).astype(np.float)).flatten()
        print("- After scaling:", Xt[col][:10])

Xt = Xt.astype(np.float)
X = Xt.transpose()

print("First 5 rows", X[:5])

var_type_str = "".join(var_type)
print("Var type str", var_type_str)

estimator_settings = EstimatorSettings(efficient=False, randomize=True)

kde = KDEMultivariate(X, var_type_str, bw=[0.1, 0.001, 0.001, 0.001, 0.001, 0.55, 0.005, 0.05], defaults=estimator_settings)
print("Fitted")
print(kde.bw)

y_vals = [1/(1e-6 + kde.pdf(x)) for x in X[:2000]]
print("Mean", np.mean(y_vals), "Median", np.median(y_vals))
the_mean = np.mean(y_vals)
plt.plot(range(1000), y_vals[:1000])
# plt.contour(Xt[0], Xt[1], [[kde.pdf([x0, x1]) for x1 in Xt[1]] for x0 in Xt[0]])
plt.show()

x = X[315]


for i, feature in enumerate(FEATURES):
    feature_min, feature_max = np.min(Xt[i]), np.max(Xt[i])
    print(feature, feature_min, feature_max)
    if var_type[i] == "c":
        val_range = np.arange(feature_min, feature_max, (feature_max - feature_min)/100)
    else:
        val_range = np.arange(feature_min, feature_max + 1, 1)

    def make_x(val):
        new_x = x.copy()
        new_x[i] = val
        return new_x

    plt.plot(val_range, [(1/(1e-6 + kde.pdf(make_x(val))))/1e6 for val in val_range], label=feature)

    plt.legend()
    plt.show()

