import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from statsmodels.nonparametric.kernel_density import KDEMultivariate, EstimatorSettings

from data.adult import load_data, FEATURES, VALUES

sns.set()
print(FEATURES)
var_type = np.array([ 'c',
    'c',
    'c',
    'c',
    'u',
    'o',
    'u',
    'u',
    'u',
    'u',
    'u',
    'u'])
assert len(var_type) == len(FEATURES)
for type_, feature in zip(var_type, FEATURES):
    print(type_, feature)

X, Y = load_data()

features_to_keep = ["marital-status", "workclass", "age", "occupation", "education-num", "education"] #, "marital-status", "age", "occupation"]  # , "workclass", "education", "education-num", "marital-status", "occupation"]
feature_mask = [True if feature in features_to_keep else False for feature in FEATURES]

var_type = var_type[feature_mask]
used_features = [feature for feature in FEATURES if feature in features_to_keep]

Xt = X[:20000].transpose()[feature_mask]
for type_, col in zip(var_type, range(Xt.shape[0])):
    if type_ != "c":
        print(used_features[col])
        print(Xt[col][:10])
        enc = OrdinalEncoder(categories=[VALUES[used_features[col]] + ["?"]])
        enc.fit(Xt[col].reshape(-1, 1))
        Xt[col] = enc.transform(Xt[col].reshape(-1, 1)).flatten().astype(np.int)
        print(Xt[col][:10])
    else:
        scaler = StandardScaler()
        Xt[col] = scaler.fit_transform(Xt[col].reshape(-1, 1).astype(np.float)).flatten()

# sns.jointplot(Xt[0], Xt[1], kind="kde")
# plt.show()

X = Xt.transpose().astype(np.float)

var_type_str = "".join(var_type)
print(var_type_str)

estimator_settings = EstimatorSettings(efficient=False, randomize=True)

kde = KDEMultivariate(X, var_type_str, bw=[10.0, 0.5, 0.5, 0.5, 0.5, 0.5], defaults=estimator_settings)
print("Fitted")
print(kde.bw)
# print(kde.imse(kde.bw))
# for feature, bw in zip(used_features, kde.bw):
#     print(feature, bw)

plt.plot(range(1000), [1/kde.pdf(x) for x in X[:1000]])
# plt.contour(Xt[0], Xt[1], [[kde.pdf([x0, x1]) for x1 in Xt[1]] for x0 in Xt[0]])
plt.show()
