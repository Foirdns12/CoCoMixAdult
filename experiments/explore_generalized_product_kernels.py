import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.kernel_density import KDEMultivariate
sns.set()

left_half = np.random.normal(loc=1.0,
                             scale=0.25,
                             size=5000)
right_half = np.random.normal(loc=2.0,
                              scale=0.5,
                              size=5000)
continuous = np.concatenate([left_half, right_half])
np.random.shuffle(continuous)

class_probability = continuous/(np.max(continuous) - np.min(continuous)) + np.random.normal(0.0, 0.1)
categorical = np.where(class_probability < 0.25, np.ones_like(class_probability), 2*np.ones_like(class_probability))
categorical = np.where(class_probability > 0.75, 3*np.ones_like(class_probability), categorical)

X = np.vstack([categorical, continuous])
kde = KDEMultivariate(X, "uc", bw=[0.01, 0.1])

sns.jointplot(categorical, continuous, kind="scatter")
plt.show()

xx = np.hstack([np.linspace(-1, 5, num=500), np.linspace(0, 5, num=500), np.linspace(0, 5, num=500)])
yy = np.hstack([np.ones(shape=(500,)), 2*np.ones(shape=(500,)), 3*np.ones(shape=(500,))])
zz = kde.pdf(np.vstack([yy, xx]))
ax = plt.subplot()
sns.distplot(continuous[categorical == 1], kde=False, norm_hist=True, ax=ax)
sns.distplot(continuous[categorical == 2], kde=False, norm_hist=True, ax=ax)
sns.distplot(continuous[categorical == 3], kde=False, norm_hist=True, ax=ax)
sns.lineplot(xx, zz, hue=yy.astype(int), ax=ax)
plt.show()
