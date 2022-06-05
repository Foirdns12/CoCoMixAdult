"""Generates a synthetic dataset as described in Poyiadzi et al. (2019)"""
import os

import numpy as np
import pandas as pd


PATH = os.path.dirname(os.path.abspath(__file__))

# horizontal cloud of blue points to the left of the figure
# 200 points distributed uniformly at random across the y-axis
blue_1_y = np.random.uniform(-0.5, 9.5, 200)

# and sampled from a mean-zero Gaussian with 0.4 standard deviation on the x-axis
blue_1_x = np.random.normal(0.0, 0.4, 200)

blue_1_label = np.zeros_like(blue_1_y)

# vertical cloud of red points to the bottom of the figure
# 200 points distributed uniformly at random across the x-axis
red_1_x = np.random.uniform(0, 5, 200)

# and sampled from a mean-zero Gaussian with 0.5 standard deviation on the y-axis
red_1_y = np.random.normal(0.0, 0.5, 200)

red_1_label = np.ones_like(red_1_y)

# vertical cloud of red points to the top-right of the figure
# 100 points sampled from a Gaussian distribution with (3.5,8.0) mean and 0.5 standard deviation.
red_2_x = np.random.normal(3.5, 0.5, 100)
red_2_y = np.random.normal(8.0, 0.5, 100)

red_2_label = np.ones_like(red_2_y)

x = np.hstack([blue_1_x, red_1_x, red_2_x])
y = np.hstack([blue_1_y, red_1_y, red_2_y])

labels = np.hstack([blue_1_label, red_1_label, red_2_label])

perm = np.random.permutation(len(labels))
x = x[perm]
y = y[perm]
labels = labels[perm]

data = np.vstack([x, y]).transpose()

assert np.min(data, axis=0)[0] == np.min(x)
assert np.min(data, axis=0)[1] == np.min(y)
assert np.max(data, axis=0)[0] == np.max(x)
assert np.max(data, axis=0)[1] == np.max(y)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    df = pd.DataFrame({"x": x, "y": y, "label": labels})
    sns.scatterplot("x", "y", hue="label", data=df)
    plt.show()
