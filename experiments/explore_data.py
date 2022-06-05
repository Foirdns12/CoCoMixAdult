import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data.adult import load_data, CATEGORICAL, FEATURES, VALUES

sns.set()

X, Y = load_data()

Xt = X.transpose()

df = pd.DataFrame({label: column for label, column in zip(FEATURES, Xt)})
print(df.columns)
print(df.shape)

sns.jointplot("age", "education-num", kind="kde", data=df)
plt.show()
