import matplotlib.pyplot as plt
import seaborn as sns

from data.immodata import load_data
from models.immo_rf import load_model

model = load_model()
samples, targets, features = load_data(fillna=True)

for i, sample in enumerate(samples[:50]):
    print(model.predict([sample])[0], "\t", targets[i])

delta = (model.predict(samples) - targets) / targets

ax = sns.distplot(delta[delta < 5])
ax.set_xlim([0, 5])
plt.show()
