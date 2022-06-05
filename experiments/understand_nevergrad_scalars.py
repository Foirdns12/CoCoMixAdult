import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import nevergrad as ng

sns.set()

# Simple example to verify that we sample from
# a normal distribution around the current value
# when we set sigma to non-mutable
age = ng.p.Scalar(mutable_sigma=False)
age.set_mutation(sigma=10.0)

values = []
for _ in range(10000):
    age.value = 0.0
    age.mutate()
    values.append(age.value)

fig, ax = plt.subplots()
sns.distplot(values, ax=ax)
sns.distplot(np.random.normal(0.0, 10.0, size=10000), ax=ax)
plt.show()

