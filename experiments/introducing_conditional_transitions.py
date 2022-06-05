import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np

from lib.choice import ConditionalTransitionChoice

# Nevergrad supports scalars:
age = ng.p.Scalar(50.0)
age.integer = True
age.set_bounds(10, 120)
age.set_mutation(0.1, None)
age.mutate()

ages = []
for i in range(100):
    age.mutate()
    ages.append(age.value)

plt.plot(range(100), ages)
plt.xlabel("Iteration")
plt.ylabel("Age")
plt.show()


# Nevergrad also supports Transition choices, which move forward and backwards along a list of choices
# We can set the transition weights as [stay at current choice, move 1 step, move 2 steps, ...]
# Note that the direction is chosen at random, and the probabilities are computed as softmax(transitions)
levels = ["kindergarden", "elementary", "middle", "high", "bachelor", "master"]
education = ng.p.TransitionChoice(levels,
                                  transitions=[10.0, 25.0, 0.001])
education.value = "high"

education_levels = []
for i in range(50):
    education.mutate()
    education_levels.append(education.value)

plt.plot(range(50), education_levels)
plt.xlabel("Iteration")
plt.ylabel("Education Level")
plt.show()


# We can expand this concept to allow for transitions across a matrix of values
# For example, one can transition from "Married" to both "Divorced" or "Widowed",
# but never from "Single" to "Divorced".
#
# The weights are now given as a matrix, where entry (N, M) encodes the weight
# of the transition between choice N to choice M. Note that no softmax is used here,
# but the weights are normalized to a sum of 1 and taken as probabilities. (I find this
# to be much less confusing, because then setting (N, M)=0 rules out that transition.)
#
# (Note: This is a somewhat artifical example, as during optimization one would in fact
# want to be able to "go back" in later iterations. There, given the case that a
# person is "Married", one would remove the "Single" option entirely.
# This is an example of real-world knowledge not encoded in the tranining data
# entering the search process for a suitable counterfactual.)
civil_status = ConditionalTransitionChoice(["Single", "Married", "Divorced", "Widowed"],
                                           transitions=np.array([[40, 2, 0, 0],
                                                                 [0, 100, 5, 2],
                                                                 [0, 5, 25, 0],
                                                                 [0, 2, 0, 50]]))

civil_status.value = "Single"

states = []
for i in range(100):
    civil_status.mutate()
    states.append(civil_status.value)

plt.plot(range(100), states)
plt.xlabel("Iteration")
plt.ylabel("Civil status")
plt.show()
