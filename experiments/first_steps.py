import nevergrad as ng
import numpy as np

from data.adult import load_test, VALUES, FEATURES
from models.adult_rf import load_model


def make_decision_function(target_class):
    """Create a function that calculates the probability that a sample belongs to the target_class."""

    def decision_function(age, workclass, education, education_num, marital_status, occupation, relationship,
                          race, sex, capital_gain, capital_loss, hours_per_week, native_country):
        return 1 - clf.predict_proba([[age, workclass, education, education_num,
                                       marital_status, occupation, relationship,
                                       race, sex, capital_gain, capital_loss,
                                       hours_per_week, native_country]])[0][target_class]

    return decision_function


# Load the pre-trained model which predicts whether a person earns more than 50k/year (class 1) or not (class 0)
clf = load_model()
X, Y = load_test()
print("Test score:", clf.score(X, Y))

# Pick one sample as the fact
num = 24
fact = X[num]
fact_class = Y[num]
foil_class = abs(fact_class - 1)
print(f"Person {num} is in class {fact_class}")
print(fact)

# Create the decision function f(sample) = foil class probability
dec_func = make_decision_function(foil_class)
print(f"Probability for class {fact_class}: {dec_func(*list(fact)):0.4f}")

# Define the parameters that the algorithm can vary
# For details, see https://facebookresearch.github.io/nevergrad/parametrization.html
age = ng.p.Scalar(fact[0])  # initialize to the age of the fact
age.set_bounds(16, 120, method="constraint")
age.set_integer_casting()


workclass = ng.p.Choice(VALUES["workclass"] + ["?"])  # chosen from all possible values for 'workclass'
workclass.value = fact[1]  # initialize to workclass of the fact

# values for 'education' in ascending order
education = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Prof-school",
             "Assoc-acdm", "Assoc-voc", "Some-college", "Bachelors", "Masters",
             "Doctorate"]
transitions = np.exp(-np.arange(0, len(education) + 1, step=1))  # how likely is it to transition N steps?
education = ng.p.TransitionChoice(["?"] + education, transitions=transitions)  # choose the previous/next value from an ordered list
education.value = fact[2]  # initialize to education of the fact

education_num = ng.p.Scalar(fact[3])
education_num.set_bounds(0, 20, method="constraint")

# We only vary the first four parameters and set the remaining 10 to the fact values
instru = ng.p.Instrumentation(age, workclass, education, education_num, *list(fact[4:]))

# Initialize a 1+1 optimizer (which is a very basic evolutionary algorithm)
optimizer = ng.optimizers.OnePlusOne(parametrization=instru, budget=200)
result = optimizer.minimize(dec_func)
print(result.value)
print(f"Probability for {foil_class}: {1 - dec_func(*list(result.value[0])):0.4f}")

optimizer = ng.optimizers.FastGADiscreteOnePlusOne(parametrization=instru, budget=200)
result = optimizer.minimize(dec_func)
print(result.value)
print(f"Probability for {foil_class}: {1 - dec_func(*list(result.value[0])):0.4f}")