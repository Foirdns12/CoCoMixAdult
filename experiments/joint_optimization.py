import nevergrad as ng
import numpy as np

from data.reduced_adult import load_data, FEATURES, VALUES, load_test
from lib.choice import ConditionalTransitionChoice
from lib.kde import initialize_pdf
from models.reduced_adult_rf import load_model

EPSILON = 1e-6

X, _, _ = load_data()

#print(FEATURES)
var_types = np.array(["c", "u", "o", "u", "u", "u", "c", "c"])
assert len(var_types) == len(FEATURES)

categorical_values = [VALUES[feature] + ["?"] for idx, feature in enumerate(FEATURES)
                      if var_types[idx] != "c"]
#print(np.var(VALUES[3]))


# taken from reduced-density experiments
bw = [0.1, 0.001, 0.001, 0.001, 0.001, 0.55, 0.005, 0.05]

pdf, _ = initialize_pdf(X, var_types, categorical_values, bw)


def make_decision_function(predict_proba, pdf, target_class, alpha):
    def decision_function(age, workclass, education, marital_status, occupation, sex,
                          capital_gain, hours_per_week, print_=False):
        sample = [age, workclass, education, marital_status, occupation, sex,
                  capital_gain, hours_per_week]
        first = (1 - predict_proba([sample])[0][target_class])

        second = alpha * (1 / (EPSILON + pdf(sample)))

        if print_:
            print(first, second, first + second)
            print(alpha)

        return first + second

    return decision_function


clf = load_model()
X, Y, STD = load_test()
print(STD)

print("Test score:", clf.score(X, Y))

# Pick one sample as the fact
num = 29
fact = X[num]
fact_class = Y[num]
foil_class = abs(fact_class - 1)
print(f"Person {num} is in class {fact_class}")
print(fact)

# Create the decision function f(sample) = foil class probability
dec_func = make_decision_function(clf.predict_proba, pdf, foil_class, 0.00001)
print(f"Current value for decision function: {dec_func(*list(fact), print_=True):0.4f}")
print(clf.predict_proba([list(fact)]))


# Define the parameters that the algorithm can vary
# For details, see https://facebookresearch.github.io/nevergrad/parametrization.html
age = ng.p.Scalar(fact[0])  # initialize to the age of the fact
age.set_bounds(16, 120, method="constraint")
age.set_integer_casting()
age.set_mutation()

workclass = ng.p.Choice(VALUES["workclass"] + ["?"])  # chosen from all possible values for 'workclass'
workclass.value = fact[1]  # initialize to workclass of the fact

# values for 'education' in ascending order
education = ["Preschool", "1st-4th", "5th-6th", "7th-8th", "9th", "10th", "11th", "12th", "HS-grad", "Prof-school",
             "Assoc-acdm", "Assoc-voc", "Some-college", "Bachelors", "Masters",
             "Doctorate"]
transitions = np.exp(-np.arange(0, len(education) + 1, step=1))  # how likely is it to transition N steps?
education = ng.p.TransitionChoice(["?"] + education,
                                  transitions=transitions)  # choose the previous/next value from an ordered list
education.value = fact[2]  # initialize to education of the fact

marital_status = ng.p.Choice(VALUES["marital-status"] + ["?"])
marital_status.value = fact[3]

occupation = ng.p.Choice(VALUES["occupation"] + ["?"])
occupation.value = fact[4]

#sex = ng.p.Choice(VALUES["sex"] + ["?"])
#sex.value = fact[5]

capital_gain = ng.p.Scalar(fact[6])
capital_gain.set_mutation(sigma=(STD[6]**0.8)*0.3) #anpassung von exponent fuehrt zu warnings und dividebyzero warnings
print((STD[6]**0.8)*0.3)
print((STD[7]**0.8)*0.3)
capital_gain.set_bounds(-100000, 100000, method="constraint")
#capital_gain.set_integer_casting()#, "Scalar{int}[recombination=average,sigma=Log{exp=50}[recombination=average,sigma=50]]"

hours_per_week = ng.p.Scalar(fact[7])#.affined(100,-10).bounded(-10000,10000)
hours_per_week.set_bounds(10, 100, method="constraint")
hours_per_week.set_mutation(sigma=(STD[7]**0.8)*0.3)
hours_per_week.set_integer_casting()#, "Scalar{int}[recombination=average,sigma=Log{exp=1.2}[recombination=average,sigma=1.0]]"

# We vary everything except the sex
instru = ng.p.Instrumentation(age, workclass, education, marital_status, occupation, fact[5], capital_gain, hours_per_week)

optimizer = ng.optimizers.OnePlusOne(parametrization=instru, budget=100)
result = optimizer.minimize(dec_func)
print(result.value)
print(f"Current value for decision function {dec_func(*list(result.value[0]), print_=True):0.4f}")
print(clf.predict_proba([list(result.value[0])]))
