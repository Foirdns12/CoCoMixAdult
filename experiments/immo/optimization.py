from models.immo_rf import load_model
import numpy as np
from experiments.immo.load import load_vorbereitung
from data.immodata import load_data
import nevergrad as ng
from lib.choice import ConditionalTransitionChoice
from sklearn.model_selection import train_test_split
from lib.kde import initialize_pdf

EPSILON = 1e-6
samples, targets, features = load_data(fillna=True)#d
pdf, transmatrices, distmatrices, MAD, categorical_values, samplestest, targetstest, var_types, features = load_vorbereitung()
bw = [0.001, 0.001, 0.023, 0.11, 0.4, 0.001, 0.0128, 0.9976076555, 0.001, 0.001, 0.152, 0.001, 0.001, 0.001]#d
samplestrain,samplestest,targetstrain,targetstest = train_test_split(samples,targets,train_size=.7)#d
pdf = initialize_pdf(samplestrain, var_types, categorical_values, bw)#d



# TODO: Wenn Datensatz behalten: Bevölkerungsdichte mit aufnehmen aus Postleitzahl, siehe immoscout scraper online
print(features)
print("Matrix")
print(transmatrices["obj_cellar"])
transmatrices["obj_cellar"] = [[1 - 0.8114885803262701, 0.8114885803262701],
                               [0.8586036281815275, 1 - 0.8586036281815275]]


def make_decision_function(predict_proba, pdf, target_class, alpha, beta, orig, num_classes):

    target = np.zeros(num_classes)
    target[target_class] = 1.0

    def decision_function(*args, print_=False):

        sample = np.array(args)

        first = np.power(np.abs(target - predict_proba([sample]))[0], 2)

        second = 0
        r=0
        for i in range(0, len(sample)):
            if var_types[i] == "c":
                second = second + np.abs(sample[i] - orig[i]) / MAD[features[i]]
            else:
                if sample[i] != orig[i]:
                    dist = distmatrices[features[i]]
                    #print(dist)
                    val = categorical_values[r]
                    s = 9999
                    o = 9999
                    for j in range(0, len(val)):
                        if sample[i] == val[j]:
                            s = j
                        if orig[i] == val[j]:
                            o = j
                    second = second + beta * (1 / (EPSILON + dist[o][s])) - beta
                r=r+1

        third = alpha * (1 / (EPSILON + pdf(sample)))

        if print_:
            print(first, second, third, first + second + third)
            print(alpha)

        return first + second + third

    return decision_function


clf = load_model()

X = samplestest
Y = targetstest

print("Test score:", clf.score(X, Y))

# Pick one sample as the fact
num = 32
fact = X[num]
fact_class = clf.predict([list(fact)])[0]
foil_class = abs(fact_class + 50000)
print(fact)
print(f"Object {num} is estimated to be worth {fact_class}")
print(pdf(fact))

dec_func = make_decision_function(clf.predict, pdf, foil_class, 0.0001, 0.5, fact)

print(dec_func(*list(fact), print_=True))
print(f"Current value for decision function: {dec_func(*list(fact), print_=True):0.4f}")

# numerische Variablen

obj_lotArea = ng.p.Scalar(fact[2])
obj_lotArea.set_bounds(a_min=0, method="constraint")
obj_lotArea.set_mutation(sigma=(MAD[features[2]] ** 0.8) * 0.3)

obj_yearConstructed = ng.p.Scalar(fact[3])
obj_yearConstructed.set_bounds(a_max=2023, method="constraint")
obj_yearConstructed.set_mutation(sigma=(MAD[features[3]] ** 0.8) * 0.3)
obj_yearConstructed.set_integer_casting()

obj_noParkSpaces = ng.p.Scalar(fact[4])
obj_noParkSpaces.set_bounds(a_min=0, method="constraint")
obj_noParkSpaces.set_mutation(sigma=(MAD[features[4]] ** 0.8) * 0.3)
obj_noParkSpaces.set_integer_casting()

obj_livingSpace = ng.p.Scalar(fact[6])
obj_livingSpace.set_bounds(a_min=0, method="constraint")
obj_livingSpace.set_mutation(sigma=(MAD[features[6]] ** 0.8) * 0.3)

obj_noRooms = ng.p.Scalar(fact[10])
obj_noRooms.set_bounds(a_min=0, method="constraint")
obj_noRooms.set_mutation(sigma=(MAD[features[10]] ** 0.8) * 0.3)
obj_noRooms.set_integer_casting()

obj_heatingType = ConditionalTransitionChoice(categorical_values[0],
                                              transitions=transmatrices["obj_heatingType"])
obj_heatingType.value = fact[0]

obj_newlyConst = ConditionalTransitionChoice(categorical_values[1],
                                             # informationen über neubau sind eigentlich komplett im baujahr enthalten
                                             transitions=transmatrices["obj_newlyConst"])
obj_newlyConst.value = fact[1]

obj_cellar = ConditionalTransitionChoice(categorical_values[2],
                                         transitions=transmatrices["obj_cellar"])
obj_cellar.value = fact[5]

# geo_krs = ConditionalTransitionChoice(categorical_values["geo_krs"],
#                                           transitions=transmatrices["geo_krs"])
# geo_krs.value = fact[7]

obj_condition = ConditionalTransitionChoice(categorical_values[4],
                                            transitions=transmatrices["obj_condition"])
obj_condition.value = fact[8]

obj_interiorQual = ConditionalTransitionChoice(categorical_values[5],
                                               transitions=transmatrices["obj_interiorQual"])
obj_interiorQual.value = fact[9]

obj_rented = ConditionalTransitionChoice(categorical_values[6],
                                         transitions=transmatrices["obj_rented"])
obj_rented.value = fact[11]

obj_buildingType = ConditionalTransitionChoice(categorical_values[7],
                                               transitions=transmatrices["obj_buildingType"])
obj_buildingType.value = fact[12]

obj_barrierFree = ConditionalTransitionChoice(categorical_values[8],
                                              transitions=transmatrices["obj_barrierFree"])
obj_barrierFree.value = fact[13]

instru = ng.p.Instrumentation(obj_heatingType, obj_newlyConst, obj_lotArea, obj_yearConstructed, obj_noParkSpaces,
                              obj_cellar, obj_livingSpace, fact[7], obj_condition, obj_interiorQual, obj_noRooms,
                              obj_rented, obj_buildingType, obj_barrierFree)
# print(obj_heatingType)
# obj_heatingType.mutate()
# print(obj_heatingType)

optimizer = ng.optimizers.OnePlusOne(parametrization=instru, budget=100)
result = optimizer.minimize(dec_func)
print(result.value)
print(f"Current value for decision function {dec_func(*list(result.value[0]), print_=True):0.4f}")
print(clf.predict([list(result.value[0])]))
