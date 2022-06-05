import nevergrad as ng
import numpy as np
from demonstration.demonstration_data import load_df, fill_numerical_column_by_cond_median, ORDERED_CATEGORICAL_VALUES, \
    FEATURES, CATEGORICAL, NUMERICAL
from demonstration.demonstration_immo_model import load_model_from_weights
from demonstration.density_estimation.compute_kde import kde, preprocess
from experiments.immo.load import load_vorbereitung
from lib.choice import ConditionalTransitionChoice

# TODO: Wie gewichten wir zwischen kategorischen und numerischen features
# TODO: Warum sind nach fill_numerical_column_by_cond_median immernoch nan's vorhanden?
#  -> Sind keine vorhanden, siehe Test
# TODO:Optimierungsgewichte sinnvoll einstellen
# TODO: Warum hat newly const keinen einfluss auf die dichte
# TODO: warum wird das Alter immer nach unten in Richtung 1970 korrigiert?

# samples, targets = demonstration_data.load_data(train=False)
df_test = load_df(train=False)
df_train = load_df(train=True)

# Manche Spalten haben noch NaNs
for column in df_test.columns:
    print(column, np.any(df_test[column].isna()))

df_test = fill_numerical_column_by_cond_median(source_df=df_train,
                                               condition_column="obj_buildingType",
                                               target_columns=["obj_numberOfFloors", "obj_noParkSpaces",
                                                               "obj_livingSpace", "obj_lotArea", "obj_noRooms",
                                                               "obj_yearConstructed"],
                                               target_df=df_test)

# Jetzt hat keine Spalte mehr NaNs
for column in df_test.columns:
    print(column, np.any(df_test[column].isna()))
assert not np.any([np.any(df_test[column].isna()) for column in df_test.columns])

model = load_model_from_weights()
EPSILON = 1e-20
transmatrices, distmatrices, MAD, categorical_values, var_types, features = load_vorbereitung()
print(features)

def pdf(sample):
    return kde.pdf(preprocess(sample.reshape(1, -1)))


categorical_values[3] = ORDERED_CATEGORICAL_VALUES["obj_interiorQual"]


def make_decision_function(predict_proba, pdf, target_class, alpha, beta, orig, num_classes):
    """

    :param predict_proba:
    :param pdf:
    :param target_class:
    :param alpha:
    :param beta:
    :param orig:
    :param num_classes:
    :return:
    """
    target = np.zeros(num_classes)
    target[target_class] = 1.0

    numerical_values = np.array(var_types) == "c"
    mad = np.array([MAD[feature] if feature in NUMERICAL else None
                    for feature in FEATURES], dtype=np.float)

    categorical_features = [feature for feature in FEATURES
                            if feature in CATEGORICAL]

    def compute_distance(feature, val1, val2):
        if val1 == val2:
            return 1.0

        dist = distmatrices[feature]
        values = list(categorical_values[categorical_features.index(feature)])

        idx1 = values.index(val1)
        idx2 = values.index(val2)

        return 1 / (EPSILON + dist[idx1][idx2])

    def decision_function(*args, print_=False):
        # prediction
        p_sample = {feature: np.array([[value]]) for feature, value in zip(FEATURES, args)}

        _prediction = predict_proba(p_sample)[0]
        # print(f"- Class: {np.argmax(_prediction)}, target probability: {_prediction[target_class]:0.4f}")
        first = np.sum(np.power(_prediction - target, 2))

        # distance
        sample = np.array(args)

        _second = np.zeros_like(sample, dtype=np.float)

        # numerical distance term
        _second[numerical_values] = np.abs(
            sample[numerical_values].astype(np.float) - orig[numerical_values].astype(np.float)) / mad[numerical_values]

        # categorical distance term
        categorical_distances = np.array([compute_distance(feature, val1, val2) if feature in CATEGORICAL else None
                                          for feature, val1, val2 in zip(FEATURES, orig, sample)], dtype=np.float)
        _second[~numerical_values] = beta * categorical_distances[~numerical_values] - beta
        second = np.sum(_second)

        # density
        third = alpha * (1 / (EPSILON + pdf(sample)))

        if print_:
            print(f"Losses:\n"
                  f"- prediction {first}\n"
                  f"- distance {second}\n"
                  f"- density: {third}\n"
                  f"= total {first + second + third}")
            print(f"Prediction:\n"
                  f"- class {np.argmax(_prediction)} with probability {np.max(_prediction)}\n"
                  f"- target class {target_class} with probability {_prediction[target_class]}")

        return 40 * first + second + 80 * third

    return decision_function


# Pick one sample as the fact
num = 11
# factl=df_train.iloc[num]
# fact=np.array([float(factl.iloc[0]),float(factl.iloc[1]),float(factl.iloc[2]),float(factl.iloc[3]),float(factl.iloc[4]),float(factl.iloc[5]), factl.iloc[6], factl.iloc[7],factl.iloc[8],factl.iloc[9],factl.iloc[10],factl.iloc[11],factl.iloc[12],factl.iloc[13]])
# fact = samples[4]
# pfact=preprocess(fact.reshape(1,-1))
fact = df_test[FEATURES].iloc[num]
tfact = fact.to_dict()
tfact = {k: np.array([[v]]) for k, v in tfact.items()}
prediction = model.predict(tfact)[0]

fact_class = np.argmax(prediction)
foil_class = 4  # abs(fact_class + 3)
print(tfact)
print(f"Object {num} is estimated to be in class {fact_class}")
# print(pdf(pfact))

nclasses = len(prediction)
print(nclasses)
nfact = fact.to_numpy()
print(nfact)
dec_func = make_decision_function(model.predict, pdf, foil_class, alpha=1e-20, beta=0.1,
                                  orig=nfact, num_classes=nclasses)

print(dec_func(*list(fact), print_=True))
print(f"Current value for decision function: {dec_func(*list(fact), print_=True):0.4f}")

# numerische Variablen

obj_lotArea = ng.p.Scalar(fact[4])
obj_lotArea.set_bounds(0, 600000, method="constraint")
obj_lotArea.set_mutation(sigma=(MAD[features[4]] ** 0.5) * 1.1)
print('obj_lotArea')
print((MAD[features[4]] ** 0.5) * 1.1)


obj_yearConstructed = ng.p.Scalar(fact[0])
obj_yearConstructed.set_bounds(a_min=0, a_max=2023, method="constraint")
obj_yearConstructed.set_mutation(sigma=(MAD[features[0]] ** 0.5) * 1.1)
obj_yearConstructed.set_integer_casting()
print('year_constructed')
print((MAD[features[0]] ** 0.5) * 1.1)

obj_noParkSpaces = ng.p.Scalar(fact[5])
obj_noParkSpaces.set_bounds(a_min=0, a_max=200, method="constraint")
obj_noParkSpaces.set_mutation(sigma=(MAD[features[5]] ** 0.5) * 1.1)
obj_noParkSpaces.set_integer_casting()
print('obj_noParkspaces')
print((MAD[features[5]] ** 0.5) * 1.1)

obj_livingSpace = ng.p.Scalar(fact[1])
obj_livingSpace.set_bounds(a_min=0, a_max=10000, method="constraint")
obj_livingSpace.set_mutation(sigma=(MAD[features[1]] ** 0.5) * 1.1)
print('livspace')
print((MAD[features[1]] ** 0.5) * 1.1)

obj_noRooms = ng.p.Scalar(fact[2])
obj_noRooms.set_bounds(a_min=0, a_max=1000, method="constraint")
obj_noRooms.set_mutation(sigma=(MAD[features[2]] ** 0.5) * 1.1)
obj_noRooms.set_integer_casting()
print('obj_noRooms')
print((MAD[features[2]] ** 0.5) * 1.1)

obj_numberOfFloors = ng.p.Scalar(fact[3])
obj_numberOfFloors.set_bounds(a_min=0, a_max=20, method="constraint")
obj_numberOfFloors.set_mutation(sigma=(MAD[features[3]] ** 0.5) * 1.1)
obj_numberOfFloors.set_integer_casting()
print('obj_nofloors')
print((MAD[features[3]] ** 0.5) * 1.1)

# print(categorical_values[6])
# print(fact[10])

obj_heatingType = ConditionalTransitionChoice(categorical_values[4],
                                              transitions=transmatrices["obj_heatingType"])
obj_heatingType.value = fact[10]

obj_newlyConst = ConditionalTransitionChoice(categorical_values[5],
                                             # informationen über neubau sind eigentlich komplett im baujahr enthalten
                                             transitions=transmatrices["obj_newlyConst"])
obj_newlyConst.value = fact[11]

obj_cellar = ConditionalTransitionChoice(categorical_values[1],
                                         transitions=transmatrices["obj_cellar"])
obj_cellar.value = fact[7]

# geo_krs = ConditionalTransitionChoice(categorical_values["geo_krs"],
#                                           transitions=transmatrices["geo_krs"])
# geo_krs.value = fact[7]

obj_condition = ConditionalTransitionChoice(categorical_values[2],
                                            transitions=transmatrices["obj_condition"])
obj_condition.value = fact[8]

obj_interiorQual = ConditionalTransitionChoice(categorical_values[3],
                                               transitions=transmatrices["obj_interiorQual"])
obj_interiorQual.value = fact[9]

obj_buildingType = ConditionalTransitionChoice(categorical_values[0],
                                               transitions=transmatrices["obj_buildingType"])
obj_buildingType.value = fact[6]

# obj_regio1 = ConditionalTransitionChoice(categorical_values[6], transitions=transmatrices["obj_regio1"])
# obj_regio1.value = fact[12]

instru = ng.p.Instrumentation(obj_yearConstructed, obj_livingSpace, obj_noRooms, obj_numberOfFloors, obj_lotArea,
                              obj_noParkSpaces, obj_buildingType, obj_cellar, obj_condition, obj_interiorQual,
                              obj_heatingType, obj_newlyConst, fact[12], fact[13])
# print(obj_heatingType)
# obj_heatingType.mutate()
# print(obj_heatingType)

optimizer = ng.optimizers.FastGADiscreteOnePlusOne(parametrization=instru, budget=1000)
result = optimizer.minimize(dec_func, verbosity=0)
print('fact')
print(nfact)
print('result')
print(result.value)
print(f"Current value for decision function {dec_func(*list(result.value[0]), print_=True):0.4f}")
