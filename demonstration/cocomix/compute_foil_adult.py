import os
import os
import uuid
from typing import Union

import numpy as np
import pandas as pd

from demonstration.cocomix.cocomix import make_decision_function
from demonstration.demonstration_data import FEATURES, VAR_TYPES, \
    ALL_CATEGORICAL_VALUES
from demonstration.density_estimation.bandwidths_adult import FINAL_BANDWIDTHS
from demonstration.density_estimation.compute_kde import kde, preprocess
from demonstration.proxy_measures.analyze import analyze_foils, instantiate_all_measures
from demonstration.transition_matrices.compute_distance_matrices_adult import compute_distance_matrix
from demonstration.transition_matrices.unit_matrices import get_subspace_unit_transition_matrix
from lib.optimizer.optimizer import Optimizer
from lib.optimizer.parametrization import Constant, Scalar, Categorical, Parametrization

PATH = os.path.dirname(os.path.abspath(__file__))


def pdf(sample):
    return kde.pdf(preprocess(sample.reshape(1, -1)))

def get_losses(history, step: Union[int, str] = "last"):
    if step == "last":
        idx = len(history["step"]) - 1
    elif step == "first":
        idx = 0
    else:
        idx = history["step"].index(step)
    return {k: v[idx] for k, v in history.items()}




def conf_parametrization_use_case(fact_sample, df_train, transition_matrices, distance_matrices,boundaries):
    # final configuration step. In this case the configuration/characteristics for the features which are used for the optimization

    fact = fact_sample[FEATURES].to_numpy()[0]

    age = Scalar(fact[0])
    age.sigma = FINAL_BANDWIDTHS[FEATURES[0]]
    age.cast_to_integer = True

    fnlwgt = Scalar(fact[1])
    fnlwgt.sigma = FINAL_BANDWIDTHS[FEATURES[2]]
    fnlwgt.cast_to_integer = True

    educationnum = Scalar(fact[2])
    educationnum.sigma = FINAL_BANDWIDTHS[FEATURES[4]]
    educationnum.cast_to_integer = True

    hoursperweek = Scalar(fact[3])
    hoursperweek.sigma = FINAL_BANDWIDTHS[FEATURES[10]]
    hoursperweek.cast_to_integer = True




    if boundaries == True:
        age.set_bounds(lower_bound=df_train['age'].min(),
                        upper_bound=df_train['age'].max())
        fnlwgt.set_bounds(lower_bound=df_train['fnlwgt'].min(),
                          upper_bound=df_train['fnlwgt'].max())
        educationnum.set_bounds(lower_bound=df_train['education-num'].min(),
                            upper_bound=df_train['education-num'].max())
        hoursperweek.set_bounds(lower_bound=df_train['hours-per-week'].min(),
                                  upper_bound=df_train['hours-per-week'].max())



    workclass = Categorical(choices=ALL_CATEGORICAL_VALUES[FEATURES[4]],
                                transition_matrix=np.array(transition_matrices[FEATURES[4]]))

    education = Categorical(choices=ALL_CATEGORICAL_VALUES[FEATURES[5]],
                            transition_matrix=np.array(transition_matrices[FEATURES[5]]))

    maritalstatus = Categorical(choices=ALL_CATEGORICAL_VALUES[FEATURES[6]],
                                transition_matrix=np.array(transition_matrices[FEATURES[6]]))

    occupation = Categorical(choices=ALL_CATEGORICAL_VALUES[FEATURES[7]],
                                transition_matrix=np.array(transition_matrices[FEATURES[7]]))

    relationship = Categorical(choices=ALL_CATEGORICAL_VALUES[FEATURES[8]],
                                transition_matrix=np.array(transition_matrices[FEATURES[8]]))

    race = Categorical(choices=ALL_CATEGORICAL_VALUES[FEATURES[9]],
                                transition_matrix=np.array(transition_matrices[FEATURES[9]]))

    sex = Categorical(choices=ALL_CATEGORICAL_VALUES[FEATURES[10]],
                                transition_matrix=np.array(transition_matrices[FEATURES[10]]))

    nativecountry = Categorical(choices=ALL_CATEGORICAL_VALUES[FEATURES[11]],
                                transition_matrix=np.array(transition_matrices[FEATURES[11]]))


    return age, fnlwgt, educationnum, hoursperweek, workclass, education, maritalstatus, occupation, relationship, race, sex, nativecountry


def use_case_adjustments(transition_matrices, distance_matrices, fact_sample):
    fact = fact_sample[FEATURES].to_numpy()[0]
    categorical_values = ALL_CATEGORICAL_VALUES.copy()

    return categorical_values

def p_fact_adult (p_fact):
    p_fact = pd.DataFrame.from_dict(p_fact)
    return p_fact



def find_foil(model, transition_matrices, distance_matrices, mad, df_train, df_test,boundaries, fact_sample, lambda_=100.0, mu=1.0,
              alpha=50.0, beta=0.1, budget=1000,
              densitycut=10, densityaddloss=1.0, densityscaler=0.05):
    print('--------------')
    print(fact_sample)
    print('--------------')
    p_fact = {k: np.array(v) for k, v in fact_sample[FEATURES].to_dict(orient="list").items()}
    print("\n\nThe fact", p_fact)
    print('#########')
    print(p_fact)
    #Zusatz für Adult Data Frame
    p_fact = p_fact_adult(p_fact)
    prediction = model.predict_proba(p_fact)[0]
    fact_class = np.argmax(prediction)
    if fact_class == 0:
        foil_class = fact_class + 1
    else:
        foil_class = fact_class - 1
    print("Fact class:", fact_class)
    print("Foil class:", foil_class)

    fact = fact_sample[FEATURES].to_numpy()[0]
    categorical_values = ALL_CATEGORICAL_VALUES.copy()

    parametrization = Parametrization(
        *conf_parametrization_use_case(fact_sample, df_train, transition_matrices, distance_matrices,boundaries))

    optimizer = Optimizer(parametrization, mutation_rule="1/n")

    decision_function = make_decision_function(fact=fact,
                                               target_class=foil_class,
                                               input_features=FEATURES,
                                               var_types=VAR_TYPES,
                                               categorical_values=categorical_values,
                                               model=model,
                                               mad=mad,
                                               pdf=pdf,
                                               distance_matrices=distance_matrices,
                                               lambda_=lambda_,
                                               mu=mu,
                                               alpha=alpha,
                                               beta=beta,
                                               densitycut=densitycut,
                                               densityaddloss=densityaddloss,
                                               densityscaler=densityscaler)

    result, history = optimizer.minimize(decision_function, budget=budget)

    return fact, result, history, fact_class, prediction


def calculate_foils(configuration, mad, df_test, model, transition_matrices, distance_matrices, df_train, n=100,
                    factset=None, randomstate=0, metrics=True, boundaries=False):
    def calc(sample):
        fact, result, history, fact_cl, predfact = find_foil(model, transition_matrices, distance_matrices, mad,
                                                             df_train, df_test,boundaries, fact_sample=sample,
                                                             **configuration)

        print("Fact", sample[FEATURES].to_numpy()[0], get_losses(history, "first"))
        print("Foil", result, get_losses(history, "last"))

        fact_for_analysis = {f: v for f, v in zip(FEATURES, fact)}
        foil_for_analysis = {f: v for f, v in zip(FEATURES, result)}

        predclass = get_losses(history, "last")["predicted_class"]
        predvect = get_losses(history, "last")["prediction"]
        id = uuid.uuid4()
        return ((fact_for_analysis, foil_for_analysis, history, configuration, str(id), predclass, fact_cl,
                "0"))

    if factset is not None and randomstate > 0:
        print('Randomstate and n omitted as foilset is given')
    if randomstate > 0:
        randomState = np.random.RandomState(seed=randomstate)
    if metrics:
        measures = instantiate_all_measures(mad)
    record = []
    # print(df_test['factID'])
    if factset is None:
        if randomstate > 0:
            samples = df_test.sample(n=n, random_state=randomstate)
        else:
            samples = df_test.sample(n=n)
        # print(sample)
        for i in range(n):
            sample = samples.iloc[[i]]
            record.append(calc(sample))
    if factset is not None:
        for factid in factset:
            fact = df_test[df_test['factID'] == factid]
            print('fakt')
            print(fact)
            if len(fact['factID']) > 1:
                print('identical observation found in test set')
            if len(fact['factID']) < 1:
                print('factID not found in test set')
            record.append(calc(fact))

    result = {}
    import json
    conf = json.dumps(configuration)
    full_set = [(fact, foil, np.array(history["pdf"]), int(predclass), int(fact_cl)) for fact, foil, history, _, _, predclass, fact_cl, _ in record]

    full_set2 = []
    for fact, foil, history, _, foilid, predclass, fact_cl, factid in record:
        first = {'fact': fact, 'factid': factid, 'foil': foil, 'foilid': foilid, 'history': history, 'conf': conf,
                 'predclass': predclass, 'fact_cl': fact_cl}
        full_set2.append(first)

    result['record'] = record
    result['configuration'] = configuration
    if metrics:
        result['joint_analysis'] = analyze_foils(full_set, measures)
    return result, full_set2
