import os
from demonstration.proxy_measures.analyze import analyze_foils, instantiate_all_measures
from demonstration.competing_approaches.wachter.util import calculate_mad
from demonstration.demonstration_data import load_df, FEATURES, VAR_TYPES
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib.ticker as tick

def load_foils():
    PATH = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(PATH, 'daten', '20210325-18181745_foils.json'), "rt") as f:
        record = json.load(f)
    for filename in os.listdir(os.path.join(PATH, 'daten')):
        print(filename)
        if filename != '20210325-18181745_foils.json':
            with open(os.path.join(PATH, 'daten', filename), "rt") as f:
                record += json.load(f)
    return record
def load_foilsbeta():
    PATH = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(PATH, 'datenbeta', '20210331-19392612_betafoils.json'), "rt") as f:
        record = json.load(f)
    for filename in os.listdir(os.path.join(PATH, 'datenbeta')):
        print(filename)
        if filename != '20210331-19392612_betafoils.json':
            with open(os.path.join(PATH, 'datenbeta', filename), "rt") as f:
                record += json.load(f)
    return record


if __name__ == "__main__":
    df_train = load_df(train=True)
    PATH = os.path.dirname(os.path.abspath(__file__))
    mad = calculate_mad(df_train[[feature for feature, var_type in zip(FEATURES, VAR_TYPES)
                                  if var_type == "c"]].to_numpy())
    mad[mad == 0.0] = 0.5
    measures = instantiate_all_measures(mad)
    record=load_foils()

    configuration = {
        "lambda_": 120.0,
        "mu": 1.25,
        "alpha": 3,
        "beta": 0.5,
        "budget": 1000,
        "densitycut": 8,
        "densityaddloss": 1.5,
        "densityscaler": 0.05
    }
    conf2 = json.dumps(configuration)
    full_set = []
    for data in record:
        if data['conf'] == conf2:
            # if data['factid'] in factset:
            full_set.append((data['fact'], data['foil'], np.array(data['history']["pdf"]), 4, 6))
    print('start analysis')
    analysis = analyze_foils(full_set, measures)
    print('passed')
    sparsitymad = analysis['sparsity']['mad']
    closenessmad = analysis['cocomix_distance']['mad']
    feasibility1mad = analysis['density_opt_step_min']['mad']
    feasibility2mad = analysis['density_min_start']['mad']
    typicalitymad = analysis['density_foil']['mad']

    print('sparsity')
    print(sparsitymad)
    print(analysis['sparsity']['mean'])
    print('closeness')
    print(closenessmad)
    print(analysis['cocomix_distance']['mean'])
    print('fesibilityper optimization step')
    print(feasibility1mad)
    print(analysis['density_opt_step_min']['mean'])
    print('fesibility min whole optim')
    print(feasibility2mad)
    print(analysis['density_min_start']['mean'])
    print('typicality')
    print(typicalitymad)
    print(analysis['density_foil']['mean'])

    record = load_foilsbeta()

    configuration = {
        "lambda_": 120.0,
        "mu": 2.25,
        "alpha": 0,
        "beta": 0.625,
        "budget": 1000,
        "densitycut": 8,
        "densityaddloss": 1.5,
        "densityscaler": 0.05
    }
    conf2 = json.dumps(configuration)
    full_set = []
    for data in record:
        if data['conf'] == conf2:
            # if data['factid'] in factset:
            full_set.append((data['fact'], data['foil'], np.array(data['history']["pdf"]), 4, 6))
    print('start analysis')
    analysis = analyze_foils(full_set, measures)
    print('passed')
    sparsitymad = analysis['sparsity']['mad']
    closenessmad = analysis['cocomix_distance']['mad']
    feasibility1mad = analysis['density_opt_step_min']['mad']
    feasibility2mad = analysis['density_min_start']['mad']
    typicalitymad = analysis['density_foil']['mad']

    print('sparsity')
    print(sparsitymad)
    print(analysis['sparsity']['mean'])
    print('closeness')
    print(closenessmad)
    print(analysis['cocomix_distance']['mean'])
    print('fesibilityper optimization step')
    print(feasibility1mad)
    print(analysis['density_opt_step_min']['mean'])
    print('fesibility min whole optim')
    print(feasibility2mad)
    print(analysis['density_min_start']['mean'])
    print('typicality')
    print(typicalitymad)
    print(analysis['density_foil']['mean'])


