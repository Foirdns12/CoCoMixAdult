import json
from demonstration.proxy_measures.analyze import analyze_foils, instantiate_all_measures
from demonstration.competing_approaches.wachter.util import calculate_mad
import os
import numpy as np
from demonstration.demonstration_data import load_df, FEATURES, VAR_TYPES
import pandas as pd


df_train = load_df(train=True)
mad = calculate_mad(df_train[[feature for feature, var_type in zip(FEATURES, VAR_TYPES)
                                if var_type == "c"]].to_numpy())
mad[mad == 0.0] = 0.5
measures = instantiate_all_measures(mad)


PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, 'datenstudie', 'studienfoils.json'), "rt") as f:
    record = json.load(f)

foils=[]
cocomix_distance=[]
cocomix_distance2=[]
cocomix_distance3=[]
cocomix_distance4=[]
wachter_distance=[]
sparsity=[]
density_min_start=[]
density_foil=[]
density_opt_step_min=[]
density_opt_step_max=[]


for data in record:
    if data["conf"] != "fake":
        set = []
        set.append((data['fact'], data['foil'], np.array(data['history']["pdf"]), 4, 6))
        analysis = analyze_foils(set, measures)
        foils.append(data['foilid'])
        cocomix_distance.append(analysis['cocomix_distance']['mean'])
        cocomix_distance2.append(analysis['cocomix_distance2']['mean'])
        cocomix_distance3.append(analysis['cocomix_distance3']['mean'])
        cocomix_distance4.append(analysis['cocomix_distance4']['mean'])
        wachter_distance.append(analysis['wachter_distance']['mean'])
        sparsity.append(analysis['sparsity']['mean'])
        density_min_start.append(analysis['density_min_start']['mean'])
        density_foil.append(analysis['density_foil']['mean'])
        density_opt_step_min.append(analysis['density_opt_step_min']['mean'])
        density_opt_step_max.append(analysis['density_opt_step_max']['mean'])
matches = pd.DataFrame(list(zip(foils, cocomix_distance,cocomix_distance2,cocomix_distance3,cocomix_distance4, wachter_distance, sparsity, density_min_start, density_foil, density_opt_step_min, density_opt_step_max)),
                       columns=['foil_id', 'cocomix_distance','cocomix_distance2','cocomix_distance3','cocomix_distance4', 'wachter_distance', 'sparsity', 'density_min_start', 'density_foil', 'density_opt_step_min', 'density_opt_step_max'])
matches.to_csv('metrics.csv')