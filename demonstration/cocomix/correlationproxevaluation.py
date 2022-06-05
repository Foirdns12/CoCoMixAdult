import pickle
import os
import pandas as pd
PATH = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from demonstration.proxy_measures.analyze import analyze_foils, instantiate_all_measures
from demonstration.competing_approaches.wachter.util import calculate_mad
from demonstration.demonstration_data import load_df, fill_numerical_column_by_cond_median, FEATURES, VAR_TYPES, \
    ALL_CATEGORICAL_VALUES
df_train = load_df(train=True)
mad = calculate_mad(df_train[[feature for feature, var_type in zip(FEATURES, VAR_TYPES)
                              if var_type == "c"]].to_numpy())

mad[mad == 0.0] = 0.5
measures = instantiate_all_measures(mad)
#for fname in os.listdir(os.path.join(PATH, "..", "cocomix")):
fname="20200622-155736_wachter_foils.pickle"
#20200623-002652_cocomix_foils.pickle
with open(os.path.join(PATH, "..", "cocomix", fname), "rb") as f:
    report = pickle.load(f)
print(fname)
#print(report[6])
ids=[(str(ids)) for _,_,_,_,_, ids,_,_,_ in report]
full_set = [(fact, foil, np.array(history["pdf"])) for fact, foil, history, _, _, _,_,_,_ in report]
#print((fact, foil) for fact, foil, _, _, _, _ in report)
values=analyze_foils(full_set, measures)
#print(ids)
#print(values['sparsity']['all'])
dta=[]
for i in range(len(ids)):
    #dta.append((ids[i],values['sparsity']['all'][i],values['cocomix_distance']['all'][i],values['wachter_distance']['all'][i],values['integrated_density']['all'][i],values['density_min_start']['all'][i],values['density_average']['all'][i]))
    dta.append((ids[i],values['density_min_start']['all'][i],values['density_foil']['all'][i]))
print(dta)
dta2=pd.DataFrame(dta)
dta2.to_csv(os.path.join(PATH, "proxysevalwachtert.csv"))

#fname="20200622-155736_wachter_foils.pickle"
#with open(os.path.join(PATH, "..", "cocomix", fname), "rb") as f:
 #   report = pickle.load(f)
#print(fname)
#print(report)
#full_set = [(fact, foil, np.array(history["pdf"])) for fact, foil, history,_,_,id,_,_,_ in report]
#print((fact, foil) for fact, foil, _, _, _, _ in report)
#print(analyze_foils(full_set, measures))