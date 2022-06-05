import pickle
import os
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
fname="20200623-002652_cocomix_foils.pickle"
with open(os.path.join(PATH, "..", "cocomix", fname), "rb") as f:
    report = pickle.load(f)
print(fname)
#print(report[6])
full_set = [(fact, foil, np.array(history["pdf"])) for fact, foil, history,_,_,_,_,_,_ in report]
#print((fact, foil) for fact, foil, _, _, _, _ in report)
print(analyze_foils(full_set, measures))

fname="20200622-155736_wachter_foils.pickle"
with open(os.path.join(PATH, "..", "cocomix", fname), "rb") as f:
    report = pickle.load(f)
print(fname)
#print(report)
full_set = [(fact, foil, np.array(history["pdf"])) for fact, foil, history,_,_,_,_,_,_ in report]
#print((fact, foil) for fact, foil, _, _, _, _ in report)
print(analyze_foils(full_set, measures))