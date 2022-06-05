import json
from demonstration.proxy_measures.analyze import analyze_foils, instantiate_all_measures
from demonstration.competing_approaches.wachter.util import calculate_mad
import os
import numpy as np
from demonstration.demonstration_data import load_df, FEATURES, VAR_TYPES
import pandas as pd
from demonstration.proxy_measures.sparsity import make_sparsity
from demonstration.density_estimation.compute_kde import kde, preprocess


df_train = load_df(train=True)
mad = calculate_mad(df_train[[feature for feature, var_type in zip(FEATURES, VAR_TYPES)
                                if var_type == "c"]].to_numpy())
mad[mad == 0.0] = 0.5
measures = instantiate_all_measures(mad)


PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PATH, 'datenstudie', 'studienfoils_final2.json'), "rt") as f:
    record = json.load(f)

foils=[]
cocomix_distance=[]
wachter_distance=[]
sparsities=[]
density_min_start=[]
density_foil=[]
density_opt_step_min=[]
density_opt_step_max=[]
correctness=[]
certifai_dist=[]
sparsity = make_sparsity(FEATURES)
def pdf(sample):
    return kde.pdf(preprocess(sample.reshape(1, -1)))
def L1(x,y):
    mad=np.array([29,41.38,2,0.5,289,1]) #Immo-spezifisch, sollte eig übergeben werden
    res=[]
    fact=np.asarray(x).astype(float)
    y.reset_index(inplace=True,drop=True)
    #print(y)
    for i in range(len(y[:][0])):
        foil=np.asarray(y.loc[i,:]).astype(float)
        res.append(np.sum(np.abs(fact-foil)/mad))
    #print(np.asarray(res))
    return np.asarray(res)


def Tab_distance(x, y):
    """Distance function for tabular data, as described in the original
    paper. This function is the default one for tabular data in the paper and
    in the set_distance function below as well. For this function to be used,
    the training set must consist of a .csv file as specified in the class
    instatiation above. This way, pandas can be used to infer whether a
    feature is categorical or not based on its pandas datatype and, as such, it is important that all columns
    in the dataframe have the correct datatype.

    Arguments:
        x (pandas.dataframe): the input sample from the training set. This needs to be
        a row of a pandas dataframe at the moment, but the functionality of this
        function will be extended to accept also numpy.ndarray.

        y (pandas.dataframe or numpy.ndarray): the comparison samples (i.e. here, the counterfactual)
        which distance from x needs to be calculated.

        continuous_distance (bool): the distance function to be applied
        to the continuous features. Default is L1 function.

        con (list): list of the continuous features (i.e. columns) names

        cat (list): list of the categorical features (i.e. columns) names
    """
    con = [i for i in range(len(FEATURES)) if VAR_TYPES[i] == "c"]
    cat = [i for i in range(len(FEATURES)) if VAR_TYPES[i] != "c"]
    #print(x)
    #print(y)
    if not isinstance(x, pd.DataFrame):
         x = pd.DataFrame(x).transpose()
    if not isinstance(y, pd.DataFrame):
         y = pd.DataFrame(y,columns=x.columns)
    #print(x)
    #print(y)
    if len(cat) > 0:
        cat_distance = len(cat) - (x[cat].values == y[cat].values).sum(axis=1)
    else:
        cat_distance = 1

    if len(con) > 0:
        con_distance = L1(x[con], y[con])
    else:
        con_distance = 1
    #print(len(con) / x.shape[-1] * con_distance + len(cat) / x.shape[-1] * cat_distance)
    #return len(con) / x.shape[-1] * con_distance + len(cat) / x.shape[-1] * cat_distance
    return con_distance


for data in record:
    if data["conf"] != "fake" and data["conf"] == "generations=3000":
        #if data['Wachter']==False:
        #set = []
        #set.append((data['fact'], data['foil'], np.array(data['history']["pdf"]), 4, 6))
        #analysis = analyze_foils(set, measures)
        #foils.append(data['foilid'])
        # cocomix_distance.append(analysis['cocomix_distance']['mean'])
        # wachter_distance.append(analysis['wachter_distance']['mean'])
        #correctness.append(analysis['correctness']['mean'])
        # print('done')
        #sparsities.append(analysis['sparsity']['mean'])
        sparsities.append(sparsity(data['fact'],data['foil']))
        #density_min_start.append(analysis['density_min_start']['mean'])
        #density_foil.append(analysis['density_foil']['mean'])
        #density_opt_step_min.append(analysis['density_opt_step_min']['mean'])
        #density_opt_step_max.append(analysis['density_opt_step_max']['mean'])
        fact=[]
        foil=[]
        for cat in FEATURES:
            fact.append(data['fact'][cat])
            foil.append(data['foil'][cat])
        density_foil.append(pdf(np.array(foil)))
        #certifai_dist.append(Tab_distance(np.array(fact), [np.array(foil)]))


# print(np.mean(cocomix_distance))
# print(np.mean(wachter_distance))
# print(np.mean(correctness))
print(np.mean(sparsities))
print(np.var(sparsities))
# print(np.mean(density_min_start))
# print(np.var(density_min_start))
print(np.mean(density_foil))
print(np.var(density_foil))
# print(np.mean(density_opt_step_min))
# print(np.var(density_opt_step_min))
# print(np.mean(density_opt_step_max))
# print(np.var(density_opt_step_max))
#print(np.mean(certifai_dist))
#print(np.var(certifai_dist))

