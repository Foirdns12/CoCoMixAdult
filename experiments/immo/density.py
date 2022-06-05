import numpy as np
from experiments.esttransmatr.shortesttrans import likelipath
import pickle
from demonstration.demonstration_immo_data import load_data, FEATURES, VAR_TYPES, CATEGORICAL
from demonstration.density_estimation.bandwidths import final_bandwidths as bw
from demonstration.transition_matrices.final_matrices import load_transition_matrices
import math
import tensorflow as tf
from tensorflow.python.ops import math_ops


# in dieses script sind alle vorbereitenden Maßnahmen, die vor dem start der genereirung der counterfactuals erfolgen müssen, danach jedoch nicht mehr notwendig sind

# TODO: Wie schätzen wir die adjustment stength für numerische Variablen (bei numerischen Variablen schätzung aus varianz mgl)

samplestrain, targetstrain = load_data(train=True)

#print(len(VAR_TYPES), len(FEATURES))
assert len(VAR_TYPES) == len(FEATURES)
assert len(VAR_TYPES) == len(bw)
assert np.all([var_type != "c" for feature, var_type in zip(FEATURES, VAR_TYPES) if feature in CATEGORICAL])

VALUES = {

}
MAD = {

}

for i, var_type in enumerate(VAR_TYPES):
    if var_type != "c":
        val = np.unique(samplestrain[:, i])
        name = FEATURES[i]
        VALUES[name] = val
    else:
        x=[value for value in samplestrain[:, i] if not math.isnan(value)]
        med = np.median(x)
        MAD[FEATURES[i]] = np.median(
            np.abs(x - med))  # stats.median_absolute_deviation(samplestest[:,i]) wirft sinnlosen fehler
        if MAD[FEATURES[i]]==0:
            MAD[FEATURES[i]]=np.mean(np.abs(x - med))



categorical_values = [VALUES[feature] for idx, feature in enumerate(FEATURES)
                      if VAR_TYPES[idx] != "c"]


if __name__ == "__main__":
    transmatrices=load_transition_matrices()

    distmatrices = {

    }
    reflength=0
    for i in range(0,7):
        reflength=np.maximum(reflength,len(categorical_values[i]))
    for j in transmatrices:
        dist = likelipath(transmatrices[j])
        dist = (dist*(len(dist)-1))/(reflength-1)
        for i in range(0,len(dist)):
            dist[i,i]=1
        distmatrices[j]=dist
    print(distmatrices)
    with open('store.pckl', 'wb') as f:
        pickle.dump(
            [transmatrices, distmatrices, MAD, categorical_values,bw, VAR_TYPES,FEATURES], f)


