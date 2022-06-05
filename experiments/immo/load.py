import pickle
from lib.kde import initialize_pdf

def load_vorbereitung():
    f = open('store.pckl', 'rb')
    transmatrices, distmatrices, MAD, categorical_values,bw, var_types, features = pickle.load(f)
    #pdf = initialize_pdf(samplestrain, var_types, categorical_values, bw)
    return transmatrices, distmatrices, MAD, categorical_values, var_types, features