import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from lib.kde import initialize_pdf
from demonstration.demonstration_data import load_data

samples, targets = load_data()
for i in range(0,14):
    print(len(np.unique(samples[:,i])))


#f = open('store.pckl', 'rb')
#transmatrices, distmatrices, MAD, categorical_values, samplestrain, samplestest, targetstest, var_types, bw, features = pickle.load(f)

#samplestrain,samplestest,targetstrain,targetstest = train_test_split(samples,targets,train_size=.7)
#bw = [0.001, 0.001, 0.023, 0.11, 0.4, 0.001, 0.0128, 0.9976076555, 0.001, 0.001, 0.152, 0.001, 0.001, 0.001]
#print(len(samplestrain))
#pdf = initialize_pdf(samplestrain, var_types, categorical_values, bw)
#print(samplestest[2])
#print(pdf(samplestest[2]))