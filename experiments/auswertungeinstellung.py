import pickle
import os
from demonstration.demonstration_immo_model import load_model
from demonstration.demonstration_data import FEATURES
from demonstration.density_estimation.compute_kde import kde, preprocess
import numpy as np
model=load_model()
PATH = os.path.dirname(os.path.abspath(__file__))
prederror=0
diferror=0
densitycount=0
for i in (4,5,16,31,44,51):#(2,3,8,16,19,23):
    print(i)
    with open(os.path.join(PATH, "init37generatedfoils"+str(i)+".pickle"), "rb") as f:
        data8 = pickle.load(f)
    with open(os.path.join(PATH, "init38generatedfoils" + str(i) + ".pickle"), "rb") as f:
        data9 = pickle.load(f)
    #print(data8[1])
    #with open(os.path.join(PATH, "init17generatedfoils"+str(i)+".pickle"), "rb") as f:
    #    data9 = pickle.load(f)
    print(data9[1])
    #p_sample = {feature: np.array([[value]]) for feature, value in zip(FEATURES, data9[1])}
    #print(np.argmax(model.predict(p_sample)[0]))
    print(data8[1])
    print(data9[0])
    print(data8[0])
    # print("difference in prediction error")
    # print(data8[4][1]-data9[4][1])
    # prederror=prederror+data8[4][1]-data9[4][1]
    # print("difference in distace")
    # print(data8[4][2]-data9[4][2])
    # diferror=diferror+data8[4][2]-data9[4][2]
    # print("ratio of density")
    #foil8 = preprocess(np.array(data8[1]).reshape(1,-1))
    #foil9 = preprocess(np.array(data9[1]).reshape(1, -1))
    #fact=preprocess(np.array(data9[0]).reshape(1, -1))
    #print(kde.pdf(foil8.reshape(1,-1)))
    #print(kde.pdf(foil8.reshape(1,-1))/kde.pdf(fact.reshape(1,-1)))
    #print(kde.pdf(foil9.reshape(1, -1)) / kde.pdf(fact.reshape(1, -1)))
    #if kde.pdf(foil8.reshape(1,-1)) < kde.pdf(foil9.reshape(1, -1)):
     #   densitycount=densitycount-1
    #if kde.pdf(foil8.reshape(1,-1)) > kde.pdf(foil9.reshape(1, -1)):
    #    densitycount=densitycount+1

# print("Gesamt")
# print("difference in prediction error")
# print(prederror)
# print("difference in distace")
# print(diferror)
# print("density")
# print(densitycount)

