import pandas as pd
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as st
PATH = os.path.dirname(os.path.abspath(__file__))


empir = pd.ExcelFile(os.path.join(PATH, "data_evaluation_filtered2.xlsx")).parse(0)
proxy = pd.ExcelFile(os.path.join(PATH, "proxysevalcombined.xlsx")).parse(0)
id=""
spars=[]
plau=[]
ty=[]
sui=[]
con=[]
proxspars=[]
proxdensavg=[]
sparsity = []
plaus = []
typ = []
suit = []
cons = []
for i in range(len(proxy["id"])):
    id =proxy['id'][i]
    #sparsity=[]
    #plaus=[]
    #typ=[]
    #suit=[]
    #cons=[]
    for j in range(len(empir["player.sparseness"])):
        if id == empir["player.foil_id"][j]:
            sparsity.append(empir["player.sparseness"][j])
            plaus.append(empir["player.plausibility1"][j]+empir["player.plausibility2"][j]+empir["player.plausibility3"][j])
            typ.append(empir["player.typicality1"][j]+empir["player.typicality2"][j]+empir["player.typicality3"][j])
            suit.append(empir["player.suitability"][j])
            cons.append(empir["player.conciseness"][j])
            proxspars.append(proxy['b'][i])
            proxdensavg.append(proxy['g'][i])
    #spars.append(np.mean(sparsity))
    #plau.append(np.mean(plaus))
    #ty.append(np.mean(typ))
    #sui.append(np.mean(suit))
    #con.append(np.mean(cons))
print(sparsity)
print(proxspars)
#print(np.corrcoef(spars, proxy["b"].to_numpy()))

#matplotlib.style.use('ggplot')
#plt.scatter(proxy["b"].to_numpy(),spars)
#plt.show()
#print(st.pearsonr(proxy["b"].to_numpy(),spars))
#print(st.linregress(proxy["b"].to_numpy(),spars))

matplotlib.style.use('ggplot')
plt.scatter(proxspars,sparsity)
plt.show()
print(st.pearsonr(proxspars,sparsity))
print(st.linregress(proxspars,sparsity))#falsche richtung
print(st.linregress(proxspars,cons)) #sig

matplotlib.style.use('ggplot')
plt.scatter(proxdensavg,plaus)
plt.show()
print(st.pearsonr(proxdensavg,plaus))
print(st.linregress(proxdensavg,plaus))


matplotlib.style.use('ggplot')
plt.scatter(proxdensavg,typ)
plt.show()
print(st.pearsonr(proxdensavg,typ))
print(st.linregress(proxdensavg,typ))


# #negative ergebnisse erwartet, nicht erfüllt
# #cocomixdist
# plt.scatter(proxy["c"].to_numpy(),con)
# plt.show()
# print(st.linregress(proxy["c"].to_numpy(),con))
#
# #wachter dist
# plt.scatter(proxy["d"].to_numpy(),con)
# plt.show()
# print(st.linregress(proxy["d"].to_numpy(),con))

