import pandas as pd
from lib.kde import initialize_pdf
from data.reduced_adult import load_data, FEATURES, VALUES, load_test
import numpy as np
import math
#df = pd.ExcelFile("C:/Users/phili/Desktop/MA/Datensatz/Immobereinigtv2.xlsx").parse(0)

#df = df.drop(columns=["obj_firingTypes","obj_picturecount","obj_telekomDownloadSpeed","ga_cd_via","obj_courtage","obj_telekomUploadSpeed","obj_zipCode","obj_telekomDownloadSpeed","obj_telekomInternet","obj_numberOfFloors","obj_lastRefurbish"])

X, _,_ = load_data()

var_types = np.array(["c", "u", "o", "u", "u", "u", "c", "c"])
assert len(var_types) == len(FEATURES)

categorical_values = [VALUES[feature] + ["?"] for idx, feature in enumerate(FEATURES)
                      if var_types[idx] != "c"]
bw = [0.1, 0.001, 0.001, 0.001, 0.001, 0.55, 0.005, 0.05]#bandwith für workclass unda auch allgemein sehr klein, deshlab die großen unterschiede und oftmals gleichen werte

pdf = initialize_pdf(X, var_types, categorical_values, bw)
a=0.5
b=0.5
#testbeispielBildung
z=2 #über z iterieren um transition Matrix für alle kategorischen variablen zu erhalten, z gibt den Spaltenwert an also nur Spalten zulassen, die auch kategorische Variablen sind
values = np.unique(X[:, z])  # Wertebereich der kategorsichen Variable
#print(values)
transmatrix = []
k = 0
for str in values:  # iteriert über alle möglichen Werte der kategorischen Variable in Spalte z
    sample = []
    pdfs = []
    pdfssij = []
    transvectorbar = []
    for i in range(0,
                   len(X)):  # für alle Datenpunkt, die den Wert str in der kategorischen Variable in Spalte z haben
        if X[i, z] == str:
            sample.append(
                X[i])  # schreibe in array für später, damit nicht immer alle datenpunkte iteriert werden müssen
            pdfs.append(pdf(X[i]))  # berechne die dichte der datenpunkte
    pdfssii = np.mean(pdfs)  # pdf_aa in artifact V7
    for str2 in values:
        pdfsij = []
        if str2 != str:
            for i in range(0, len(sample)):
                xbar = sample[i]
                xbar[
                    z] = str2  # ändere den Wert der kategorischen Variable in Spalte z und messe dann die Dichte dieses künstlichen Datenpunkts
                pdfsij.append(pdf(
                    xbar))  # die pdf wird nie null sondern ist immer minimal ein Wert mit e-5, liegt vllt daran, dass ursprünglicher Punkt immer in Reichweite
            pdfssij.append(np.mean(pdfsij))  # pdf_ai in artifact v7 für alle alternativen klassen
    #print("step 1 succeeded")
    i = 0

    paa = (2 / (1 + math.exp(-b * (pdfssii / (np.mean(pdfssij)))))) - 1  # berechne p'_aa
    for str2 in values:
        if str2 != str:
            pibar = (1-paa)*(np.power(pdfssij[i],a)) / (np.sum(np.power(pdfssij,a)))  # hier noch hoch a einfügen
            transvectorbar.append(
                pibar)  # berechne p'_ai für alle alternativen Klassen und schreibe sie in ein array, aktuell ergibt noch die Summe aller p'_ai 1, hier mglw p'_aa mit einbeziehen und direkt normieren und dafür p'_aa anpassen
        if str2 == str:
            i = i - 1
            pibar = (2 / (1 + math.exp(-b * (pdfssii / (np.mean(pdfssij)))))) - 1  # berechne p'_aa
            transvectorbar.append(pibar)
        i = i + 1

    #print("Transvektorbar:")
    #print(transvectorbar)
    print(transvectorbar)
    transmatrix.append(transvectorbar)  # schreibe Vektor in Matrix
    #print("calculation for one string succeeded")
    #print(transvector)

print(transmatrix)

