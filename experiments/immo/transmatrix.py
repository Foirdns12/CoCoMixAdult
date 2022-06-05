import numpy as np
import math

def esttransmatrix(X,pdf,features,var_types,a=1,b=1):
    Transmatrices={

    }
    for z in range(0,len(var_types)):
        if var_types[z] != "c":
            print(features[z])
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
                        transvectorbar.append(paa)
                    i = i + 1

                #print("Transvektorbar:")
                #print(transvectorbar)
                print(transvectorbar)
                transmatrix.append(transvectorbar)  # schreibe Vektor in Matrix
                #print("calculation for one string succeeded")
                #print(transvector)

            #print("transition matrix")
            #print(transmatrix)
            print(transmatrix)
            Transmatrices[features[z]]=transmatrix
    return(Transmatrices)