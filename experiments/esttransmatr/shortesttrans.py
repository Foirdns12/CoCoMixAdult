import numpy as np

def likelipath(transmatrix):
    for i in range(0, len(transmatrix)):
        transmatrix[i][i] = 0
    return likelipath2(transmatrix,transmatrix) #hier kann auch noch die Whrscheinlichkeit in distanzen umgerechnet werden




def likelipath2(transmatrix,orig,it=1):
    if it <= len(transmatrix)-2:
        transmatrixa=transmatrix
        for i in range(0,len(transmatrix)):
            transmatrixa[i][i]=0
        dist=likelipath2(np.dot(transmatrixa,orig),orig,it+1)
    else:
        for i in range(0, len(transmatrix)):
            transmatrix[i][i] = 1
        return transmatrix
    distmatrix=np.maximum(transmatrix,dist)
    return distmatrix
