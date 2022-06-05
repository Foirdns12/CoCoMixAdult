import nevergrad as ng
from random import random,randrange

#TODO: Mglw anders lösen als mit Rekursion

def startoptimization(decfct,ngvariables,n,parent,maxstep,fixn=True):
    optval=decfct(*list(parent),print=True)
    if fixn:
        foil = optimizefix(decfct, ngvariables, n, parent, maxstep, optval) #es werden immer n variablen geändert
    else:
        foil=optimize(decfct,ngvariables,n,parent,maxstep,optval)  #jede variable wird mit wkt n/anzahl variablen geändert, könnte bei n größer 1 besser funktionieren
    return foil


def optimize(decfct,ngvariables,n,parent,optstep,optval):
    if optstep == 0:
        return parent
    i=0
    mutation=parent.copy()
    for var in ngvariables:
        ngvariables[var].value=parent[i]
        if random() <= n/len(parent):
            ngvariables[var].mutate()
            mutation[i]=ngvariables[var].value
            print('mutation')
            print(mutation)
            print('parent')
            print(parent)
        i=i+1
    if decfct(*list(mutation))<=optval:
        print('new')
        optval= decfct(*list(mutation))
        return(optimize(decfct=decfct,ngvariables=ngvariables,n=n,parent=mutation,optstep=(optstep-1),optval=optval))
    else:
        return(optimize(decfct=decfct,ngvariables=ngvariables,n=n,parent=parent,optstep=(optstep-1),optval=optval))


def optimizefix(decfct,ngvariables,n,parent,optstep,optval):
    if optstep == 0:
        return parent

    mutatvar = set()
    while len(mutatvar) < n:
        number = randrange(0, len(parent))
        if number not in mutatvar:
            mutatvar.add(number)
    print(mutatvar)
    i=0
    mutation=parent.copy()
    for var in ngvariables:
        print(var)
        ngvariables[var].value=parent[i]
        if i in mutatvar:
            ngvariables[var].mutate()
            mutation[i]=ngvariables[var].value
            print('mutation')
            print(mutation)
            print('parent')
            print(parent)
        i=i+1
    if decfct(*list(mutation))<=optval:
        print('new')
        optval= decfct(*list(mutation))
        return(optimizefix(decfct=decfct,ngvariables=ngvariables,n=n,parent=mutation,optstep=(optstep-1),optval=optval))
    else:
        return(optimizefix(decfct=decfct,ngvariables=ngvariables,n=n,parent=parent,optstep=(optstep-1),optval=optval))



