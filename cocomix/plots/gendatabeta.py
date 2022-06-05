from demonstration.cocomix.compute_foil_immo import calculate_foils
import os
import datetime


if __name__ == "__main__":
    PATH = os.path.dirname(os.path.abspath(__file__))
    import json
    import numpy as np


    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(NpEncoder, self).default(obj)



    #x1 = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25]
    x1 = [2.5]
    x2 = [0.125,0.25,0.375, 0.5,0.625, 0.75,0.875, 1]
    rands = [12,12,30,45,55,65,88]
    numbers=[1,5,5,3,5,3,3]
    i=0
    for rand in rands:
        foilsettot = []
        number=numbers[i]
        i=i+1
        for mu in x1:
            print('##################################################')
            print(mu)
            print('##################################################')
            for beta in x2:
                configuration = {
                    "lambda_": 120.0,
                    "mu": mu,
                    "alpha": 0,
                    "beta": beta,
                    "budget": 1000,
                    "densitycut": 8,
                    "densityaddloss": 1.5,
                    "densityscaler": 0.05
                }
                result, foilset = calculate_foils(configuration, n=number, randomstate=rand,metrics=False,Wachter2=True)
                foilsettot = foilsettot + foilset


        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fnamejsn = f"{current_time+str(rand)}_betafoils.json"
        with open(os.path.join(PATH,'daten',fnamejsn), "wt") as f:
            json.dump(foilsettot, f, indent=4, cls=NpEncoder)
        print('saved')
        print(rand)