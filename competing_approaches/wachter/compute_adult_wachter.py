import datetime
import os
import numpy as np
import pandas as pd


from demonstration.competing_approaches.wachter.util import calculate_mad
from demonstration.demonstration_adult_data import load_df, fill_numerical_column_by_cond_median, FEATURES, VAR_TYPES
from models.adult_rf import df_train,df_test
from demonstration.cocomix.compute_foil_adult import calculate_foils
from demonstration.transition_matrices.unit_matrices import get_unit_distance_matrices, get_unit_transition_matrices
from model_instances import adult_model

PATH = os.path.dirname(os.path.abspath(__file__))
WACHTER = False

# Prepare data

#df_train = load_df(train=True)
#df_test = load_df(train=False,WithID=True)
df_train = df_train
df_test = df_test
assert not np.any([np.any(df_test[column].isna()) for column in df_test.columns])

mad = calculate_mad(df_train[[feature for feature, var_type in zip(FEATURES, VAR_TYPES)
                              if var_type == "c"]].to_numpy())

# TODO: How to fix this?? MAD is 0 for one feature
mad[mad == 0.0] = 0.5

#Load remaining components

model = adult_model.load_model()

transition_matrices = get_unit_transition_matrices()
distance_matrices = get_unit_distance_matrices()


configuration= {
    "lambda_": 120.0,
    "mu": 2.25,
    "alpha": 0.0,
    "beta": 0.625,
    "budget": 1000,
    "densitycut": 8,
    "densityaddloss": 1.5,
    "densityscaler": 0.05
}


if __name__ == "__main__":
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


    rand=60
    for i in range(3):
        result, foilset = calculate_foils(configuration,mad,df_test,model,transition_matrices,distance_matrices,df_train,n=1,factset=None,randomstate=rand,metrics=True)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fnamejsn = f"{current_time+str(rand)}_foilshs_wachter.json"
        with open(os.path.join(PATH,'datenstudie',fnamejsn), "wt") as f:
            json.dump(foilset, f, indent=4, cls=NpEncoder)