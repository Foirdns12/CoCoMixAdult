import os
import pickle
import numpy as np
from demonstration.demonstration_data import ALL_CATEGORICAL_VALUES

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, "tuned_transition_matrices_adult.pickle"), "rb") as f:
    transition_matrices = pickle.load(f)

for feature, values in ALL_CATEGORICAL_VALUES.items():
    try:
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)
    except KeyError:
        print("No transition matrix for", feature)

# Prevent transition to "no_information"

for feature, values in ALL_CATEGORICAL_VALUES.items():
    if feature not in transition_matrices:
        print(f"No transition matrix found for {feature}, skipping.")
        continue

    print(values)
    if "no_information" in values:
        idx = values.index("no_information")
        print("no_information index", idx)
        transition_matrices[feature][:idx, idx] = 0.0
        transition_matrices[feature][idx + 1:, idx] = 0.0
'''
# nur semi-strikt geordnete wechsel
condition = transition_matrices['obj_condition']
cond_vals = ALL_CATEGORICAL_VALUES["obj_condition"]
# no information -> well-kept , need-of-renovation, refurbished, fully-renovated
condition[0,0]= 0.5
condition[0, 1:4] = 0.0
condition[0,4]=0.01
condition[0, 5] = 0.0
condition[0,6]=0.02
condition[0,7]=0.14
condition[0,8]=0.05
condition[0, 9:] = 0.0
# mint_condition -> first time use, modernized
condition[1, 4:] = 0.0
#  mint_condition <- first_time_use -> modernized, fully_renovated
condition[2, 5:] = 0.0
# first_time_use, mint_condition <- modernized -> fully_renovated, first_time_use_after_ref
condition[3,0]=0.0
condition[3, 6:] = 0.0
# modernized <- fully_renovated -> first_time_use_after_ref, ref, well-kept
condition[4, :3] = 0.0
condition[4, 8:] = 0.0
# fully_renovated <- first_time_use_after_ref -> ref, well-kept
condition[5, :4] = 0.0
condition[5, 8:] = 0.0
# fully_renovated, first_time_use_after_ref <- ref -> well-kept
condition[6, :4] = 0.0
condition[6, 8:] = 0.0
# fully_renovated, first_time_use_after_ref, ref <- well-kept -> need_of_ren, neg
condition[7, :4] = 0.0
condition[7, 10:] = 0.0
# fully_renovated, first_time_use_after_ref, ref, well-kept <- need of renov -> neg, ripe for
condition[8, :4] = 0.0
# well_kept, need-of-renovation <- neg -> ripe-for-demolishion
condition[9, :7] = 0.0
# neg, need for renov <- ripe for
condition[10, :8] = 0.0

# nur strikt geordnete wechsel, transition von no_information zu allen möglich, wechsel zurück möglich
interior_qual = transition_matrices['obj_interiorQual']
iq_vals = ALL_CATEGORICAL_VALUES["obj_interiorQual"]
interior_qual[0,2]=0.1
# for i, val in enumerate(iq_vals):
#     if i > 1:
#         interior_qual[i, :i - 2] = 0.0
#     if i>0:
#         interior_qual[i, i + 2:] = 0.0

# nie von Altbau zu Neubau, erzeugt sehr häufig unsinnige Foils
#newly_const = transition_matrices["obj_newlyConst"]
#nc_vals = ALL_CATEGORICAL_VALUES["obj_newlyConst"]
#newly_const[nc_vals.index("n")][nc_vals.index("y")] = 0.0
'''
native_country = transition_matrices['native-country']
native_country[40,:]=1/40
#können das nicht einfach rausnehmen aber im modell behalten, sonst wird es exterm schwer foils zu finde, können es nur ganz aus dem Modell löschen
print("")
for feature, values in ALL_CATEGORICAL_VALUES.items():
    print(feature)
    try:
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)
            assert np.sum(row) > 0.0
    except KeyError:
        print("No transition matrix for", feature)

print(transition_matrices)
with open(os.path.join(PATH, "final_transition_matrices_adult.pickle"), "wb") as f:
    pickle.dump(transition_matrices, f)
