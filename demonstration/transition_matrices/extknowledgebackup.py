import os
import pickle

from demonstration.demonstration_data import ALL_CATEGORICAL_VALUES

PATH = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PATH, "tuned_transition_matrices.pickle"), "rb") as f:
    transition_matrices = pickle.load(f)

for feature, values in ALL_CATEGORICAL_VALUES.items():
    if feature!="geo_krs":
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)

# Prevent transition to "no_information"

for feature, values in ALL_CATEGORICAL_VALUES.items():
    if feature not in transition_matrices:
        print(f"No transition matrix found for {feature}, skipping.")
        continue
    print(values)
   # if "no_information" in values:
     #   idx = values.idx("no_information")

      #  transition_matrices[feature][:idx, idx] = 0.0
       # transition_matrices[feature][idx + 1:, idx] = 0.0

cond=transition_matrices['obj_condition']  #kein wechsel von demolition zu allem anderen außer need of renovation mgl
for i in range(1,8):                        # kein wechsel zu demolition außer von no information oder need of renovation
    cond[10,i]=0
    cond[i,10]=0
cond[10,9]=0
cond[9,10]=0
transition_matrices['obj_condition']=cond

intqual=transition_matrices['obj_interiorQual']  #keine wechsel zwischen simple und luxus
intqual[1][4]=0
intqual[4][1]=0

transition_matrices['obj_interiorQual']=intqual

print(transition_matrices)

for feature, values in ALL_CATEGORICAL_VALUES.items():
    if feature!="geo_krs":
        for row, value in zip(transition_matrices[feature], values):
            print(value, row)


#print(transition_matrices)
#with open(os.path.join(PATH, "final_transition_matrices.pickle"), "wb") as f:
#    pickle.dump(transition_matrices, f)
