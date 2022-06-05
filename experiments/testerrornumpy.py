import numpy as np
import pandas as pd
from demonstration import demonstration_immo_data
from demonstration.density_estimation.compute_kde import kde, preprocess
from demonstration.demonstration_data import load_df, fill_numerical_column_by_cond_median
from demonstration.demonstration_immo_model import load_model_from_weights

samples, targets = demonstration_immo_data.load_data(train=False)

# print("Das Sample enthält sowohl Strings als auch numerische Daten:")
# print(samples[3])
# preprocess(samples[3].reshape(1,-1))
#
# print("Bevor etwas ausgerechnet werden kann, muss es erstmal durch's Pre-Processing:")
# samples = preprocess(samples)
# print(samples[3])
#
# print("Dann kann man auch überprüfen, ob und wo es NaNs enthält:")
# print(np.isnan(samples[3]))
#
# print("Und man kann dann auch die Dichte berechnen:")
# print(kde.pdf(samples[3].reshape(1, -1)))

#sampls=pd.DataFrame(samples,index=samples[:,0])
#print(sampls.iloc[3].todict())

df_test = load_df(train=False)
df_train = load_df(train=True)

df_test = fill_numerical_column_by_cond_median(source_df=df_train,
                                               condition_column="obj_buildingType",
                                               target_columns=["obj_numberOfFloors", "obj_noParkSpaces"],
                                               target_df=df_test)
print(df_test)


samples,_=demonstration_immo_data.load_data(train=False)
samplestr,_=demonstration_immo_data.load_data(train=True)

samples=fill_numerical_column_by_cond_median(source_df=samplestr,
                                               condition_column="obj_buildingType",
                                               target_columns=["obj_numberOfFloors", "obj_noParkSpaces"],
                                               target_df=samples)
model=load_model_from_weights()
num=3
sample=samples[2]
tsample= df_test.iloc[num]
#tfact.iloc[0]=1999
#tsample[0] = float(sample[0])
# tfact[1]=float(129)
# tfact[2]=float(6)
# tfact[3]=float(2)
# tfact[4]=float(689)
# tfact[5]=float(2)
# tfact[6]='villa'

print(tsample)
for i in range(0, len(sample)):
    print(i)
    print(sample[i])
    if i <= 5:
        tsample.iloc[i] = float(sample[i])
    else:
        tsample.iloc[i] = str(sample[i])
tsample[5]=float(3)
print(tsample)
tfact = tsample.to_dict()
tfact = {k: np.array([[v]]) for k, v in tfact.items()}
target=np.zeros(8)
target[3]=1
print(model.predict(tfact)[0])
first = (1 - np.sum(model.predict(tfact)[0]*target))
print(first)

