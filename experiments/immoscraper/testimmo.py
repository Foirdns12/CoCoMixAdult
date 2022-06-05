from sklearn.model_selection import train_test_split
import pandas as pd
from numpy import mean, absolute
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import numpy as np

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler=StandardScaler()
df = pd.ExcelFile("C:/Users/phili/Desktop/MA/Datensatz/Immobereinigtv2.xlsx").parse(0)
dfy = df["obj_purchasePrice"].values.ravel()

#dfyenc=dfy



#imp=imp.fit(dftrain)
#dftrainna=imp.fit(dftrain)
#dftestna=imp.fit(dftest)
df["obj_lotArea"].fillna(df["obj_lotArea"].mean(),inplace=True)
df["obj_yearConstructed"].fillna(df["obj_yearConstructed"].mean(),inplace=True)
df["obj_noParkSpaces"].fillna(df["obj_noParkSpaces"].mean(),inplace=True)
df["obj_livingSpace"].fillna(df["obj_livingSpace"].mean(),inplace=True)
df["obj_noRooms"].fillna(df["obj_noRooms"].mean(),inplace=True)

df = df.drop(columns=["obj_firingTypes","obj_purchasePrice","obj_picturecount","obj_telekomDownloadSpeed","ga_cd_via","obj_courtage","obj_telekomUploadSpeed","obj_zipCode","obj_telekomDownloadSpeed","obj_telekomInternet","obj_numberOfFloors","obj_lastRefurbish"])

dftrain,dftest,dfytrain,dfytest = train_test_split(df,dfy,train_size=.7)

lab_enc = preprocessing.LabelEncoder()
dfytrainenc = lab_enc.fit_transform(dfytrain)

categorical_features = ["obj_regio1","obj_heatingType","obj_newlyConst","obj_cellar","geo_krs","obj_condition","obj_interiorQual","obj_buildingType","obj_barrierFree","obj_rented","obj_regio3"]
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
scalar_features = ["obj_lotArea","obj_yearConstructed","obj_noParkSpaces","obj_livingSpace","obj_noRooms"]
scalar_transformer = Pipeline(steps=[('scaler', StandardScaler())])
preprocessor = ColumnTransformer(transformers=[
    ('scalar', scalar_transformer, scalar_features),
    ('categorical', categorical_transformer, categorical_features)
])
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=10))])

clf.fit(dftrain, dfytrainenc)


pred=clf.predict(dftest)
preddecode=lab_enc.inverse_transform(pred)
#print(clf.score(dftest, dfytest))
print(mean(absolute(preddecode-dfytest)))