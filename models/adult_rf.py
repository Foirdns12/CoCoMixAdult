import os
import pandas as pd

from data.adult import  NUMERICAL, CATEGORICAL
from demonstration.demonstration_data import  load_data

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer

PATH = os.path.dirname(os.path.abspath(__file__))

#Random Forest with Accuracy 82,02%

# Importing dataset
df_train_tuple = load_data(train=True)
df_test_tuple = load_data(train=False)

X_train = pd.DataFrame(df_train_tuple[0],columns=["age","fnlwgt","education-num","hours-per-week","workclass","education","marital-status","occupation","relationship","race","sex", "native-country"])
Y_train = pd.DataFrame(df_train_tuple[1], columns = ["label"])

df_train = X_train.copy()
df_train["label"] = Y_train.copy()
df_train[["age","fnlwgt","education-num","hours-per-week"]] = df_train[["age","fnlwgt","education-num","hours-per-week"]].apply(pd.to_numeric)


X_test = pd.DataFrame(df_test_tuple[0],columns=["age","fnlwgt","education-num","hours-per-week","workclass","education","marital-status","occupation","relationship","race","sex", "native-country"])
Y_test = pd.DataFrame(df_test_tuple[1], columns=["label"])

df_test = X_test.copy()
df_test["label"] = Y_test.copy()
df_test[["age","fnlwgt","education-num","hours-per-week"]] = df_test[["age","fnlwgt","education-num","hours-per-week"]].apply(pd.to_numeric)


X = pd.concat([X_train,X_test])
Y = pd.concat([Y_train,Y_test])

X_train = X.head(26048)
Y_train = Y.head(26048)

X_test = X.tail(6513)
Y_test = Y.tail(6513)

numeric_features = NUMERICAL
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)


categorical_features = ["workclass",
"education",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"native-country"]

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


model = Pipeline(
    steps=[("preprocessor", preprocessor), ('rfc', RandomForestClassifier(max_depth = 102, n_estimators = 40, random_state = 42))]
)

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)


#print('Accuracy score:',round(accuracy_score(Y_test, Y_pred) * 100, 2))#
#print('F1 score:',round(f1_score(Y_test, Y_pred) * 100, 2))

#cm = print(confusion_matrix(Y_test, Y_pred))


#joblib.dump(model, "../model_instances/adult_rf.joblib")



