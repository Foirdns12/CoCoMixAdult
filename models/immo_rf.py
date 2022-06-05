import joblib
import os
from numpy import mean, absolute
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from data.immodata import load_data, CATEGORICAL

PATH = os.path.dirname(os.path.abspath(__file__))


def train_model(x_train, x_test, y_train, y_test):
    samples, targets, features = load_data(fillna=True)


    categorical_features = [feature_idx for feature_idx, feature in enumerate(features)
                            if feature in CATEGORICAL]
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    scalar_features = [feature_idx for feature_idx, feature in enumerate(features)
                       if feature not in CATEGORICAL]
    scalar_transformer = Pipeline(steps=[('scaler', StandardScaler())])

    preprocessor = ColumnTransformer(transformers=[
        ('scalar', scalar_transformer, scalar_features),
        ('categorical', categorical_transformer, categorical_features)
    ])
    reg = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestRegressor(n_estimators=10))])

    reg.fit(x_train, y_train)
    pred = reg.predict(x_test)
    print(reg.score(x_test, y_test))
    print("Mean average deviation")
    print(mean(absolute(pred-y_test)))

    joblib.dump(reg, os.path.join(PATH, "immo_rf.joblib"))


def load_model():
    return joblib.load(os.path.join(PATH, "immo_rf.joblib"))


if __name__ == "__main__":
    samples, targets, features = load_data(fillna=True)
    x_train, x_test, y_train, y_test = train_test_split(samples, targets, train_size=.7)
    train_model(x_train, x_test, y_train, y_test)
