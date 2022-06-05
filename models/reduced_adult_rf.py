# see
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html#column-transformer-with-mixed-types
import joblib
import os

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from data.reduced_adult import load_data, CATEGORICAL, load_test


PATH = os.path.dirname(os.path.abspath(__file__))


def train_model():
    categorical_features = [feature_idx for feature_idx, is_categorical in enumerate(CATEGORICAL) if is_categorical]
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    scalar_features = [feature_idx for feature_idx, is_categorical in enumerate(CATEGORICAL) if not is_categorical]
    scalar_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[
        ('scalar', scalar_transformer, scalar_features),
        ('categorical', categorical_transformer, categorical_features)
    ])
    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=75))])
    X, Y = load_data()
    clf.fit(X, Y)
    X_test, Y_test = load_test()
    print(clf.score(X_test, Y_test))
    joblib.dump(clf, os.path.join(PATH, "reduced_adult_rf.joblib"))


def load_model():
    return joblib.load(os.path.join(PATH, "reduced_adult_rf.joblib"))


if __name__ == "__main__":
    train_model()
