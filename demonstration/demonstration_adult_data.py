"""Load data for demonstration.

All data used in the demonstration is loaded through this module
to ensure consistency in the features, their type, and their order.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from data.adult import FEATURES,VAR_TYPES
from data import adult

assert len(VAR_TYPES) == len(FEATURES)
assert np.all([var_type != "c" for feature, var_type in zip(FEATURES, VAR_TYPES) if feature in adult.CATEGORICAL])
assert np.all([var_type == "c" for feature, var_type in zip(FEATURES, VAR_TYPES) if feature in adult.NUMERICAL])

TARGET = "label"

CATEGORICAL = [feature for feature in FEATURES if feature in adult.CATEGORICAL]
NUMERICAL = [feature for feature in FEATURES if feature in adult.NUMERICAL]

UNORDERED_CATEGORICAL_VALUES = {

    "workclass": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
                  "Never-worked"],
    "marital-status": ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed",
                       "Married-spouse-absent", "Married-AF-spouse"],
    "occupation": ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                   "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                   "Priv-house-serv", "Protective-serv", "Armed-Forces"],
    "relationship": ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Male", "Female"],
    "native-country": ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
                       "Outlying-US(Guam-USVI-etc)", "India",
                       "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
                       "Poland", "Jamaica", "Vietnam", "Mexico",
                       "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
                       "Columbia", "Hungary", "Guatemala", "Nicaragua",
                       "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong",
                       "Holand-Netherlands"]
}

ORDERED_CATEGORICAL_VALUES = {
      "education": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th",
                  "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]
}

ALL_CATEGORICAL_VALUES = {k: v for k, v in UNORDERED_CATEGORICAL_VALUES.items()}
ALL_CATEGORICAL_VALUES.update(ORDERED_CATEGORICAL_VALUES)

assert np.all([feature in ALL_CATEGORICAL_VALUES for feature in FEATURES if feature in CATEGORICAL])


def load_df(train=True,WithID=False):
    if WithID:
        _df=adult.load_df(columns=FEATURES + [TARGET], fillna=False)
        _samples = _df[FEATURES]
        _targets = _df[TARGET]
        samples, targets = _split_data(_samples, _targets, train)
        df = pd.DataFrame(samples, columns=FEATURES)

        df[TARGET] = targets
    else:
        _df = adult.load_df(columns=FEATURES + [TARGET], fillna=False)
        _samples = _df[FEATURES]
        _targets = _df[TARGET]
        samples, targets = _split_data(_samples, _targets, train)
        df = pd.DataFrame(samples, columns=FEATURES)
        df[TARGET] = targets
    return df


def load_data(train=True):
    _samples, _targets = adult.load_data(features=FEATURES, target=TARGET, fillna=False)
    samples, targets = _split_data(_samples, _targets, train)
    return samples, targets


def fillna(samples, method="median"):
    for i, var_type in enumerate(VAR_TYPES):
        if var_type == "c":
            column = samples[:, i].astype(np.float)
            if method == "median":
                value = np.median(column[~np.isnan(column)])
            elif method == "mean":
                value = np.mean(column[~np.isnan(column)])
            else:
                raise ValueError(f"Unknown method {method}")
            column[np.isnan(column)] = value
            samples[:, i] = column

    return samples


RANDOM_STATE = 323
TRAIN_SIZE = 0.80


def _split_data(_samples, _targets, train):
    if train:
        samples, _, targets, _ = train_test_split(_samples, _targets,
                                                  random_state=RANDOM_STATE, train_size=TRAIN_SIZE)
    else:
        _, samples, _, targets = train_test_split(_samples, _targets,
                                                  random_state=RANDOM_STATE, train_size=TRAIN_SIZE)
    return samples, targets


def _collect_categorical_values():
    df = load_df(train=True)
    values = {}
    for feature in FEATURES:
        if feature in CATEGORICAL:
            values[feature] = list(df[feature].unique())

    return values


def _compute_nan_rate():
    df = load_df(train=True)
    for feature in FEATURES:
        if feature in NUMERICAL:
            print(feature, len(df[feature][df[feature].isna()]) / len(df[feature]))


def fill_numerical_column_by_cond_median(source_df, condition_column, target_df, target_columns):
    conditional_values = list(target_df[condition_column].unique())
    for column in target_columns:
        column_values = target_df[column].to_numpy().astype(np.float)
        for value in conditional_values:
            sub_series = source_df[source_df[condition_column] == value][column].dropna()
            column_values[(np.isnan(column_values)) & (target_df[condition_column] == value)] = sub_series.median()
        assert np.all(~np.isnan(column_values))
        target_df[column] = column_values
    return target_df
