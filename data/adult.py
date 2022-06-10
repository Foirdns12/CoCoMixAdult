import os

import numpy as np
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))

FEATURES = [
"age",
"fnlwgt",
"education-num",
"hours-per-week",
"workclass",
"education",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"native-country"]

VAR_TYPES = [
    'c',
    'c',
    'c',
    'c',
    'u',
    'o',
    'u',
    'u',
    'u',
    'u',
    'u',
    'u']


COLUMNS = ["age",
"workclass",
"fnlwgt",
"education",
"education-num",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"capital-gain",
"capital-loss",
"hours-per-week",
"native-country"]

CATEGORICAL = ["workclass",
"education",
"marital-status",
"occupation",
"relationship",
"race",
"sex",
"native-country",
"label"]

NUMERICAL = ["age",
"fnlwgt",
"education-num",
"hours-per-week"]

ALL = NUMERICAL + CATEGORICAL

def load_df(columns=None, fillna="mean"):
    """
    :param columns: List of columns to keep
    :param fillna: If not False, method to replace NaNs in numerical columns.
    :return: pd.DataFrame with the specified columns
    """
    columns = columns or ALL

    df = pd.read_csv(os.path.join(PATH, "adultdata", "adult.txt"), sep=",",header=None)
    df.columns = COLUMNS + ["label"]
    df = df[["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week","workclass","education","marital-status","occupation","relationship","race","sex", "native-country","label"]]

    df= df.drop(columns=["capital-gain","capital-loss"])

    for column in df:
        if df[column].dtype == "object":
           df[column] = df[column].str.strip()

    columns = NUMERICAL + CATEGORICAL

    assert np.all([col in columns for col in df.columns])

    if not np.all([col in df.columns for col in columns]):
        missing_columns = [col for col in columns if col not in df.columns]
        raise ValueError(f"Not all columns are in the loaded DataFrame. Missing columns: {missing_columns}")

    df = df[columns]


    if fillna:
        for column in df.columns:
            if column not in CATEGORICAL:
                if fillna == "median":
                    df[column].fillna(df[column].median(), inplace=True)
                elif fillna == "mean":
                    df[column].fillna(df[column].mean(), inplace=True)
                else:
                    raise ValueError(f"Unknown fillna method {fillna}")

    for feature, var_types in zip(FEATURES, VAR_TYPES):
        if var_types == "u" or var_types == "o":
            df[feature] = df[feature].str.strip()
    df["label"] = df["label"].str.strip()

    df = df.replace('?', np.nan)
    columns_with_nan = ['workclass', 'occupation', 'native-country']
    for column in columns_with_nan:
        if column in CATEGORICAL:
            df[column].fillna(df[column].mode()[0], inplace=True)
        if column not in CATEGORICAL:
            df[column].fillna(df[column].dropna().median(), inplace=True)


    return df

def load_data(features=FEATURES, target="label", fillna="mean"):
    """

       :param features: List of columns to use as features
       :param target: Name of column to use as the target
       :param fillna: see `load_df` for details
       :return: Samples as np.ndarray of size (n_samples, n_features),
                Targets as np.ndarray of size (n_samples,)
       """
    columns = NUMERICAL + CATEGORICAL + ["label"]

    if features is not None:
        if target in features:
            raise ValueError(f"Target column '{target}' cannot be in feature columns.")

    features or [col for col in columns]
    df = load_df(fillna=fillna)

    targets = df[target].values.ravel()
    dfx = df.drop(columns=[target])
    assert np.all([col in dfx.columns for col in features])
    samples = dfx.to_numpy()

    assert samples.shape[1] == len(features)

    return samples, targets




if __name__ == "__main__":
   df= load_data()


