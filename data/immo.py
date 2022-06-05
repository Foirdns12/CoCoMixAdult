import os

import numpy as np
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))

CATEGORICAL = [
    "obj_regio1",
    "obj_heatingType",
    "obj_newlyConst",
    "obj_firingTypes",
    "obj_courtage",
    "obj_cellar",
    "geo_krs",
    "obj_zipCode",
    "obj_condition",
    "obj_interiorQual",
    "obj_rented",
    "obj_buildingType",
    "obj_barrierFree",
    "obj_regio3",
    "obj_regio4",
    "factID"
]

NUMERICAL = [
    "obj_lotArea",
    "obj_yearConstructed",
    "obj_noParkSpaces",
    "obj_livingSpace",
    "obj_noRooms",
    "obj_purchasePrice",
    "obj_numberOfFloors",
    "obj_lastRefurbish",
    "Einwohnerdichte_PLZ"
]

ALL = NUMERICAL + CATEGORICAL

# columns that we drop in any case as they are values
# internal to the platform or promotional
_DROP = [
    #"obj_purchasePriceRange",
    "obj_telekomUploadSpeed",
    "obj_telekomDownloadSpeed",
    "obj_telekomInternet"
    #"ga_cd_via"
]


def load_df(columns=None, fillna="mean"):
    """

    :param columns: List of columns to keep
    :param fillna: If not False, method to replace NaNs in numerical columns.
    :return: pd.DataFrame with the specified columns
    """
    columns = columns or ALL

    df = pd.ExcelFile(os.path.join(PATH, "immodata","immo1.6fix.xlsx")).parse(0) #"immo1.5fix.xlsx")).parse(0)
    df = df.drop(columns=_DROP)
    df["geo_krs"]=df["geo_krs"].str.lower()
    #print(ALL)
    #print(df.columns)
    #print([col in ALL for col in df.columns])
    assert np.all([col in ALL for col in df.columns])

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

    return df


def load_data(features=None, target="obj_purchasePrice", fillna="mean"):
    """

    :param features: List of columns to use as features
    :param target: Name of column to use as the target
    :param fillna: see `load_df` for details
    :return: Samples as np.ndarray of size (n_samples, n_features),
             Targets as np.ndarray of size (n_samples,)
    """
    if features is not None:
        if target in features:
            raise ValueError(f"Target column '{target}' cannot be in feature columns.")
    features = features or [col for col in ALL if col != target]

    columns = features + [target]
    df = load_df(columns=columns, fillna=fillna)

    targets = df[target].values.ravel()

    dfx = df.drop(columns=[target])
    assert np.all([col in dfx.columns for col in features])
    samples = dfx.to_numpy()

    assert samples.shape[1] == len(features)

    return samples, targets
