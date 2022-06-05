import os

import pandas as pd

from data.adult import VALUES, CATEGORICAL, FEATURES, COLUMNS

PATH = os.path.dirname(os.path.abspath(__file__))

columns_to_drop = ["fnlwgt", "education-num", "relationship", "native-country", "race", "capital-loss"]
CATEGORICAL = [categorical for i, categorical in enumerate(CATEGORICAL) if FEATURES[i] not in columns_to_drop]
FEATURES = [feature for feature in FEATURES if feature not in columns_to_drop]

def load(kind):
    class_label = {
        "data": ">50K",
        "test": ">50K."
    }

    df = pd.read_csv(os.path.join(PATH, "adult", f"adult.{kind}"))
    df.columns = COLUMNS + ["label"]
    df["capital-gain"] = df["capital-gain"] - df["capital-loss"]
    df = df.drop(columns=columns_to_drop)
    stdv = []
    for feature, categorical in zip(FEATURES, CATEGORICAL):

        if categorical:
            df[feature] = df[feature].str.strip()
            stdv.append("cat")

        else:
            stdv.append(df[feature].std())


    df["label"] = df["label"].str.strip()

    for i, column_label in enumerate(df.columns):
        if column_label != "label":
            assert column_label == FEATURES[i]

    samples = df.drop(columns=["label"]).to_numpy()
    targets = (df["label"] == class_label[kind]).astype(int)
    assert len(targets.unique()) == 2

    return samples, targets.to_numpy(), stdv


def load_data():
    return load("data")


def load_test():
    return load("test")

