import datetime
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

from data.immodata import load_df, CATEGORICAL

PATH = os.path.dirname(os.path.abspath(__file__))

tf.keras.backend.set_floatx("float64")


def df_to_dataset(dataframe, shuffle=True, batch_size=32, target="obj_PurchasePrice"):
    """From https://www.tensorflow.org/tutorials/structured_data/feature_columns"""
    dataframe = dataframe.copy()
    labels = dataframe.pop(target)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def build_model(feature_layer):
    return tf.keras.models.Sequential([
        feature_layer,
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(10, activation='softmax')
    ])


def capped_mean_absolute_percentage_error(y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    diff = tf.clip_by_value(math_ops.abs(
        (y_true - y_pred) / K.maximum(math_ops.abs(y_true), K.epsilon())), 0.0, 1.0)
    return 100. * K.mean(diff, axis=-1)


def train_model(target="obj_purchasePrice"):
    df, target, test, train, val = prepare_data(target)

    print(len(train), 'train examples')
    print(len(val), 'validation examples')
    print(len(test), 'test examples')

    target = "target"
    train_ds = df_to_dataset(train, target=target)
    val_ds = df_to_dataset(val, shuffle=False, target=target)
    test_ds = df_to_dataset(test, shuffle=False, target=target)

    feature_columns = []
    for column in df.columns:
        if column == target:
            continue

        if column in CATEGORICAL:
            cat_values = len(df[column].unique())
            print(column, cat_values)

            if cat_values < 20:
                categorical_column = feature_column.categorical_column_with_vocabulary_list(
                    column, list(df[column].unique()))
                the_feature_column = feature_column.indicator_column(categorical_column)
            else:
                categorical_column = feature_column.categorical_column_with_hash_bucket(column,
                                                                                        cat_values
                                                                                        )
                the_feature_column = feature_column.embedding_column(categorical_column,
                                                                     int(cat_values / 25) + 1)
        else:
            mean = train[column].mean()
            std = train[column].std()

            the_feature_column = feature_column.numeric_column(column,
                                                               normalizer_fn=lambda x: (x - mean) / std)

        feature_columns.append(the_feature_column)

    feature_layer = layers.DenseFeatures(feature_columns)
    model = build_model(feature_layer)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(PATH, "logs", current_time)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_ds, validation_data=val_ds, epochs=100, callbacks=[tensorboard_callback])
    print(model.summary())

    model.save(os.path.join(PATH, "immo_nn.tf"), save_format="tf")


def prepare_data(target):
    data_path = os.path.join(PATH, f"nn_data_{target}.pickle")
    if os.path.exists(data_path):
        with open(data_path, "rb") as f:
            print("Load existing dataset")
            return pickle.load(f)
    else:
        print("Prepare dataset")
        df = load_df(fillna=False)
        print(df.head())
        df["target"] = pd.qcut(df[target], 10, labels=False)
        # replace NaNs in numeric columns with quantile mean
        for column in df.columns:
            if column not in CATEGORICAL:
                df[column] = df.apply(
                    lambda row: df[df["target"] == row["target"]][column].dropna().mean()
                    if np.isnan(row[column]) else row[column],
                    axis=1
                )
        train, test = train_test_split(df, test_size=0.2)
        train, val = train_test_split(train, test_size=0.2)

        with open(data_path, "wb") as f:
            pickle.dump((df, target, test, train, val), f)

    return df, target, test, train, val


def load_model():
    return tf.keras.models.load_model(os.path.join(PATH, "immo_nn.tf"))


def inspect_confusion_matrix():
    from sklearn.metrics import confusion_matrix

    model = load_model()

    df, target, test, train, val = prepare_data("obj_purchasePrice")

    y_true = test["target"].to_numpy()

    test_ds = df_to_dataset(test, shuffle=False, target="target")

    y_pred = np.argmax(model.predict(test_ds), axis=1)

    cm = confusion_matrix(y_true, y_pred, normalize='true')

    for row in cm:
        print([f"{100*val:0.2f}" for val in row])


if __name__ == "__main__":
    # train_model()
    inspect_confusion_matrix()
