import datetime
import json
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers, metrics


from demonstration.demonstration_data import load_df, CATEGORICAL, RANDOM_STATE, fill_numerical_column_by_cond_median, \
    FEATURES, NUMERICAL

PATH = os.path.dirname(os.path.abspath(__file__))

TARGET = "obj_purchasePrice"


def prepare_data(drop_target=True):
    data_path = os.path.join(PATH, f"nn_data_{TARGET}_{str(drop_target)}.pickle")

    if False:#os.path.exists(data_path):
        print("Load existing dataset")
        with open(data_path, "rb") as f:
            df_train, df_test, train, val = pickle.load(f)
    else:
        print("Prepare dataset")
        df_train = load_df(train=True)

        if True:
            bins = np.array([0.0, 106.0, 206.00, 306.0, 406.0, 503.0, 715.0, 1000.0, 1000000.0]) * 1000.0
            df_train["target"] = pd.cut(df_train[TARGET], bins=bins, labels=False, right=True, retbins=False)
            print("Train")
            print(df_train[TARGET][df_train["target"].isna()])
        else:
            df_train["target"], bins = pd.qcut(df_train[TARGET], 12, labels=False, retbins=True)
        if drop_target:
            df_train.drop(columns=[TARGET], inplace=True)

        with open(os.path.join(PATH, "bins.json"), "wt") as f:
            json.dump([edge for edge in bins], f)

        df_train = fill_numerical_column_by_cond_median(source_df=df_train,
                                                        condition_column="obj_buildingType",
                                                        target_df=df_train,
                                                        target_columns=[ ])#"obj_noParkSpaces",])
                                                                         #"obj_numberOfFloors"])
        fill_nan_with_quantile_median(df_train)

        train, val = train_test_split(df_train, test_size=0.2, random_state=RANDOM_STATE)

        df_test = load_df(train=False)

        bc_bins = np.broadcast_to(bins, (len(df_test), bins.shape[0]))
        df_test["target"] = np.argmin(np.abs(bc_bins - df_test[TARGET].to_numpy().reshape(-1, 1)), axis=1)
        print("Test")
        print(df_test[TARGET][df_test["target"].isna()])

        if drop_target:
            df_test.drop(columns=[TARGET], inplace=True)

        df_test = fill_numerical_column_by_cond_median(source_df=df_train,
                                                       condition_column="obj_buildingType",
                                                       target_df=df_test,
                                                       target_columns=[col for col in FEATURES
                                                                       if col in NUMERICAL])
        fill_nan_with_median(df_test, df_train)

        with open(data_path, "wb") as f:
            pickle.dump((df_train, df_test, train, val), f)

    # for _df in (df_train, train, df_test, val):
    #     _df.drop(columns=["obj_regio1"], inplace=True)

    return df_train, df_test, train, val


def fill_nan_with_median(df, source_df):
    """Replace NaNs in numeric columns with median of column in source DataFrame"""
    for column in df.columns:
        if column not in CATEGORICAL:
            df[column].fillna(source_df[column].dropna().median(), inplace=True)
    return df


def fill_nan_with_quantile_median(df):
    """Replace NaNs in numeric columns with quantile median"""

    def _quantile_median(df, row, column, cache):
        try:
            return cache[(column, row["target"])]
        except KeyError:
            # median_value = df[df["target"] == row["target"]][column].dropna().median()
            # if np.isnan(median_value):
            median_value = df[(df["target"] >= row["target"] - 1) & (df["target"] <= row["target"] + 1)][
                column].dropna().median()
            if np.isnan(median_value):
                median_value = df[column].dropna().median()
            cache[(column, row["target"])] = median_value
            return cache[(column, row["target"])]

    return _fill_nan(df, _quantile_median)


def _fill_nan(df, func):
    cache = {}
    for column in df.columns:
        if column not in CATEGORICAL and column != "target":
            df[column] = df.apply(
                lambda row: func(df, row, column, cache)
                if np.isnan(row[column]) else row[column],
                axis=1
            )

    return df


def df_to_dataset(dataframe, shuffle=True, batch_size=64, target="obj_PurchasePrice"):
    """From https://www.tensorflow.org/tutorials/structured_data/feature_columns"""
    dataframe = dataframe.copy()
    labels = dataframe.pop(target)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def build_model(feature_layer, num_classes, layer_conf):
    conf = [feature_layer]
    for dropout_rate, neurons in layer_conf:
        if dropout_rate > 0.0:
            conf.append(layers.Dropout(dropout_rate))
        conf.append(layers.Dense(neurons, activation='relu'))
    conf.append(layers.Dense(num_classes, activation='softmax'))

    return tf.keras.models.Sequential(conf)


def train_model(layer_conf=None):
    layer_conf = layer_conf or []

    df, test, train, val = prepare_data()
    target = "target"

    train_ds = df_to_dataset(train, target=target)
    val_ds = df_to_dataset(val, shuffle=False, target=target)
    test_ds = df_to_dataset(test, shuffle=False, target=target)

    model, num_classes = create_model(target, layer_conf)

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", off_by_one_accuracy])

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(PATH, "logs", current_time)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor="val_off_by_one_accuracy",
                                                               patience=20, restore_best_weights=True)

    val_cnts = train["target"].value_counts()
    class_weight = {x: val_cnts.sum() / val_cnts[x] for x in range(num_classes)}

    history = model.fit(train_ds, validation_data=val_ds, class_weight=class_weight,
                        epochs=100, callbacks=[tensorboard_callback, early_stopping_callback], verbose=2)
    best_val_oboa = early_stopping_callback.best
    print(model.summary())
    print(model.evaluate(test_ds))

    model.save(os.path.join(PATH, f"{current_time}immo_nn.tf"), save_format="tf")
    model.save_weights(os.path.join(PATH, f"{current_time}immo_nn.weights"))

    return current_time, best_val_oboa


def create_model(target="target", layer_conf=None):
    df, test, train, val = prepare_data()

    num_classes = len(train[target].unique())

    feature_columns = create_feature_columns(target, train)
    feature_layer = layers.DenseFeatures(feature_columns)
    model = build_model(feature_layer, num_classes, layer_conf)

    return model, num_classes


def create_feature_columns(target, train, cross=None):
    cross = cross or []
    crossed = [pair[0] for pair in cross] + [pair[1] for pair in cross]

    feature_columns = []

    for column in train.columns:
        if column == target or column == TARGET:
            continue

        print(column)
        if column in CATEGORICAL:
            cat_values = len(train[column].unique())

            categorical_column = feature_column.categorical_column_with_vocabulary_list(
                column, list(train[column].unique()))
            if cat_values < 20 or column in crossed:
                print(f"Column {column} with {cat_values} values as one-hot column")
                the_feature_column = feature_column.indicator_column(categorical_column)
            else:
                print(f"Column {column} with {cat_values} values as embedding column")
                the_feature_column = feature_column.embedding_column(categorical_column,
                                                                     10)
        else:
            integer_rate = np.sum(train[column].astype(np.int) == train[column].astype(np.float)) / len(
                train[column])
            if integer_rate < 0.95:
                print(f"Column {column} with {integer_rate:0.2f} percent integers as scaled column")
                mean = train[column].mean()
                std = train[column].std()
                the_feature_column = feature_column.numeric_column(column,
                                                                   normalizer_fn=lambda x: (x - mean) / std)
            else:
                print(f"Column {column} with {integer_rate:0.2f} percent integers as unscaled column")
                the_feature_column = feature_column.numeric_column(column)

        feature_columns.append(the_feature_column)

        first_column, second_column = None, None
        for column1, column2 in cross:
            for column in feature_columns:
                if column.name == column1:
                    first_column = column
                if column.name == column2:
                    second_column = column
            feature_columns.remove(column1)
            feature_columns.remove(column2)
            crossed_column = feature_column.crossed_column([first_column, second_column], 10000)
            the_feature_column = feature_column.embedding_column(crossed_column, 10)

            feature_columns.append(the_feature_column)

    return feature_columns


def load_model(code="20201214-133037"):
    # cannot load from TF format with custom metric, use workaround found at
    # https://github.com/tensorflow/tensorflow/issues/33646#issuecomment-566433261
    model = tf.keras.models.load_model(os.path.join(PATH, f"{code}immo_nn.tf"),
                                       custom_objects={"off_by_one_accuracy": off_by_one_accuracy},
                                       compile=False)
    #model= hub.KerasLayer(os.path.join(PATH, f"{code}immo_nn.tf"),trainable=True,signature="tokens", output_key="pooled_output")

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", off_by_one_accuracy])
    return model


def load_model_from_weights(code="20201214-133037", layer_conf=((0.0, 256), (0.1, 256))):
    # chaining in TransferLayer does not work when loading stored model,
    # did not investigate in detail, but of course it's messy...
    model, _ = create_model(layer_conf=list(layer_conf))
    model.load_weights(os.path.join(PATH, f"{code}immo_nn.weights"))
    return model


def off_by_one_accuracy(y_true, y_pred):
    return (metrics.sparse_categorical_accuracy(y_true - 1, y_pred) +
            metrics.sparse_categorical_accuracy(y_true, y_pred) +
            metrics.sparse_categorical_accuracy(y_true + 1, y_pred))


def inspect_confusion_matrix(code="", model=None):
    from sklearn.metrics import confusion_matrix

    if model is None:
        model = load_model()

    df, test, train, val = prepare_data()

    for name, df in (("test", test), ("train", train), ("val", val)):
        print(name)

        y_true = df["target"].to_numpy()

        data_ds = df_to_dataset(df, shuffle=False, target="target")
        prediction = model.predict(data_ds)
        y_pred = np.argmax(prediction, axis=1)

        y_conf = prediction[range(len(y_true)), y_true]
        correct = y_true == y_pred
        print(np.mean(y_conf[correct]), np.mean(y_conf[~correct]))

        cm = confusion_matrix(y_true, y_pred, normalize='true')
        print('confusion  matrix')
        for i, row in enumerate(cm):
            print([f"{100 * val:0.2f}" for val in row], np.sum(row[max(0, i - 1):i + 2]))

    return cm

#20210222-161559
if __name__ == "__main__":
    #model = load_model(code="20210222-161559")
    #inspect_confusion_matrix(model=model)

    #print('shouldhavestopped')
    df_train, df_test, train, val = prepare_data()
    print(df_train["target"].value_counts())
    val_cnts = df_train["target"].value_counts()
    class_occ = [val_cnts[x] for x in range(8)]
    print(class_occ)
    class_weights = [val_cnts.sum()/val_cnts[x] for x in range(8)]

    print(class_weights)
    configurations = [[(0.0, 256), (0.1, 256)]] #,  [(0.0, 512), (0.1, 512)],
                      # [(0.1, 256), (0.1, 256)],  [(0.1, 512), (0.1, 512)],
                     # [(0.0, 256), (0.0, 256)],  [(0.0, 512), (0.0, 512)]]
    for configuration in configurations:
        for i in range(1):
            code, best_oboa = train_model(layer_conf=configuration)

            wmodel = load_model_from_weights(code=code, layer_conf=configuration)
            inspect_confusion_matrix(model=wmodel)
            smodel = load_model(code=code)
            inspect_confusion_matrix(model=smodel)

            with open(os.path.join(PATH, "log.txt"), "at") as f:
                f.write(f"\n{code},'{str(configuration)}',{best_oboa:0.4f}")

    # # 20200416-143225,'[(0.0, 256), (0.1, 256)]',0.8654
    # inspect_confusion_matrix("20200416-143225")
    # model = load_model_from_weights()
    # inspect_confusion_matrix(model=model)
