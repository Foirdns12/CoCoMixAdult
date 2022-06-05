import os

import tensorflow as tf
import numpy as np

from data.synthetic import data, labels

PATH = os.path.dirname(os.path.abspath(__file__))


def build_model(hidden_layers=2, hidden_units=10):
    input_ = tf.keras.layers.Input((2,))
    x = input_
    for _ in range(hidden_layers):
        x = tf.keras.layers.Dense(hidden_units, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    return tf.keras.models.Model(inputs=input_, outputs=output)


def train_model():
    model = build_model()
    print(model.summary())
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy', tf.keras.metrics.Precision(thresholds=0.5)])

    early_stopping = tf.keras.callbacks.EarlyStopping("val_loss",
                                                      patience=100, restore_best_weights=True)
    model.fit(data, labels, validation_split=0.2, epochs=1000,
              callbacks=[early_stopping], verbose=2)

    model.save(os.path.join(PATH, "synthetic.h5"))
    return model


def load_model():
    return tf.keras.models.load_model(os.path.join(PATH, "synthetic.h5"))


if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set()

    trained_model = train_model()

    print(trained_model.predict(data).reshape((500,)).shape)

    df = pd.DataFrame({
        "x": data.transpose()[0],
        "y": data.transpose()[1],
        "label": labels,
        "predicted": trained_model.predict(data).reshape((500,))
    })

    df["predicted_label"] = df["predicted"] > 0.5

    print(df.head())

    sns.scatterplot("x", "y", hue="label", data=df)
    plt.show()

    sns.scatterplot("x", "y", hue="predicted_label", data=df)
    plt.show()
