import tensorflow as tf
from demonstration.demonstration_immo_model import off_by_one_accuracy
import os

PATH = os.path.dirname(os.path.abspath(__file__))

def load_model(code):
    # cannot load from TF format with custom metric, use workaround found at
    # https://github.com/tensorflow/tensorflow/issues/33646#issuecomment-566433261
    model = tf.keras.models.load_model(os.path.join(PATH, f"{code}immo_nn.tf"),
                                       custom_objects={"off_by_one_accuracy": off_by_one_accuracy},
                                       compile=False)
    #model= hub.KerasLayer(os.path.join(PATH, f"{code}immo_nn.tf"),trainable=True,signature="tokens", output_key="pooled_output")

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", off_by_one_accuracy])
    return model