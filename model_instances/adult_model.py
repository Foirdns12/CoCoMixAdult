import pickle
import os
import joblib

PATH = os.path.dirname(os.path.abspath(__file__))

def load_model():
    # cannot load from TF format with custom metric, use workaround found at
    # https://github.com/tensorflow/tensorflow/issues/33646#issuecomment-566433261
    #model = pickle.load(open(os.path.join(PATH, f"{code}adult_rf.tf"),'rb'))
    model = joblib.load(open(os.path.join(PATH, "adult_rf.joblib"),'rb'))
    return model

