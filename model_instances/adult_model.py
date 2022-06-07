import pickle
import os
import joblib

PATH = os.path.dirname(os.path.abspath(__file__))

def load_model():
    model = joblib.load(open(os.path.join(PATH, "adult_rf.joblib"),'rb'))
    return model

