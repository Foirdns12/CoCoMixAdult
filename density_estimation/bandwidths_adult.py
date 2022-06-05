from demonstration.demonstration_adult_data import FEATURES, VAR_TYPES

FINAL_BANDWIDTHS = {
    "age": 3.0,
    "workclass": 0.4,
    "fnlwgt": 10000.0,
    "education": 0.6,
    "education-num": 0.5,
    "marital-status": 0.4,
    "occupation": 0.6,
    "relationship": 0.35,
    "race": 0.5,
    "sex": 0.2,
    "hours-per-week": 1.0,
    "native-country": 0.96
}

final_bandwidths = [FINAL_BANDWIDTHS[feature] for feature in FEATURES]

if __name__ == "__main__":
    for feature, var_type, bw in zip(FEATURES, VAR_TYPES, final_bandwidths):
        print(feature, var_type, bw)



