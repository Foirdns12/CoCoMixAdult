from demonstration.demonstration_data import FEATURES, VAR_TYPES

FINAL_BANDWIDTHS = {
    "obj_yearConstructed": 2.0,
    "obj_livingSpace": 7.5,
    "obj_noRooms": 0.7,
    "obj_numberOfFloors": 0.5,
    "obj_lotArea": 40.0,
    "obj_noParkSpaces": 0.5,
    "obj_buildingType": 0.4,
    "obj_cellar": 0.3,
    "obj_condition": 0.5,
    "obj_interiorQual": 0.4,
    "obj_heatingType": 0.65,
    "obj_newlyConst": 0.1,
    "obj_regio1": 0.9,
    "geo_krs": 0.9976  # all probabilities are equal
}

final_bandwidths = [FINAL_BANDWIDTHS[feature] for feature in FEATURES]

if __name__ == "__main__":
    for feature, var_type, bw in zip(FEATURES, VAR_TYPES, final_bandwidths):
        print(feature, var_type, bw)