#import the final bandwidths of the use case
from demonstration.demonstration_data import FEATURES, VAR_TYPES
from demonstration.density_estimation.bandwidths_adult import FINAL_BANDWIDTHS, final_bandwidths


if __name__ == "__main__":
    for feature, var_type, bw in zip(FEATURES, VAR_TYPES, final_bandwidths):
        print(feature, var_type, bw)

