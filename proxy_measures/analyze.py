import os
from typing import List, Dict, Any, Tuple, Callable

import numpy as np

from demonstration.proxy_measures.closeness import make_distance,create_reference_points,min_distance_to_nearest_datapoint,make_cat_distance, naive_distance, weighted_distance, wachter_distance, \
    euclidean_distance
from demonstration.proxy_measures.sparsity import make_sparsity, make_categorical_sparsity
from demonstration.proxy_measures.correctness import immo_correctness, adult_correctness
from demonstration.proxy_measures.path_density import integrated_density, density_delta, density_variation, density_min_start, density_average, density_foil, density_opt_step_max,density_opt_step_min
from scipy import stats
from demonstration.demonstration_adult_data  import FEATURES, VAR_TYPES
PATH = os.path.dirname(os.path.abspath(__file__))
from numpy import mean, absolute




def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)

def analyze_foils(pairs: List[Tuple[Dict[str, Any], Dict[str, Any], np.ndarray,float,float]],
                  measures: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], float]]):
    def compute_measure(measure_fn: Callable[[Dict[str, Any], Dict[str, Any], np.ndarray,float,float], float]) -> Dict[str, float]:
        print(pairs)
        measure_val = [measure_fn(fact=fact, foil=foil, densities=densities,predclass=predcl,fact_cl=factcl) for fact, foil, densities,predcl,factcl in pairs]
        #measure_val = [measure_fn(fact=fact, foil=foil, densities=densities) for
        #               fact, foil, densities in pairs]
        return {"mean": float(np.mean(measure_val)),
                #"mad": float(stats.median_abs_deviation(measure_val)),
                "mad":float(mad(np.array(measure_val))),
                #"min": float(np.min(measure_val)),
                #"max": float(np.max(measure_val)),
                #"median": float(np.median(measure_val)),
                "std": float(np.std(measure_val)),
                "all": measure_val
                }

    return {measure: compute_measure(measure_fn) for measure, measure_fn in measures.items()}


def instantiate_all_measures(mad: np.ndarray) -> Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], float]]:
    return {
        ##"fully_naive_distance": make_distance(naive_distance, euclidean_distance()),
        #"wachter_distance": make_distance(naive_distance, wachter_distance(mad)),
        "cocomix_distance": make_distance(weighted_distance, wachter_distance(mad),beta=0.5),
        # "cocomix_distance2": make_distance(weighted_distance, wachter_distance(mad),beta=0.25),
        # "cocomix_distance3": make_distance(weighted_distance, wachter_distance(mad),beta=0.1),
        # "cocomix_distance4": make_distance(weighted_distance, wachter_distance(mad),beta=0.35),
        "sparsity": make_sparsity(FEATURES),
        #"cat_sparsity" : make_categorical_sparsity(FEATURES,VAR_TYPES),
        # # #"integrated_density": integrated_density,
        # # #"density_delta": density_delta,
        # # #"density_variation": density_variation,
        #"density_min_start": density_min_start,
        # # #"density_average": density_average,
        "density_foil": density_foil,
        #"density_opt_step_min": density_opt_step_min,
        # "density_opt_step_max": density_opt_step_max,
        #"cat_distance": make_cat_distance(weighted_distance),
        #"cat_distance_naive": make_cat_distance(naive_distance),
        # #"fact_distance": min_distance_to_nearest_datapoint(make_distance(weighted_distance, wachter_distance(mad)),create_reference_points())
        "correctness": adult_correctness
    }

# df_train = load_df(train=True)
# mad = calculate_mad(df_train[[feature for feature, var_type in zip(FEATURES, VAR_TYPES)
#                               if var_type == "c"]].to_numpy())
# mad[mad == 0.0] = 0.5
# version = "20200604-202834"
# with open(os.path.join(PATH, f"{version}generatedfoils" + ".pickle"), "rb") as f:
#     data = pickle.load(f)
# beta = 0.5
# minpdf=[]
# avpdf=[]
# distnaive=[]
# distweighted=[]
# spars=[]
# for i in range(10):
#     fact = data[i][0]
#     foil = data[i][1]
#     history = data[i][2]
#     minpdf.append(min(history['pdf']))
#     avpdf.append(np.mean(history['pdf'][1:]))
#     fact = dict(zip(FEATURES, fact))
#     foil = dict(zip(FEATURES, foil))
#     distnaive.append(distance(fact, foil, naive_distance, wachter_distance(mad), beta))
#     distweighted.append(distance(fact, foil, weighted_distance, wachter_distance(mad), beta))
#     spars.append(sparsity(fact, foil))
#     #könnten hier dann auchb direkt auf foilbasis vergleichen, wäre vor allem bei den pdfs besser
#
# print('average pdf')
# print(np.mean(avpdf))
# print('minpdf')
# print(np.mean(minpdf))
# print('distanaive')
# print(np.mean(distnaive))
# print('distweighted')
# print(np.mean(distweighted))
# print('sparsity')
# print(np.mean(spars))
#
