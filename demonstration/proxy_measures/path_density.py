from typing import Union, Dict, List, Any

import numpy as np


def integrated_density(densities: np.ndarray, **kwargs: Any) -> float:
    if len(densities) > 1:
        return float(np.sum(np.diff(densities)))
    else:
        print("missing densities for foil")
        return 0


def density_delta(densities: np.ndarray, **kwargs: Any) -> float:
    if len(densities) > 1:
        return float(np.max(densities) - np.min(densities))
    else:
        print("missing densities for foil")
        return 0


def density_variation(densities: np.ndarray, **kwargs: Any) -> float:
    if len(densities) > 1:
        return float(np.std(densities))
    else:
        print("missing densities for foil")
        return 0


def density_min_start(densities: np.ndarray, **kwargs: Any) -> float:
    if len(densities) > 1:
        return float((np.min(densities)) / densities[0])
    else:
        print("missing densities for foil")
        return 0


def density_average(densities: np.ndarray, **kwargs: Any) -> float:
    if len(densities) > 1:
        return float(np.mean(densities))
    else:
        print("missing densities for foil")
        return 0


def density_foil(densities: np.ndarray, **kwargs: Any) -> float:
    if len(densities) > 1:
        return float(densities[len(densities) - 1])
    else:
        print("missing densities for foil")
        return 0


def density_opt_step_min(densities: np.ndarray, **kwargs: Any) -> float:
    if len(densities) > 1:
        minstep = 0
        for i in range(len(densities) - 2):
            if (1 - densities[i + 1] / densities[i]) > minstep:
                minstep = (1 - densities[i + 1] / densities[i])
        return minstep
    else:
        print("missing densities for foil")
        return 0


def density_opt_step_max(densities: np.ndarray, **kwargs: Any) -> float:
    if len(densities) > 1:
        maxstep = 1
        for i in range(len(densities) - 2):
            if (densities[i + 1] / densities[i]) > maxstep:
                maxstep = (densities[i + 1] / densities[i])
        return maxstep
    else:
        print("missing densities for foil")
        return 0
