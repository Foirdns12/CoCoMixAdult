import numpy as np
from demonstration.transition_matrices.compute_distance_matrices_adult import compute_distance_matrix

transition_matrix = np.array([
        [0.5, 0.5, 0.0, 0.1],
        [0.1, 0.5, 0.5, 0.2],
        [0.2, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.3, 1]
    ])

print(compute_distance_matrix((transition_matrix)))