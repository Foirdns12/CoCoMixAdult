import numpy as np

data = np.array([[1, 2.3], [0, 3.1], [0, 5.5], [1, 4.0]])
print("Shape of data:", data.shape)
print("data:", data)

Xi = data[:, 0]

h = 0.25
x = 0

kernel_value = np.zeros_like(Xi)

idx = Xi == x
print(idx)
assert np.all(np.equal(np.array([False, True, True, False]), idx))

# Kernel value where categorical value is the same
check = (idx * (1 - h))  # Boolean array is cast to integer
print("check", check)
assert np.all(np.equal(np.array([0, 0.75, 0.75, 0]), check))
assert check.shape == Xi.shape

print("check[idx]", check[idx])  # Return only where idx is True
assert check[idx].shape == (2,)
assert kernel_value[idx].shape == (2,)

kernel_value[idx] = (idx * (1 - h))[idx]
print("kernel_value", kernel_value)

# Kernel value where categorical value is different
num_values = len(np.unique(Xi))
kernel_value[~idx] = h/(num_values - 1)
print("Full kernel_value", kernel_value)
