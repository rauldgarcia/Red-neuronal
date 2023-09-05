import numpy as np

# Values from the earlier previous when we described
# What a neural network is
layer_outputs = [4.8, 1.21, 2.385]

# For each value in a vector, calculate the exponential value
exp_values = np.exp(layer_outputs)
print('Exponentiated values:')
print(exp_values)

# Now normalize values
norm_values = exp_values / np.sum(exp_values)
print('Normalized exponentiated values:')
print(norm_values)
print('Sum of normalized values', np.sum(norm_values))