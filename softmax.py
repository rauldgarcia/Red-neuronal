import math

# Layer from the previous output when we described
# what a neural network is
layer_outputs = [4.8, 1.21, 2.385]

# e - mathematical constant, we use E here to match a common coding
# style where constants are uppercased
# For each value in a vector, calculate the exponential value
exp_values = []
for output in layer_outputs:
    exp_values.append(math.e ** output)

print('Exponientiated values:')
print(exp_values)

# Now normalized values
norm_base = sum(exp_values) # We sum all values
norm_values = []
for value in exp_values:
    norm_values.append(value/norm_base)
print('Normalized exponentiated values:')
print(norm_values)

print('Sum of normalized values:', sum(norm_values))