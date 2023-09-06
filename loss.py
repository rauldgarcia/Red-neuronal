import math
import numpy as np

# An example output from the output layer of the neural network
softmax_output = [0.7, 0.1, 0.2]

# Ground truth
target_output = [1, 0, 0]

loss = -(math.log(softmax_output[0]) * target_output[0] +
         math.log(softmax_output[1]) * target_output[1] +
         math.log(softmax_output[2]) * target_output[2])

print(loss)

softmax_output = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])

class_targets = [0, 1, 1]

print(softmax_output[[0, 1, 2], class_targets])

neg_log = -np.log(softmax_output[
    range(len(softmax_output)), class_targets
])
average_loss = np.mean(neg_log)
print(average_loss)