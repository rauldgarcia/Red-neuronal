import matplotlib.pyplot as plt
import math
import numpy as np
import copy

def relu(num):
    return max(0, num)

def softrelu(num):
    return math.log(1 + (math.e**num))

a = 1 # leakage coefficient >= 0
b = 0.1 # input scaling > 0
c = 0.7 # inhibitory feedback gain >= 0
t = 1 # time delay > 0

listx = []
listy = []
listy2 = []
time = 0
last_y = 0
listx.append([0])
listy.append([0])
listy2.append([0])


for signal in range(0, 101):
    listx.append([time])
    input_signal = np.sin(0.1 * signal) + 0.5 * np.sin(0.5 * signal)
    new_y = -(a*input_signal) + (b*time) - (c*last_y)
    listy.append([new_y])
    listy2.append([input_signal])
    last_y = copy.deepcopy(new_y)
    time += t

plt.plot(listx, listy)
plt.plot(listx, listy2)
plt.show()