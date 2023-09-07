import matplotlib.pyplot as plt
import math

a = 1 # leakage coefficient
b = 1 # input scaling
c = 0 # inhibitory feedback gain

listx = []
listy = []
t = 0
for x in range (0, 11):
    listx.append([x])
    listy.append([(k * (math.e ** (-a*t))) + (x/a)])
    t += 1

plt.plot(listx, listy)
plt.show()