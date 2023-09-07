import matplotlib.pyplot as plt
import math

k = 1 # Constant
a = 1 # Rate of the leak

listx = []
listy = []
t = 0
for x in range (-10, 11):
    listx.append([x])
    listy.append([(k * (math.e ** (-a*t))) + (x/a)])
    t += 1

plt.plot(listx, listy)
plt.show()