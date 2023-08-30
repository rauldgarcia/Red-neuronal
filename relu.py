import math
import matplotlib.pyplot as plt

def softrelu(num):
    return math.log(1 + (math.e**num))

def relu(num):
    return max(0, num)

listx = []
listy = []
listy2= []
for x in range(-10, 10):
    listx.append([x])
    listy.append([softrelu(x)])
    listy2.append([relu(x)])

plt.plot(listx, listy)
plt.plot(listx, listy2)
plt.show()