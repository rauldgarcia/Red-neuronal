import matplotlib.pyplot as plt

def relu(num):
    return max(0, num)

w1 = -1
bias = 0.5

w2 = -2
bias2 = 1

listx = []
listy = []
for x in range(-10, 11):
    listx.append([x/10])
    output_n1=relu(((x/10) * w1) + bias)
    listy.append([relu(((output_n1) * w2) + bias2)])

plt.plot(listx, listy)
plt.scatter(0.25,relu(((relu(((0.25) * w1) + bias)) * w2) + bias2))
plt.show()