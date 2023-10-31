import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_region(x, y, classifier, resolution=0.02):

    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Plot the decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0],
                    y=x[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolors='black')


class Perceptron(object):
    """Perceptron classifier.
    
    Parameters:
    eta: float
        learning rate (between 0 and 1)
    n_iter: int
        passes over the training dataset
    random_state: int
        random number generator seed for random weight initialization
        
    Attributes:
    w_: 1d-array
        weights after fitting
    errors_: list
        number of misclassifications (updates) in each epoch"""
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        """Fit training data
        
        Parameters:
        x : {array-like}, shape = [n_samples, n_features]
            training vectors
        y : array-like, shape = [n_samples]
            target values
            
        Returns
        self : object"""

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, x):
        # Calculate net input
        return np.dot(x, self.w_[1:] + self.w_[0])
    
    def predict(self, x):
        # Return class label after unit step
        return np.where(self.net_input(x) >= 0.0, 1, -1)


class AdalineGD(object):

    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2
            self.cost_.append(cost)
        
        return self
    
    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def activation(self, x):
        return x
    
    def predict(self, x):
        return np.where(self.activation(self.net_input(x)) >= 0, 1, -1)
    


class AdalineSGD(object):

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, x, y):

        self._initialize_weights(x.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            cost = []
            for xi, target in zip(x, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self
    
    def partial_fit(self, x, y):
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)
        return self
    
    def _shuffle(self, x, y):
        r = self.rgen.permutation(len(y))
        return x[r], y[r]
    
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost
    
    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
    
    def activation(self, x):
        return x
    
    def predict(self, x):
        return np.where(self.activation(self.net_input(x)) >= 0, 1, -1)


df = pd.DataFrame([[4, 4, 'A', 1, 1],
                   [3.5, 3.5, 'A', 1, 1],
                   [3, 4, 'A', 1, 1],
                   [3.5, 4, 'A', 1, 1],
                   [5, 2, 'B', 1, 0],
                   [4.5, 2.5, 'B', 1, 0],
                   [5, 3, 'B', 1, 0],
                   [4.5, 3, 'B', 1, 0],
                   [3, 1, 'C', 0, 0],
                   [2.5, 1.5, 'C', 0, 0],
                   [3.5, 2, 'C', 0, 0],
                   [4, 1.2, 'C', 0, 0],
                   [1, 2, 'D', 0, 1],
                   [1.5, 2.5, 'D', 0, 1],
                   [2, 3, 'D', 0, 1],
                   [1.5, 3.5, 'D', 0, 1]], columns=['x', 'y', 't', 'y1', 'y2'])

# Select setosa and versicolor
y1 = df.iloc[:, [3]].values
y2 = df.iloc[:, [4]].values

# Extract sepal length and petal lenght
x = df.iloc[:, [0,1]].values

keys = np.array(range(len(x)))
np.random.shuffle(keys)
x = x[keys]
y1 = y1[keys]
y2 = y2[keys]

print('x')
print(x)
print('y1')
y1 = np.reshape(y1, (len(y1),))
y2 = np.reshape(y2, (len(y2),))
print(y1)


ada1 = AdalineSGD(n_iter=20000, eta=0.05, random_state=1).fit(x, y1)
plot_decision_region(x, y2, classifier=ada1)
ada2 = AdalineSGD(n_iter=20000, eta=0.05, random_state=1).fit(x, y2)
plot_decision_region(x, y2, classifier=ada2)

plt.title('Adaline - Stochastic Gradient Descent')
plt.xlabel('Sepal length [std]')
plt.ylabel('Petal length [std]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

'''plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Average Cost')
plt.show()

print()

sns.scatterplot(data=df, x='x', y='y', hue='t')
plt.show()'''