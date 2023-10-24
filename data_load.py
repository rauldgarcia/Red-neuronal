import numpy as np
import cv2
import os
import nnfs
import matplotlib.pyplot as plt

# Loads a MNIST dataset
def load_mnist_dataset(dataset, path):

    # Scan all the directiories and create a list of labels
    labels = os.listdir(os.path.join(path, dataset))

    # Create list for samples and labels
    x = []
    y = []

    # For each label folder
    for label in labels:
        # And for each image in given folder
        for file in os.listdir(os.path.join(path, dataset, label)):
            # Read the image
            image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)

            # And append it and a label to the lists
            x.append(image)
            y.append(label)

    # Convert the data to proper numpy arrays and return
    return np.array(x), np.array(y).astype('uint8')

# MNIST dataset (train + test)
def create_data_mnist(path):

    # Load both sets separately
    x, y = load_mnist_dataset('train', path)
    x_test, y_test = load_mnist_dataset('test', path)

    # And return all the data
    return x, y, x_test, y_test

nnfs.init()

# Create dataset
x, y, x_test, y_test = create_data_mnist('fashion_mnist_images')

# Scale features
x = (x.astype(np.float32) - 127.5) / 127.5
x_test = (x.astype(np.float32) - 127.5) / 127.5

# Reshape to vectors
x = x.reshape(x.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

keys = np.array(range(x.shape[0]))

np.random.shuffle(keys)
x = x[keys]
y = y[keys]

plt.imshow((x[4].reshape(28, 28)))
plt.show()

print(y[4])