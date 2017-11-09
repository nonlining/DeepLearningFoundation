#-------------------------------------------------------------------------------
# Name:        Digits
# Purpose:
#
# Author:      Min-Jung Wang
#
# Created:     07/07/2017
# Copyright:   (c) Min-Jung Wang 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.utils import shuffle, resample
import numpy as np
from miniflow import *

np.random.seed(1)

def main():

    data = load_digits()

    max_value = np.amax(data['data'])
    min_value = np.amin(data['data'])

    X_ = normalized(data['data'], max_value, min_value)
    X_ = X_.reshape(X_.shape[0], 1, 8, 8)
    print "Input size", X_.shape

    y_ = data['target']
    n_values = np.max(y_) + 1
    y_ = np.eye(n_values, dtype=int)[y_]

    # parameters
    fitter_numbers = 2
    kernel_size = (3,3)
    pooling_size = (2,2)

    # init layers
    W_layer1 = np.random.normal(0, 0.1, (fitter_numbers, kernel_size[0]* kernel_size[1]))
    W_layer1 = W_layer1.reshape(fitter_numbers, 1, kernel_size[0], kernel_size[1])
    b_layer1 = np.zeros(fitter_numbers, )
    W_layer2 = np.random.normal(0, 0.1, (25*fitter_numbers, 10))
    b_layer2 = np.zeros(10)

    # network
    X, y = Input(), Input()

    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()

    conv_layer1 = Conv(X, W1, b1, (1,1), 0)

    pooling1 = Pooling(conv_layer1, pooling_size, (1,1), 0)

    activation_1 = Relu(pooling1)

    dropout1 = Dropout(activation_1, 0.5)

    linear = Linear(dropout1, W2, b2)

    output = Softmax(linear, y)

    feed_dict = {
        X: X_,
        y: y_,
        W1: W_layer1,
        b1: b_layer1,
        W2:W_layer2,
        b2:b_layer2
    }

    graph = topological_sort(feed_dict)

    trainables = [W1, b1, W2, b2]
    epochs = 10
    learning_rate=1e-2

    for i in range(epochs):

        forward_and_backward(graph)
        #forward(graph)
        for t in trainables:
            partial = t.gradients[t]
            t.value -= learning_rate * partial

        loss = graph[-1].diff
        print loss
    print W1.value

    #plt.figure()
    #plt.plot(loss_list)
    #plt.show()


if __name__ == '__main__':
    main()
