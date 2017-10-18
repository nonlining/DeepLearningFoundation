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

epochs = 10

np.random.seed(1)

def main():
    data = load_digits()

    max_value = np.amax(data['data'])
    min_value = np.amin(data['data'])

    X_ = normalized(data['data'], max_value, min_value)


    y_ = data['target']

    n_features = X_.shape[1]
    print X_.shape


    # parameters
    fitter_numbers = 8
    kernel_size = (3,3)

    # init layers
    W_layer1 = np.random.normal(0, 0.1, (fitter_numbers, kernel_size[0]*kernel_size[1]))
    b_layer1 = np.random.normal(0, 0.1, (fitter_numbers, ))

    W_layer2 = np.random.normal(0, 0.1, (fitter_numbers, 1))
    b_layer2 = np.random.normal(0, 0.1, (1, ))

    # test
    #print X_[0].shape, W_layer1.shape
    #print conv(X_[0], (8,8) ,W_layer1, kernel_size, (1,1)) + b_layer1[0]

    # network
    X, y = Input(), Input()

    W1, b1 = Input(), Input()

    W2, b2 = Input(), Input()

    conv_layer1 = Conv(X, W1, b1, (8,8), kernel_size, (1,1))

    activation_1 = Relu(conv_layer1)

    #output = soft_max(activation_1, y)


    feed_dict = {
        X: X_,
        y: y_,
        W1: W_layer1,
        b1: b_layer1,
        W2: W_layer2,
        b2: b_layer2
    }

    graph = topological_sort(feed_dict)

    trainables = [W1, b1, W2, b2]
    forward(graph)

    out_shape = graph[-2].value.shape
    print out_shape


if __name__ == '__main__':
    main()
