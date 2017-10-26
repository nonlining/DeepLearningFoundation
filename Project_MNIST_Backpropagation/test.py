#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Nonlining
#
# Created:     05/10/2017
# Copyright:   (c) Nonlining 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
from miniflow import *
import time

np.random.seed(1)

def main():

    n_hidden = 10
    kernel_size = (3,3)

    W_layer0 = np.array([[1., 0., 1., 1., 0., 1., 1., 0., 1.]])
    b_layer1 = np.ones(n_hidden)

    X_ = [float(i+1) for i in range(64)]
    X_ = np.array(X_).reshape(-1,64)

    for i in range(3):
        X_ = np.vstack((X_,X_))



    W_layer1 = W_layer0
    for i in range(9):
        W_layer1 = np.vstack((W_layer1,W_layer0))

    print X_.shape
    print W_layer1.shape
    print b_layer1.shape

    X = Input()
    y = Input()

    W1, b1 = Input(), Input()
    conv1 = Conv(X ,W1, b1, (8,8) , kernel_size, (1,1))
    activation_1 = Relu(conv1)
    #cost = MSE(y, activation_1)

    feed_dict = {
        X: X_,
        #y: y_,
        W1: W_layer1,
        b1: b_layer1,
    }

    graph = topological_sort(feed_dict)

    trainables = [W1, b1]

    forward_and_backward(graph)
    print graph

    print graph[-1].value.shape




if __name__ == '__main__':
    main()
