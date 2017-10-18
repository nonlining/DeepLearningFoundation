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

    n_hidden = 8
    kernel_size = (3,3)

    W_layer1 = np.array([[1, 0, 1, 1, 0, 1, 1, 0, 1]])
    b_layer1 = np.zeros(n_hidden)

    X_ = [i+1 for i in range(64)]
    X_ = np.array(X_).reshape(-1,64)

    for i in range(3):
        X_ = np.vstack((X_,X_))


    for i in range(3):
        W_layer1 = np.vstack((W_layer1,W_layer1))

    X = Input()
    W1, b1 = Input(), Input()
    conv1 = Conv(X ,W1, b1, (8,8) , kernel_size, (1,1))
    activation_1 = Relu(conv1)

    feed_dict = {
        X: X_,
        W1: W_layer1,
        b1: b_layer1,
    }

    graph = topological_sort(feed_dict)

    trainables = [W1, b1]

    forward_and_backward(graph)

    #print graph[-1].value




if __name__ == '__main__':
    main()
