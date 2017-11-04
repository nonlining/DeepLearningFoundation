#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      MJWang
#
# Created:     03/11/2017
# Copyright:   (c) MJWang 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
from miniflow import *
import time


def main():
    X_data = [[0.,0.], [1.,0.] ,[0.,1.] ,[1.,1.]]

    X_data = np.array(X_data)

    y_label = [[0., 1., 1., 0.]]
    y_label = np.array(y_label)

    W_layer1 = np.random.randn(2,2)
    b_layer1 = np.random.randn(2)
    W_layer2 = np.random.randn(2,1)
    b_layer2 = np.random.randn(1)

    X, y = Input(), Input()
    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()

    L1 = Linear(X, W1, b1)
    S1 = Sigmoid(L1)

    L2 = Linear(S1, W2, b2)

    S2 = Sigmoid(L2)

    cost = MSE(y, S2)

    feed_dict = {
        X: X_data,
        y: y_label,
        W1: W_layer1,
        b1: b_layer1,
        W2: W_layer2,
        b2: b_layer2,
    }

    graph = topological_sort(feed_dict)

    trainables = [W1, b1, W2, b2]

    epochs = 500000
    learning_rate=0.1
    for i in range(epochs):

        forward_and_backward(graph)
        for t in trainables:
            partial = t.gradients[t]
            t.value -= learning_rate * partial
        if i%10000 == 0:
            print "epoch",i,"MSE",graph[-1].value

    print "W", W1.value
    print "B",b1.value[0]

    X.value = np.array([[1,1], [1,0], [0,1],[0,0]])
    res = predict(graph)
    print graph[-3].value
    print res


if __name__ == '__main__':
    tStart = time.time()
    main()
    tEnd = time.time()
    print tEnd - tStart