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

np.random.seed(1)

def main():

    n_hidden = 8
    kernel_size = (3,3)

    W_layer1 = np.array([[1, 0, 1, 1, 0, 1, 1, 0, 1]])
    W_layer2 = np.random.normal(0, 0.1, (n_hidden, kernel_size[0]*kernel_size[1]))

    b_layer1 = np.zeros(n_hidden)

    print W_layer1, b_layer1


    X = [i+1 for i in range(64)]
    print X

    result = conv(X, (8,8) ,W_layer1, kernel_size, (1,1), b_layer1)
    print result

if __name__ == '__main__':
    main()
