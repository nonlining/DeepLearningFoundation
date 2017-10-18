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


def conv2(arr, shape, kernels, kernel_size, strides, b):
    out_height = (shape[0] - kernel_size[0])/strides[0] + 1
    out_width  = (shape[1] - kernel_size[1])/strides[1] + 1


    c = np.zeros((kernels.shape[0] , out_height*out_width))

    l = 0
    for kernel in kernels:
        count = 0
        y = 0
        x = 0
        for j in range(out_height):
            for i in range(out_width):
                res = 0
                for k in range(0, kernel_size[1]):
                    res += np.dot(kernel[k*kernel_size[0]:(k+1)*kernel_size[0]], arr[x + y*shape[1] + shape[1]*k  :x + y*shape[1] + kernel_size[0] + shape[1]*k])
                c[l][count] = res + b[l]
                count += 1
                x = x + strides[1]
            x = 0
            y = y + strides[0]
        l += 1

    return c

def conv3(arr, shape, kernels, kernel_size, stride, b):
    N, H, W = arr.shape

    out_height = (shape[0] - kernel_size[0])/stride[0] + 1
    out_width  = (shape[1] - kernel_size[1])/stride[1] + 1

    img = np.pad(arr, [(0,0), (0, 0), (0, 0)], 'constant')

    col = np.zeros((N, kernel_size[0], kernel_size[1], out_height, out_width))

    c = np.zeros((kernels.shape[0] , out_height*out_width))
    for y in range(kernel_size[0]):
        y_max = y + stride[0]*out_height
        for x in range(kernel_size[1]):
            x_max = x + stride[1]*out_width
            col[:,y,x,:,:] = img[:,y:y_max:stride[0], x:x_max:stride[1]]

    col = col.transpose(0, 3, 4, 1, 2).reshape(N*out_height*out_width, -1)
    res = np.dot(kernels, col.T) + b[0]

    res = res.reshape(N, kernels.shape[0] ,out_height*out_height)

    print res


    return res

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):

    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    print col

    return col

def main():

    n_hidden = 8
    kernel_size = (3,3)

    W_layer1 = np.array([[1, 0, 1, 1, 0, 1, 1, 0, 1]])

    b_layer1 = np.zeros(n_hidden)

    X = [i+1 for i in range(64)]


    input = np.array(X).reshape(1, 1, 8, 8)

    start = time.time()


    result = conv2(X, (8,8) ,W_layer1, kernel_size, (1,1), b_layer1)

    print "conv2 shape",result.shape

    end = time.time()

    print end - start

    start = time.time()

    res = im2col(input, kernel_size[0], kernel_size[1])
    ans = np.dot(res, W_layer1.T)

    end = time.time()

    print end - start

    print "--------------"

    X = np.array(X).reshape(-1,64)

    for i in range(3):
        X = np.vstack((X,X))


    for i in range(3):
        W_layer1 = np.vstack((W_layer1,W_layer1))


    result2 = conv3(np.array(X).reshape(-1,8,8), (8,8) ,W_layer1, kernel_size, (1,1), b_layer1)
    print result2.shape



if __name__ == '__main__':
    main()
