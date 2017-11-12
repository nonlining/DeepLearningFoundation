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
import numpy as np
from miniflow import *
import gzip

np.random.seed(1)

key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_label':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_label':'t10k-labels-idx1-ubyte.gz'
}


def load_img(file_name):
    file_path = "./" + file_name

    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)

    data = data.reshape(-1, 784)
    data = data.reshape(-1, 1, 28, 28)
    print("image done")

    return data

def load_label(file_name):
    file_path = "./" + file_name

    with gzip.open(file_path, 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
    print("load label done")

    return labels

def one_hot_encoding(y):
    n_values = np.max(y) + 1
    y = np.eye(n_values, dtype=int)[y]
    return y


def main():
    # load data
    dataset = {}
    dataset['train_img'] =  load_img(key_file['train_img'])
    dataset['train_label'] = load_label(key_file['train_label'])
    dataset['test_img'] = load_img(key_file['test_img'])
    dataset['test_label'] = load_label(key_file['test_label'])

    X_ = normalized(dataset['train_img'])

    y_ = one_hot_encoding(dataset['train_label'])


    # parameters
    fitter_numbers = 16
    kernel_size = (3,3)
    pooling_size = (2,2)

    # init layers

    W_layer1 = np.sqrt(2.0 / 3*3) * np.random.randn(fitter_numbers, 1, kernel_size[0], kernel_size[1])
    b_layer1 = np.zeros(fitter_numbers)

    W_layer2 = np.sqrt(2.0 /16*3*3) * np.random.randn(fitter_numbers, 16, kernel_size[0], kernel_size[1])
    b_layer2 = np.zeros(fitter_numbers)

    #W_layer3 = np.sqrt(2.0 /16*3*3) * np.random.randn(fitter_numbers, 16, kernel_size[0], kernel_size[1])
    #b_layer3 = np.zeros(fitter_numbers)

    W_layer3 = np.sqrt(2.0 /16*3*3) * np.random.randn(12544, 10)
    b_layer3 = np.zeros(10)


    # network
    X, y = Input(), Input()
    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()
    W3, b3 = Input(), Input()

    conv_layer1 = Conv(X, W1, b1, (1,1), 1)

    #pooling1 = Pooling(conv_layer1, pooling_size, (1,1), 0)

    activation_1 = Relu(conv_layer1)

    conv_layer2 = Conv(activation_1, W2, b2, (1,1), 1)

    activation_2 = Relu(conv_layer2)

    linear = Linear(activation_2, W3, b3)

    output = Softmax(linear, y)

    feed_dict = {
        X: X_,
        y: y_,
        W1: W_layer1,
        b1: b_layer1,
        W2:W_layer2,
        b2:b_layer2,
        W3:W_layer3,
        b3:b_layer3
    }

    graph = topological_sort(feed_dict)

    trainables = [W1, b1, W2, b2, W3, b3]
    epochs = 10
    learning_rate=1e-2
    train_size = X_.shape[0]
    batch_size = 100

    steps_per_epoch = int(train_size/batch_size)

    loss_list = []

    for i in range(epochs):
        loss = 0
        for j in range(steps_per_epoch):

            batch_mask = np.random.choice(train_size, batch_size)
            X_batch = X_[batch_mask]
            y_batch = y_[batch_mask]

            X.value = X_batch
            y.value = y_batch


            forward_and_backward(graph)
            sgd_update(trainables)
            print(graph[-1].loss)

        print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/steps_per_epoch))

        loss_list.append(loss/steps_per_epoch)



if __name__ == '__main__':
    main()
