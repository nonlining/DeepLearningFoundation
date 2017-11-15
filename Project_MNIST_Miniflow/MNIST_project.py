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
    np.random.seed(1)
    # load data
    dataset = {}
    dataset['train_img'] =  load_img(key_file['train_img'])
    dataset['train_label'] = load_label(key_file['train_label'])
    dataset['test_img'] = load_img(key_file['test_img'])
    dataset['test_label'] = load_label(key_file['test_label'])

    X_ = normalized(dataset['train_img'])
    X_test = normalized(dataset['test_img'])

    y_ = one_hot_encoding(dataset['train_label'])
    #y_test = one_hot_encoding(dataset['test_label'])
    y_test = dataset['test_label']

    # parameters
    fitter_numbers = 16
    kernel_size = (3,3)

    # init layers

    W_layer1 = np.sqrt(2.0/(1*3*3)) * np.random.randn(fitter_numbers, 1, kernel_size[0], kernel_size[1])
    b_layer1 = np.zeros(fitter_numbers)

    W_layer2 = np.sqrt(2.0/(16*3*3)) * np.random.randn(fitter_numbers, 16, kernel_size[0], kernel_size[1])
    b_layer2 = np.zeros(fitter_numbers)

    fitter_numbers = 32

    W_layer3 = np.sqrt(2.0/(16*3*3)) * np.random.randn(fitter_numbers, 16, kernel_size[0], kernel_size[1])
    b_layer3 = np.zeros(fitter_numbers)

    W_layer4 = np.sqrt(2.0/(32*3*3)) * np.random.randn(fitter_numbers, 32, kernel_size[0], kernel_size[1])
    b_layer4 = np.zeros(fitter_numbers)

    fitter_numbers = 64

    W_layer5 = np.sqrt(2.0/(32*3*3)) * np.random.randn(fitter_numbers, 32, kernel_size[0], kernel_size[1])
    b_layer5 = np.zeros(fitter_numbers)

    W_layer6 = np.sqrt(2.0/(64*3*3)) * np.random.randn(fitter_numbers, 64, kernel_size[0], kernel_size[1])
    b_layer6 = np.zeros(fitter_numbers)

    W_layer7 = np.sqrt(2.0/(64*4*4)) * np.random.randn(64*4*4, 50)
    b_layer7 = np.zeros(50)

    W_layer8 = np.sqrt(2.0 / 50) * np.random.randn(50, 10)
    b_layer8 = np.zeros(10)


    # network
    X, y = Input(), Input()
    W1, b1 = Input(), Input()
    W2, b2 = Input(), Input()
    W3, b3 = Input(), Input()
    W4, b4 = Input(), Input()
    W5, b5 = Input(), Input()
    W6, b6 = Input(), Input()
    W7, b7 = Input(), Input()
    W8, b8 = Input(), Input()

    conv_layer1 = Conv(X, W1, b1, (1,1), 1)

    activation_1 = Relu(conv_layer1)

    conv_layer2 = Conv(activation_1, W2, b2, (1,1), 1)

    activation_2 = Relu(conv_layer2)

    pooling1 = Pooling(activation_2, (2,2), (2,2), 0)

    conv_layer3 = Conv(pooling1, W3, b3, (1,1), 1)

    activation_3 = Relu(conv_layer3)

    conv_layer4 = Conv(activation_3, W4, b4, (1,1), 2)

    activation_4 = Relu(conv_layer4)

    pooling2 = Pooling(activation_4, (2,2), (2,2), 0)

    conv_layer5 = Conv(pooling2, W5, b5, (1,1), 1)

    activation_5 = Relu(conv_layer5)

    conv_layer6 = Conv(activation_5, W6, b6, (1,1), 1)

    activation_6 = Relu(conv_layer6)

    pooling3 = Pooling(activation_6, (2,2), (2,2), 0)

    linear1 = Linear(pooling3, W7, b7)

    activation_7 = Relu(linear1)

    dropout1 = Dropout(activation_7, 0.5)

    linear2 = Linear(dropout1, W8, b8)

    dropout2 = Dropout(linear2, 0.5)

    output = Softmax(dropout2, y)

    feed_dict = {
        X: X_,
        y: y_,
        W1:W_layer1,
        b1:b_layer1,
        W2:W_layer2,
        b2:b_layer2,
        W3:W_layer3,
        b3:b_layer3,
        W4:W_layer4,
        b4:b_layer4,
        W5:W_layer5,
        b5:b_layer5,
        W6:W_layer6,
        b6:b_layer6,
        W7:W_layer7,
        b7:b_layer7,
        W8:W_layer8,
        b8:b_layer8
    }

    graph = topological_sort(feed_dict)

    trainables = [W1, b1, W2, b2, W3, b3, W4, b4, W5, b5, W6, b6, W7, b7, W8, b8]
    epochs = 5
    learning_rate=1e-2
    train_size = X_.shape[0]
    test_size = X_test.shape[0]
    batch_size = 100

    steps_per_epoch = int(train_size/batch_size)

    loss_list = []

    for i in range(epochs):
        loss = 0
        index = 0
        while (index + 1)*100 <= train_size:

            #batch_mask = np.random.choice(train_size, batch_size)
            X_batch = X_[index*100: (index+1)*100]
            y_batch = y_[index*100: (index+1)*100]

            X.value = X_batch
            y.value = y_batch

            forward_and_backward(graph)
            sgd_update(trainables)
            if (index + 1)%100 == 0:
                print index+1,'/',steps_per_epoch,':',graph[-1].loss
            loss += graph[-1].loss
            index += 1

        print("Epoch: {}, Loss: {:.3f}".format(i+1, loss/float(steps_per_epoch)))
        loss_list.append(loss/steps_per_epoch)

        batch_mask = np.random.choice(test_size, 100)
        X_batch_test = X_test[batch_mask]
        y_batch_test = y_test[batch_mask]
        X.value = X_batch_test
        res = predict(graph)
        curr_num = np.sum(y_batch_test == np.argmax(res, axis=1))
        print "Epoch: {}, test acc: {:.3f}".format(i+1, curr_num/float(100))


    X.value = X_test
    res = predict(graph)

    curr_num = np.sum(y_test == np.argmax(res, axis=1))
    print "test acc for all test data : {:.3f} ".format(curr_num/float(test_size))
    # test acc: 0.969 for 5 epochs



if __name__ == '__main__':
    main()
