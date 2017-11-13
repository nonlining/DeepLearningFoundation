import numpy as np
import math

class Node:

    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.value = None
        self.outbound_nodes = []
        self.gradients = {}
        for node in inbound_nodes:
            node.outbound_nodes.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Input(Node):

    def __init__(self):
        Node.__init__(self)

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            self.gradients[self] += n.gradients[self]

class Linear(Node):

    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])
        self.ori_shape = None

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.ori_shape = X.shape

        X = X.reshape(self.inbound_nodes[0].value.shape[0], -1)
        self.value = np.dot(X, W) + b

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value, dtype = np.float64) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]

            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T).reshape(*self.ori_shape)
            if len(self.inbound_nodes[0].value.shape) > 2:
                temp = self.inbound_nodes[0].value.reshape(self.inbound_nodes[0].value.shape[0], self.inbound_nodes[0].value.shape[1], -1)
                #self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T.transpose(1,0,2), grad_cost).reshape(-1, self.inbound_nodes[1].value.shape[-1])
                self.gradients[self.inbound_nodes[1]] += np.dot(temp.T.transpose(1,0,2), grad_cost).reshape(-1, self.inbound_nodes[1].value.shape[-1])
            else:
                self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost).reshape(-1, self.inbound_nodes[1].value.shape[-1])
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)



class Conv(Node):
    def __init__(self, X, W, b, strides, padding):
        Node.__init__(self, [X, W, b])
        self.strides = strides
        self.padding = padding
        self.input_shape = None
        self.kernel_size = None
        self.Channel = None
        self.ori_shape = None


    def conv(self, X, shape, kernels, kernel_size, strides, b):
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
                        res += np.dot(kernel[k*kernel_size[0]:(k+1)*kernel_size[0]], X[x + y*shape[1] + shape[1]*k  :x + y*shape[1] + kernel_size[0] + shape[1]*k])
                    c[l][count] = res + b[l]
                    count += 1
                    x = x + strides[1]
                x = 0
                y = y + strides[0]
            l += 1
        return c

    def _conv2(self, X, input_shape, kernels, kernel_size, stride, padding, b):

        Num, Ch, Height, Width = X.shape

        out_height = (Height + 2*padding - kernel_size[0])//stride[0] + 1
        out_width  = (Width + 2*padding - kernel_size[1])//stride[1] + 1

        input = np.pad(X, [(0,0), (0,0), (padding, padding), (padding, padding)], 'constant')

        unroll_x = np.zeros((Num, Ch, kernel_size[0], kernel_size[1], out_height, out_width))

        for y in range(kernel_size[0]):
            y_max = y + stride[0]*out_height
            for x in range(kernel_size[1]):
                x_max = x + stride[1]*out_width
                unroll_x[:, :, y, x, :, :] = input[:,:,y:y_max:stride[0], x:x_max:stride[1]]


        unroll_x = unroll_x.transpose(0, 4, 5, 1, 2, 3).reshape(Num*out_height*out_width, -1)

        kernel_trans = kernels.reshape(kernels.shape[0], -1)

        res = np.dot(unroll_x, kernel_trans.T) + b
        res = res.reshape(Num, out_height*out_width, -1).transpose(0, 2, 1)
        res = res.reshape(Num, res.shape[1], out_height, out_width)


        return res, unroll_x

    def _conv2input(self, X, input_shape, kernel_size, stride, padding):
        Num, Ch, Height, Width = input_shape

        out_height = (Height + 2*padding - kernel_size[0])//stride[0] + 1
        out_width =  (Width + 2*padding - kernel_size[1])//stride[1] + 1

        unroll_x = X.reshape(Num, out_height, out_width, Ch, kernel_size[0], kernel_size[1]).transpose(0, 3, 4, 5, 1, 2)

        t = np.zeros((Num, Ch, Height + 2*padding + stride[0] - 1, Width + 2*padding + stride[1] - 1))

        for y in range(kernel_size[0]):
            y_max = y + stride[0]*out_height
            for x in range(kernel_size[1]):
                x_max = x + stride[1]*out_width
                t[:, :, y:y_max:stride[0], x:x_max:stride[1]] += unroll_x[:, :, y, x, :, :]

        t = t[:, :, padding:Height + padding, padding:Width + padding]
        t = t.reshape(Num, Ch, Height*Width)

        return t

    '''
    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        self.value = None
        for x in X:
            a = self.conv(x, self.input_shape, W, self.kernel_size, self.strides, b)

            if self.value is None:
                self.value = a
            else:
                self.value = np.dstack( (self.value ,a))
        self.value = self.value.transpose(2, 0, 1)
    '''

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value

        self.ori_shape = X.shape
        self.Num = X.shape[0]
        self.Channel = X.shape[1]
        self.kernel_size = (W.shape[2], W.shape[3])
        self.input_shape = (X.shape[2], X.shape[3])

        self.value ,self.col = self._conv2(X, self.input_shape, W, self.kernel_size, self.strides, self.padding ,b)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value, dtype = np.float64) for n in self.inbound_nodes}
        for n in self.outbound_nodes:

            W_flatten = self.inbound_nodes[1].value.reshape(self.inbound_nodes[1].value.shape[0], -1)

            fitter_numbers, filter_Size = W_flatten.shape

            grad_cost = n.gradients[self]

            grad_cost = grad_cost.reshape(grad_cost.shape[0], grad_cost.shape[1], -1)

            grad_out = grad_cost.transpose(0,2,1).reshape(-1, fitter_numbers)

            col = np.dot(grad_out, W_flatten)

            out = self._conv2input(col, self.ori_shape, self.kernel_size, self.strides, self.padding)

            self.gradients[self.inbound_nodes[0]] += out.reshape(self.Num , self.Channel,*self.input_shape)
            self.gradients[self.inbound_nodes[1]] += np.dot(grad_out.T, self.col).reshape(-1, self.Channel,*self.kernel_size)
            self.gradients[self.inbound_nodes[2]] += np.sum(np.sum(grad_cost, axis=2, keepdims=False), axis = 0,keepdims=False)


class Sigmoid(Node):

    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbound_nodes[0]] += sigmoid * (1 - sigmoid) * grad_cost


class Relu(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self):
        input_value = self.inbound_nodes[0].value
        self.mask = (input_value <= 0)
        input_value[self.mask] = 0
        self.value = input_value


    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            grad_cost[self.mask] = 0
            d = grad_cost
            self.gradients[self.inbound_nodes[0]] += d



class MSE(Node):

    def __init__(self, y, a):
        Node.__init__(self, [y, a])

    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)

        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a

        self.value = np.mean(self.diff**2)

    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff

class Dropout(Node):

    def __init__(self, x, ratio):
        Node.__init__(self, [x])
        self.dropout_ratio = ratio
        self.mask = None

    def forward(self, train_flg=True):
        x = self.inbound_nodes[0].value
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            self.value = x * self.mask
        else:
            self.value = x * (1.0 - self.dropout_ratio)

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] = grad_cost*self.mask

class Pooling(Node):
    def __init__(self, x, pooling_size, strides, padding):
        Node.__init__(self, [x])
        self.pooling_size = pooling_size
        self.stride = strides
        self.padding = padding
        self.value = None
        self.Num = None
        self.max_index = None
        self.ori_shape = None

    def forward(self):
        X = self.inbound_nodes[0].value

        Num, Ch, Height, Width = X.shape
        self.ori_shape = X.shape
        self.Num = Num

        out_height = (Height + 2*self.padding - self.pooling_size[0])/self.stride[0] + 1
        out_width  = (Width + 2*self.padding - self.pooling_size[1])/self.stride[1] + 1

        input = np.pad(X, [(0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')

        unroll_x = np.zeros((Num, Ch, self.pooling_size[0], self.pooling_size[1], out_height, out_width))

        for y in range(self.pooling_size[0]):
            y_max = y + self.stride[0]*out_height
            for x in range(self.pooling_size[1]):
                x_max = x + self.stride[1]*out_width
                unroll_x[:, :, y, x, :, :] = input[:, :, y:y_max:self.stride[0], x:x_max:self.stride[1]]

        unroll_x = unroll_x.transpose(0, 4, 5, 1, 2, 3)
        unroll_x = unroll_x.reshape(Num*out_height*out_width, -1)
        unroll_x = unroll_x.reshape(-1, self.pooling_size[0]*self.pooling_size[1])

        self.max_index = np.argmax(unroll_x, axis=1)
        self.value = np.max(unroll_x, axis=1)
        self.value = self.value.reshape(Num, out_height, out_width, Ch).transpose(0, 3, 1, 2)




    def backward(self):
        self.gradients = {n: np.zeros_like(n.value, dtype = np.float64) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            grad_cost = grad_cost.transpose(0, 2, 3, 1)
            out = np.zeros((grad_cost.size, self.pooling_size[0]*self.pooling_size[1]))

            out[np.arange(self.max_index.size), self.max_index.flatten()] = grad_cost.flatten()

            out = out.reshape(grad_cost.shape + (self.pooling_size[0]*self.pooling_size[1],))

            out = out.reshape(grad_cost.shape[0]*grad_cost.shape[1]*grad_cost.shape[2], -1)

            Num, Ch, Height, Width = self.ori_shape

            out_height = (Height + 2*self.padding - self.pooling_size[0])/self.stride[0] + 1
            out_width = (Width + 2*self.padding - self.pooling_size[1])/self.stride[1] + 1

            out = out.reshape(Num, out_height, out_width, Ch, self.pooling_size[0], self.pooling_size[1]).transpose(0, 3, 4, 5, 1, 2)

            unroll_x = np.zeros((Num, Ch, Height + 2*self.padding + self.stride[0] - 1, Width + 2*self.padding + self.stride[1] - 1))

            for y in range(self.pooling_size[0]):
                y_max = y + self.stride[0]*out_height
                for x in range(self.pooling_size[1]):
                    x_max = x + self.stride[1]*out_width
                    unroll_x[:, :, y:y_max:self.stride[0], x:x_max:self.stride[1]] += out[:, :, y, x, :, :]
            unroll_x = unroll_x[:, :, self.padding:Height + self.padding, self.padding:Width + self.padding]

            self.gradients[self.inbound_nodes[0]] += unroll_x



class Softmax(Node):

    def __init__(self, y, t):
        Node.__init__(self, [y, t])

    def _cross_entropy_error(self, y, t):

        if len(y.shape) == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        if t.size == y.size:
            t = t.argmax(axis=1)
        batch_size = y.shape[0]
        return -np.sum(np.log(y[np.arange(batch_size), t]))/batch_size

    def _soft_max(self, x):
        if len(x.shape) > 1:
            tmp = np.max(x, axis = 1)

            x -= tmp.reshape((x.shape[0], 1))
            x = np.exp(x)
            tmp = np.sum(x, axis = 1)
            x /= tmp.reshape((x.shape[0], 1))
        else:
            tmp = np.max(x)
            x -= tmp
            x = np.exp(x)
            tmp = np.sum(x)
            x /= tmp
        return x

    def forward(self):
        input_value = self.inbound_nodes[0].value
        target_value = self.inbound_nodes[1].value
        self.value = self._soft_max(input_value)
        self.loss = self._cross_entropy_error(self.value, target_value)

    def backward(self):
        batch_size = self.inbound_nodes[1].value.shape[0]

        if self.inbound_nodes[0].value.size == self.inbound_nodes[1].value.size:
            dx = (self.value - self.inbound_nodes[1].value) / batch_size

        self.gradients[self.inbound_nodes[0]] = dx
        self.gradients[self.inbound_nodes[1]] = dx


def topological_sort(feed_dict):

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]

    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()
        # feed data to input
        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)
    return L


def forward_and_backward(graph):

    for n in graph:
        n.forward()

    for n in graph[::-1]:
        n.backward()

def predict(graph):

    for n in graph[:-1]:
        if isinstance(n, Dropout):
            n.forward(False)
        else:
            n.forward()
    return graph[-2].value

def forward(graph):
    for n in graph:
        n.forward()


def sgd_update(trainables, learning_rate=1e-2):

    for t in trainables:

        partial = t.gradients[t]
        t.value -= learning_rate * partial

def normalized(x):
    return x/255.
