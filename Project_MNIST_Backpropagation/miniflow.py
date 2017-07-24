import numpy as np

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

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        print X.shape
        print W.shape
        self.value = np.dot(X, W) + b
        print self.value

    def backward(self):
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self.inbound_nodes[0]] += np.dot(grad_cost, self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] += np.dot(self.inbound_nodes[0].value.T, grad_cost)
            self.gradients[self.inbound_nodes[2]] += np.sum(grad_cost, axis=0, keepdims=False)


def conv(arr, shape, kernel, kernel_size):
    c = np.zeros((shape[0] - 2)*(shape[0] - 2))

    count = 0
    for j in range(shape[0] - 2):
        for i in range(shape[1] - 2):
            res = 0
            print kernel[0:3]
            res += kernel[0:3]*arr[i + j*shape[1]               :i + j*shape[1] + kernel_size[0]            ]
            res += kernel[3:6]*arr[i + j*shape[1] + shape[1]*1  :i + j*shape[1] + kernel_size[0]+ shape[1]*1]
            res += kernel[6:]*arr[i + j*shape[1] + shape[1]*2  :i + j*shape[1] + kernel_size[0]+ shape[1]*2]
            c[count] = np.sum(res)
            count += 1

    return c

class Conv(Node):
    def __init__(self, X, W, b):
        Node.__init__(self, [X, W, b])

    def forward(self):
        X = self.inbound_nodes[0].value
        W = self.inbound_nodes[1].value
        b = self.inbound_nodes[2].value
        print X.shape
        print W.shape
        print b.shape

        #self.value = conv(X, X.shape ,W , W.shape) + b
    def backward(self):
        pass


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





def cross_entropy():
    pass

def soft_max(list):
    return 1./(sum(list))

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

    for i in G:
        print type(i),i,G[i]


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


def forward(graph):
    for n in graph:
        n.forward()


def sgd_update(trainables, learning_rate=1e-2):

    for t in trainables:

        partial = t.gradients[t]
        t.value -= learning_rate * partial

