from _collections import OrderedDict
import theano; import theano.tensor as TT
import numpy as np

class MapGemv(theano.Op):
    def __init__(self):
        pass

    def make_node(self, alpha, A, X, beta, J):
        inputs = map(TT.as_tensor_variable,
            [alpha, A, X, beta, J])
        return theano.Apply(self, inputs, [inputs[-1].type()])

    def perform(self, node, inputs, outstor):
        alpha, A, X, beta, J = inputs

        J = J.copy() # TODO: inplace

        J *= beta
        for i in range(len(J)):
            J[i] += alpha * np.dot(X[i], A[i].T)
        outstor[0][0] = J

map_gemv = MapGemv()

