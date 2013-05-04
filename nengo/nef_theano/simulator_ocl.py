
from _collections import OrderedDict
import theano
import numpy as np
import pyopencl as cl

import simulator

ocl_perform = {}
ocl_alloc = {}

def perform(op_cls):
    def deco(f):
        ocl_perform[op_cls] = f
        return f
    return deco


def alloc(op_cls):
    def deco(f):
        ocl_alloc[op_cls] = f
        return f
    return deco


class SimulatorOCL(object):
    def __init__(self, network, context=None):
        self.network = network
        if context is None:
            context = cl.create_some_context()
        self.context = context
        self.queue = cl.CommandQueue(context)

        # dictionary for all variables
        # and the theano description of how to compute them 
        updates = OrderedDict()

        # for every node in the network
        for node in self.network.nodes.values():
            # if there is some variable to update
            if hasattr(node, 'update'):
                # add it to the list of variables to update every time step
                updates.update(node.update(self.network.dt))

        # create graph and return optimized update function
        self.step = theano.function([], [], updates=updates.items())

        self.order = self.step.maker.fgraph.toposort()

        thunks = []
        ocl_vars = {}

        for node in self.order:
            ocl_thunk = ocl_ops[type(node.op)](ocl_vars, node, queue)
            thunks.append(ocl_thunk)

    def run_steps(self, N):
        # -- copy from shared variable inputs into internal state dict
        # -- run N steps
        # -- copy from state to shared variables
        for i in xrange(N):
            self.step()


    def run(self, approx_sim_time):
        n_steps = int(approx_sim_time / self.network.dt)
        return self.run_steps(n_steps)

@alloc(simulator.MapGemv)
def ocl_map_gemv(ocl_vars, node, queue):
    J = ocl_vars[node.inputs[-1]]
    ocl_vars[node.outputs[0]] = cl.array.empty_like(J)

@perform(simulator.MapGemv)
def ocl_map_gemv(ocl_vars, node, queue):
    fn = cl.Program(queue.context, """
    """ % locals()).build().fn

    args = (queue, (size,), None, A.data, X.data, J.data)
    return fn, args

        alpha, A, X, beta, J = inputs

        J = J.copy() # TODO: inplace

        J *= beta
        for i in range(len(J)):
            J[i] += alpha * np.dot(X[i], A[i].T)
        outstor[0][0] = J


