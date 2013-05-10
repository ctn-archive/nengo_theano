
from _collections import OrderedDict
import theano
import numpy as np

class MapGemv(theano.Op):
    def __init__(self):
        pass

    def make_node(self, alpha, A, X, beta, J):
        inputs = map(theano.tensor.as_tensor_variable,
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

simulation_time = theano.tensor.fscalar(name='simulation_time')


class Simulator(object):
    def __init__(self, network):
        self.network = network
        self.simulation_steps = 0

        if self.network.tick_nodes:
            raise ValueError('Simulator does not support',
                             ' networks with tick_nodes')

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
        self.step = theano.function([simulation_time], [],
                                    updates=updates.items())

    def run_steps(self, N):
        for i in xrange(N):
            simulation_time = self.simulation_steps * self.network.dt
            self.step(simulation_time)
            self.simulation_steps += 1


    def run(self, approx_sim_time):
        n_steps = int(approx_sim_time / self.network.dt)
        return self.run_steps(n_steps)


