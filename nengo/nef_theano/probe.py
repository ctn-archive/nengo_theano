from _collections import OrderedDict

import numpy as np
import theano
import theano.tensor as TT
from .filter import Filter
import simulator


class Scribe(theano.Op):
    def __init__(self, destructive):
        self.destructive = destructive
        if destructive:
            raise NotImplementedError()

    def __hash__(self):
        return hash((type(self), ))

    def __eq__(self, other):
        return type(self) == type(other)

    def make_node(self, x, buf, i, t, dt_sample):
        x, buf, i, t, dt_sample = map(theano.tensor.as_tensor_variable,
                             [x, buf, i, t, dt_sample])
        # TODO: more accurate broadcasting output pattern
        return theano.Apply(self,
                [x, buf, i, t, dt_sample],
                [buf.type(), i.type()])

    def perform(self, node, inputs, outstor):
        x, buf, i, t, dt_sample = inputs
        i_samp = int(t / dt_sample)
        if i_samp > i:
            # we're as close to a sample point as we're going to get,
            # so take a sample
            if i_samp >= len(buf):
                # increase the buffer
                # XXX check vstack in 1D, 2D, 3D, etc.
                buf = np.vstack(
                    [buf, np.zeros((len(buf),) + buf.shape[1:])])
            elif not self.destructive:
                buf = buf.copy()
            # record the filtered value
            buf[i + 1:i_samp + 1] = x
            i = i_samp

        outstor[0][0] = buf
        outstor[1][0] = i


class Probe(object):
    """A class to record from things (i.e., origins).

    """
    buffer_size = 1000

    def __init__(self, name, target, target_name, dt_sample, pstc=0.03):
        """
        :param string name:
        :param target:
        :type target: 
        :param string target_name:
        :param float dt_sample:
        :param float pstc:
        """
        self.name = name
        self.target = target
        self.target_name = target_name
        self.dt_sample = dt_sample

        # create array to store the data over many time steps
        self.data = theano.shared(
            np.zeros((self.buffer_size,) + target.get_value().shape))

        self.i = theano.shared(-1)  # index of the last sample taken

        # create a filter to filter the data
        self.filter = Filter(name=name, pstc=pstc, source=target)

    def update(self, dt):
        """
        :param float dt: the timestep of the update
        """
        updates = self.filter.update(dt)
        new_data, new_i = Scribe(False)(self.filter.value,
                                           self.data, self.i,
                                           simulator.simulation_time, dt)
        updates[self.data] = new_data
        updates[self.i] = new_i
        return updates

    def get_data(self):
        """
        """
        return self.data.get_value(borrow=False)[:self.i.get_value() + 1]

