import collections

import numpy as np
import theano
import theano.tensor as TT

from .filter import Filter

class Probe(object):
    """A class to record from things (i.e., origins).

    """
    buffer_size = 1000

    def __init__(self, name, network, target, dt_sample, pstc=0.03):
        self.name = name
        self.target = target
        # self.t = 0
        self.dt_sample = dt_sample

        # create array to store the data over many time steps
        self.data = np.zeros((self.buffer_size,) + target.get_value().shape)
        self.i = -1 # index of the last sample taken

        # create a filter to filter the data
        self.filter = Filter(network.dt, pstc, source=target)

    def update(self):
        return self.filter.update()

    def theano_tick(self):
        i_samp = int(np.floor(self.t / self.dt_sample))
        if i_samp > self.i:
            # we're as close to a sample point as we're going to get,
            # so take a sample
            if i_samp >= len(self.data):
                # increase the buffer
                self.data = np.vstack(
                    [self.data, np.zeros((self.buffer_size,)
                                         + self.data.shape[1:])])

            # record the filtered value
            self.data[i_samp] = self.filter.value.get_value()
            self.i = i_samp

    def get_data(self):
        return self.data[:self.i+1]
