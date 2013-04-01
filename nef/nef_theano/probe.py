import collections

import numpy as np
import theano
import theano.tensor as TT

from .filter import Filter

class Probe(object):
    """A class to record from things (i.e., origins).
    """
    def __init__(self, name, network, target, target_name, dt_sample, pstc=0.03):
        """
        :param string name:
        :param Network network:
        :param target:
        :type target: theano object
        :param float dt_sample:
        :param float pstc:
        """
        self.name = name
        self.target = target
        self.target_name = target_name
        self.dt_sample = dt_sample

        # create array to store the data over many time steps
        self.data = [] #np.zeros((self.buffer_size,) + target.get_value().shape)
        self.i = -1 # index of the last sample taken

        # create a filter to filter the data
        self.filter = Filter(network.dt, pstc, source=target)

    def update(self):
        """
        """
        return self.filter.update()

    def theano_tick(self):
        """
        """
        i_samp = int(np.floor(self.t / self.dt_sample))
        if i_samp > self.i:
            # we're as close to a sample point as we're going to get,
            # so take a sample
            self.data.append(self.filter.value.get_value()) 
            self.i = i_samp

    def get_data(self):
        """Return the collected data as numpy array
        """
        return np.array(self.data[:self.i+1])
