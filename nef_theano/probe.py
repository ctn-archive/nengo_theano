
import numpy as np
import theano
import theano.tensor as TT
import collections

class Probe(object):
    """A class to record from things (i.e., origins)"""
    buffer_size = 1000

    def __init__(self, name, network, target, dt_sample, pstc=0.03):
        self.name = name
        self.target = target
        self.i = -1
        self.t = 0
        self.dt = network.dt
        self.dt_sample = dt_sample
        self.pstc = pstc

        target_value = np.zeros_like(target.get_value())
        self.data = np.zeros((self.buffer_size,) + target_value.shape)
        self.filtered_data = theano.shared(target_value) if self.pstc > 0 else None

    def update(self):
        if self.filtered_data is not None:
            alpha = TT.cast(self.dt/self.pstc , dtype='float32')
            filtered_value = self.filtered_data + alpha*(self.target - self.filtered_data)
            return collections.OrderedDict({self.filtered_data: filtered_value})
        else:
            return collections.OrderedDict()

    def theano_tick(self):
        i_samp = int(np.floor(self.t/self.dt_sample))
        if i_samp > self.i:
            self.i = i_samp
            # we're as close to a sample point as we're going to get, so take a sample
            if i_samp >= len(self.data): 
                self.data = np.vstack(  # increase the buffer
                    [self.data, np.zeros((self.buffer_size,) + self.data.shape[1:])])

            if self.filtered_data is not None:
                self.data[i_samp] = self.filtered_data.get_value()
            else:
                self.data[i_samp] = self.target.get_value()

    def get_data(self):
        return self.data[:self.i+1]

