import collections

import numpy as np
import theano
import theano.tensor as TT

from . import neuron
from .learned_termination import LearnedTermination

class hPESTermination(LearnedTermination):
    # learning_rate = 5e-7      # from nengo
    # theta_tau = 20.           # from nengo
    # scaling_factor = 20e3     # from nengo
    # supervision_ratio = 0.5   # from nengo

    learning_rate = TT.cast(5e-3, dtype='float32')
    theta_tau = 0.02
    scaling_factor = 10.
    supervision_ratio = 1.0

    def __init__(self, *args, **kwargs):
        super(hPESTermination, self).__init__(*args, **kwargs)

        # get the theano instantaneous spike raster
        # of the pre(post)-synaptic neurons
        self.pre_spikes = self.pre.neurons.output
        self.post_spikes = self.post.neurons.output
        # get the decoded error signal
        self.error_value = self.error.decoded_output

        # get gains (alphas) for post neurons
        self.encoders = self.post.encoders.astype('float32')
        self.gains = np.sqrt(
            (self.post.encoders ** 2).sum(axis=-1)).astype('float32')

        self.initial_theta = np.random.uniform(
            low=5e-5, high=15e-5,
            size=(self.post.array_size, self.post.neurons_num)).astype('float32')
        # Trevor's assumption: high gain -> high theta
        self.initial_theta *= self.gains
        self.theta = theano.shared(self.initial_theta, name='hPES.theta')

        self.pre_filtered = theano.shared(
            self.pre_spikes.get_value(), name='hPES.pre_filtered')
        self.post_filtered = theano.shared(
            self.post_spikes.get_value(), name='hPES.post_filtered')

    def reset(self):
        super(hPESTermination, self).reset()
        self.theta.set_value(self.initial_theta)

    def learn(self):
        # get the error as represented by the post neurons
        encoded_error = np.sum(self.encoders * self.error_value[None,:], axis=-1)
        print 'encoded_error.eval().shape', encoded_error.eval().shape

        print 'self.pre_filtered[0].eval().shape', self.pre_filtered[0].eval().shape
        print 'post.array_size', self.post.array_size
        supervised_rate = self.learning_rate
        #TODO: more efficient rewrite with theano batch command? 
        delta_supervised = [(supervised_rate * self.pre_filtered[0][:,None] * 
                             encoded_error[i]) for i in range(self.post.array_size)]

        unsupervised_rate = TT.cast(
            self.learning_rate * self.scaling_factor, dtype='float32')
        #TODO: more efficient rewrite with theano batch command? 
        print 'theta.size', self.theta.eval().shape
        print 'gains.shape', self.gains.shape
        print 'post stuff', (self.post_filtered[0] * (self.post_filtered[0] - self.theta) * self.gains[0]).eval().shape
        delta_unsupervised = [(unsupervised_rate * self.pre_filtered[0][None,:] * 
                             (self.post_filtered[i] * 
                             (self.post_filtered[i] - self.theta[i]) * 
                              self.gains[i])[:,None] ) for i in range(self.post.array_size)] 

        print 'delta_supervised', delta_supervised
        print 'delta_supervised[0].eval().shape', delta_supervised[0].eval().shape
        print 'self.weight_matrix.eval().shape', self.weight_matrix.eval().shape
        new_wm = (self.weight_matrix
                + TT.cast(self.supervision_ratio, 'float32') * delta_supervised
                + TT.cast(1. - self.supervision_ratio, 'float32')
                * delta_unsupervised)
        print 'new_wm.eval().shape', new_wm.eval().shape

        #new_wm = TT.unbroadcast(new_wm, 0)

        return new_wm
        
    def update(self):
        # update filtered inputs
        alpha = TT.cast(self.dt / self.pstc, dtype='float32')
        new_pre = self.pre_filtered + alpha * (
            self.pre_spikes.flatten() - self.pre_filtered)
        new_post = self.post_filtered + alpha * (
            self.post_spikes - self.post_filtered)

        # update theta
        alpha = TT.cast(self.dt / self.theta_tau, dtype='float32')
        new_theta = self.theta + alpha * (new_post - self.theta)

        print 'self.pre_filtered.eval().shape', self.pre_filtered.eval().shape
        print 'new_pre.eval().shape', new_pre.eval().shape

        return collections.OrderedDict({
                self.weight_matrix: self.learn(),
                self.pre_filtered: new_pre, 
                self.post_filtered: new_post,
                self.theta: new_theta,
                })
