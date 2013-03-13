
import numpy
import neuron
import collections
import numpy as np
import numpy.random

import theano
import theano.tensor as TT

class LearnedTermination(object):
    """This is the superclass learned_termination that attaches to an ensemble"""

    def __init__(self, pre, post, error, initial_weight_matrix, dt=1e-3, pstc=5e-3):
        """
        :param Ensemble pre: the pre-synaptic ensemble
        :param Ensemble ensemble: the post-synaptic ensemble (to which this termination is attached)
        :param error: the source of the error signal that directs learning
        :type error: Ensemble, Input, or SimpleNode
        :param np.array weight_matrix: the connection weight matrix between pre and post
        """
        self.dt = dt
        self.pstc = pstc

        self.pre = pre
        self.post = post
        self.error = error

        # initialize weight matrix
        self.initial_weight_matrix = initial_weight_matrix.astype('float32')
        self.weight_matrix = theano.shared(self.initial_weight_matrix, name='weight_matrix')

    def reset(self):
        self.weight_matrix.set_value(self.initial_weight_matrix)
    
    def learn(self): raise NotImplementedError()
    """The learning function, to be implemented by any specific learning subclasses.

    :returns The updated value for the weight matrix, as a Theano variable.
    """

    def update(self):
        """The updates to the weight matrix calculation.
        Returns a dictionary with the new weight_matrix.
        """
        # multiply the output by the attached ensemble's radius to put us back in the right range
        return collections.OrderedDict( {self.weight_matrix: self.learn()} ) 

        
class NullLearnedTermination(LearnedTermination):
    def learn(self):
        return self.weight_matrix


class hPESTermination(LearnedTermination):
    # learning_rate = 5e-7      # from nengo
    # theta_tau = 20.           # from nengo
    # scaling_factor = 20e3     # from nengo
    # supervision_ratio = 0.5   # from nengo

    learning_rate = 5e-3
    theta_tau = 0.02
    scaling_factor = 10.
    supervision_ratio = 1.0

    def __init__(self, *args, **kwargs):
        super(hPESTermination, self).__init__(*args, **kwargs)

        # get the theano instantaneous spike raster of the pre(post)-synaptic neurons
        self.pre_spikes = self.pre.neurons.output
        self.post_spikes = self.post.neurons.output
        self.error_value = self.error.origin['X'].decoded_output

        # get gains (alphas) for post neurons
        self.encoders = self.post.encoders.astype('float32')
        self.gains = np.sqrt((self.post.encoders**2).sum(axis=-1)).astype('float32')

        post_num = self.post.neurons_num
        self.initial_theta = np.random.uniform(
            low=5e-5, high=15e-5, size=post_num).astype('float32')
        self.initial_theta *= self.gains # Trevor's assumption: high gain -> high theta
        self.theta = theano.shared(self.initial_theta, name='theta')

        self.pre_filtered = theano.shared(self.pre_spikes.get_value().flatten(), name='pre_filtered')
        self.post_filtered = theano.shared(self.post_spikes.get_value().flatten(), name='post_filtered')

    def reset(self):
        super(hPESTermination, self).reset()
        self.theta.set_value(self.initial_theta)

    def learn(self):
        # get the error as represented by the post neurons
        encoded_error = (self.encoders * self.error_value[None,:]).sum(axis=-1)

        supervised_rate = TT.cast(self.learning_rate, dtype='float32')
        delta_supervised = supervised_rate * self.pre_filtered[None,:] * encoded_error[:,None]

        unsupervised_rate = TT.cast(self.learning_rate * self.scaling_factor, dtype='float32')
        delta_unsupervised = unsupervised_rate * self.pre_filtered[None,:] *\
            (self.post_filtered*(self.post_filtered-self.theta)*self.gains)[:,None]

        return self.weight_matrix +\
            TT.cast(self.supervision_ratio, 'float32')*delta_supervised +\
            TT.cast(1. - self.supervision_ratio, 'float32')*delta_unsupervised
        
    def update(self):
        # update filtered inputs
        alpha = TT.cast(self.dt/self.pstc, dtype='float32')
        new_pre = self.pre_filtered + alpha*(self.pre_spikes.flatten() - self.pre_filtered)
        new_post = self.post_filtered + alpha*(self.post_spikes.flatten() - self.post_filtered)

        # update theta
        alpha = TT.cast(self.dt/self.theta_tau, dtype='float32')
        new_theta = self.theta + alpha*(new_post - self.theta)

        return collections.OrderedDict(
            {self.weight_matrix: self.learn(),
             self.pre_filtered: new_pre,
             self.post_filtered: new_post,
             self.theta: new_theta})

