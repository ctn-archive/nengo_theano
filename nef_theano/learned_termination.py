import numpy
import neuron
import collections
import numpy as np

class Learned_Termination():
    """This is the superclass learned_termination that attaches to an ensemble"""

    def __init__(self, pre, post, error, weight_matrix):
        """
        :param Ensemble pre: the pre-synaptic ensemble
        :param Ensemble ensemble: the post-synaptic ensemble (to which this termination is attached)
        :param error: the source of the error signal that directs learning
        :type error: Ensemble, Input, or SimpleNode
        :param np.array weight_matrix: the connection weight matrix between pre and post
        """
        self.pre_spikes = pre.neurons.output # the theano instantaneous spike raster of the pre-synaptic neurons
        self.post_spikes = post.neurons.output # the theano instantaneous spike raster of the post-synaptic neurons
        self.error = error 
        self.weight_matrix = weight_matrix # the theano object that holds the connection weights (post.neurons_num x pre.neurons_num)
    
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
        
class Null_Learned_Termination(Learned_Termination):
    def learn(self):
        return self.weight_matrix
