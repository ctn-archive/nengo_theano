import numpy
import theano
from numbers import Number

class Origin(object):
    """A basic Origin, promising a set of class variables to any accessing objects.
    """
    def __init__(self, func, initial_value=None):
        """
        
        :param function func: the function carried out by this origin
        :param array initial_value: the initial_value of the decoded_output
        """
        self.func = func

        if initial_value is None:
            initial_value = self.func(0.0) # initial output value = function value with input 0.0
            if isinstance(initial_value, Number): initial_value = [initial_value] # if scalar, make it a list

        # theano internal state defining output value
        self.decoded_output = theano.shared(numpy.float32(initial_value)) 

        # find number of parameters of the projected value
        self.dimensions = len(initial_value)
