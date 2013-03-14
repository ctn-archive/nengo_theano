from theano import tensor as TT
from numbers import Number
import theano
import numpy
import origin

class Input:
    def __init__(self, name, value, zero_after=None):
        """A function input object

        :param string name: name of the function input
        :param value: defines the output decoded_output
        :type value: float or function
        :param float zero_after: time after which to set function output = 0 (s)
        """
        self.name = name
        self.t = 0
        self.function = None
        self.zero_after = zero_after
        self.zeroed = False
        self.origin = {} # dictionary of origins
        
        if callable(value): # if value parameter is a python function
            self.origin['X'] = origin.Origin(func=value)
        else:
            # if isinstance(value, Number): value = [value] # if scalar, make it a list
            self.origin['X'] = origin.Origin(func=None, initial_value=value)

    def reset(self):
        """Resets the function output state values
        """
        self.zeroed = False

    def theano_tick(self):
        """Move function input forward in time
        """
        if self.zeroed: return

        if self.zero_after is not None and self.t > self.zero_after: # zero output
            self.origin['X'].decoded_output.set_value(numpy.float32(numpy.zeros(self.origin['X'].dimensions)))
            self.zeroed=True

        if self.origin['X'].func is not None: # update output decoded_output
            value = self.origin['X'].func(self.t)
            # if value is a scalar output, make it a list
            if isinstance(value, Number): value = [value] 
            # cast as float32 for consistency / speed, but _after_ it's been made a list
            self.origin['X'].decoded_output.set_value(numpy.float32(value)) 
