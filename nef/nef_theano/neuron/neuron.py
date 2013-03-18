import theano
from theano import tensor as TT
import numpy as np


def accumulate(input, neuron, time=1.0, init_time=0.05):
    """Accumulates neuron output over time.

    Take a neuron model, run it for the given amount of time with
    fixed input. Used to generate activity matrix when calculating
    origin decoders.
    
    Returns the accumulated output over that time.

    :param input: theano function object describing the input
    :param Neuron neuron: population of neurons from which to accumulate data
    :param float time: length of time to simulate population for (s)
    :param float init_time: run neurons for this long before collecting data
                            to get rid of startup transients (s)

    """
    # create internal state variable to keep track of number of spikes
    total = theano.shared(np.zeros(neuron.size).astype('float32'))

    ### make the standard neuron update function
    # updates its dictionary of variables returned by neuron.update
    updates = neuron.update(input.astype('float32'))

    # update all internal state variables listed in updates
    tick = theano.function([], [], updates=updates)

    ### make a variant that also includes computing the total output
    # add another internal variable to change to updates dictionary
    updates[total] = total + neuron.output

    # create theano function that does it all
    accumulate_spikes = theano.function([], [], updates=updates)
    #, mode=theano.Mode(optimizer=None, linker='py'))

    # call the standard one a few times to avoid startup transients
    tick.fn(n_calls = int(init_time / neuron.dt))

    # call the accumulator version a bunch of times
    accumulate_spikes.fn(n_calls = int(time / neuron.dt))

    return total.get_value().astype('float32') / time


class Neuron(object):
    """Superclass for neuron models.

    All neurons must implement an update function,
    and should most likely define a more complicated reset function.

    """

    def __init__(self, size, dt):
        """Constructor for neuron model superclass.

        :param int size: number of neurons in this population
        :param float dt: size of timestep taken during update

        """
        self.size = size
        self.dt = dt
        # set up theano internal state variable
        self.output = theano.shared(np.zeros(size).astype('float32'))

    def reset(self):
        """Reset the state of the neuron."""
        self.output.set_value(np.zeros(self.size).astype('float32'))

    def update(self, input_current):
        """All neuron subclasses must have an update function.

        The update function takes in input_current and returns
        activity information.

        """
        raise NotImplementedError()
