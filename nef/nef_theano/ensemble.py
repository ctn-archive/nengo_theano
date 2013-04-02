import theano
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as TT
import numpy as np

from . import neuron
from . import ensemble_origin
from . import cache
from .hPES_termination import hPESTermination

class Accumulator:
    def __init__(self, ensemble, pstc):
        """A collection of terminations in the same population
        with the same time constant.

        Stores the decoded_input accumulated across these terminations;
        i.e. their summed contribution to the represented signal.
        Also stores the direct_input value, which is direct current
        input when connections are added with a weight matrix specified.

        :param Ensemble ensemble:
            the ensemble this set of terminations is attached to
        :param float pstc: post-synaptic time constant on filter

        """
        self.ensemble = ensemble

        # time constant for filter
        self.decay = np.exp(-self.ensemble.neurons.dt / pstc)
        self.decoded_total = None
        self.encoded_total = None
        self.learn_total = None

        # decoded_input should be dimensions * array_size
        # because we account for the transform matrix here,
        # so different array networks get different input
        self.decoded_input = theano.shared(np.zeros(
                (self.ensemble.array_size, self.ensemble.dimensions)
                ).astype('float32'), name='accumulator.decoded_input')
        # encoded_input specifies input into each neuron,
        # so it is array_size * neurons_num
        self.encoded_input = theano.shared(np.zeros(
                (self.ensemble.array_size, self.ensemble.neurons_num)
                ).astype('float32'), name='accumulator.encoded_input')
        # learn_input specifies input into each neuron,
        # but current from different terminations can't be amalgamated
        #TODO: make learn input a dictionary that stores a
        # shared variable of the input current
        # each different termination, for use by learned_termination
        self.learn_input = theano.shared(np.zeros(
                (self.ensemble.neurons.size)
                ).astype('float32'), name='accumulator.learn_input')

    def add_decoded_input(self, decoded_input):
        """Adds a decoded input to this accumulator.

        Adds an additional input to the current set of decoded inputs
        (with the same post-synaptic time constant pstc).
        self.new_decoded_input is the calculation of
        the contribution of all of the decoded input
        with the same filtering time constant to the ensemble.
        Input current is then calculated as the sum of all
        decoded_inputs * ensemble.encoders.

        :param decoded_input:
            theano object representing the output of the
            pre population multiplied by this termination's
            transform matrix
        """
        if self.decoded_total is None:
            # initialize internal value storing decoded input to neurons
            self.decoded_total = decoded_input 
        else:
            # add to the decoded input to neurons
            self.decoded_total = self.decoded_total + decoded_input 

        # the theano object representing the filtering operation
        self.new_decoded_input = self.decay * self.decoded_input + (
            1 - self.decay) * self.decoded_total 

    def add_encoded_input(self, encoded_input):
        """Adds an encoded input to this accumulator.

        Add an additional input to the current set of encoded inputs
        (with the same post-synaptic time constant pstc).
        self.new_encoded_input is the calculation of
        the contribution of all the encoded input
        with the same filtering time constant to the ensemble,
        where the encoded_input is exactly
        the input current to each neuron in the ensemble.

        :param encoded_input:
            theano object representing the decoded output of every
            neuron of the pre population * connection weight matrix
        """
        if self.encoded_total is None:
            # initialize internal value
            # storing encoded input (current) to neurons
            self.encoded_total = encoded_input 
        else:
            # add input encoded input (current) to neurons
            self.encoded_total = self.encoded_total + encoded_input 

        # the theano object representing the filtering operation
        self.new_encoded_input = self.decay * self.encoded_input + (
            1 - self.decay) * self.encoded_total

    def add_learn_input(self, learn_input):
        """Adds a learned input to this accumulator.

        Add an additional input to the current set of learned inputs
        (with the same post-synaptic time constant pstc).
        self.new_learn_input is the calculation of the
        contribution of all the learned inputs with the same
        filtering time constant to the ensemble,
        where the learn_input is exactly the input current
        to each neuron in the ensemble.

        :param learn_input:
            theano object representing the current output of every
            neuron of the pre population * a connection weight matrix

        """

        if self.learn_total is None:
            # initialize internal value
            # storing learned encoded input (current) to neurons
            self.learn_total = learn_input 
        else:
            # add input learn input (current) to neurons
            self.learn_total = self.learn_total + learn_input 

        # the theano object representing the filtering operation
        self.new_learn_input = self.decay * self.learn_input + (
            1 - self.decay) * self.learn_total 


class Ensemble:
    """An ensemble is a collection of neurons representing a vector space.
    
    """
    
    def __init__(self, neurons, dimensions, tau_ref=0.002, tau_rc=0.02,
                 max_rate=(200, 300), intercept=(-1.0, 1.0), radius=1.0,
                 encoders=None, seed=None, neuron_type='lif', dt=0.001,
                 array_size=1, eval_points=None, decoder_noise=0.1,
                 noise_type='uniform', noise=None):
        """Construct an ensemble composed of the specific neuron model,
        with the specified neural parameters.

        :param int neurons: number of neurons in this population
        :param int dimensions:
            number of dimensions in the vector space
            that these neurons represent
        :param float tau_ref: length of refractory period
        :param float tau_rc:
            RC constant; approximately how long until 2/3
            of the threshold voltage is accumulated
        :param tuple max_rate:
            lower and upper bounds on randomly generated
            firing rates for each neuron
        :param tuple intercept:
            lower and upper bounds on randomly generated
            x offsets for each neuron
        :param float radius:
            the range of input values (-radius:radius)
            per dimension this population is sensitive to
        :param list encoders: set of possible preferred directions
        :param int seed: seed value for random number generator
        :param string neuron_type:
            type of neuron model to use, options = {'lif'}
        :param float dt: time step of neurons during update step
        :param int array_size: number of sub-populations for network arrays
        :param list eval_points:
            specific set of points to optimize decoders over by default
        :param float decoder_noise: amount of noise to assume when computing 
            decoder    
        :param string noise_type:
            the type of noise added to the input current.
            Possible options = {'uniform', 'gaussian'}.
            Default is 'uniform' to match the Nengo implementation.
        :param float noise:
            noise parameter for noise added to input current,
            sampled at every timestep.
            If noise_type = uniform, this is the lower and upper
            bound on the distribution.
            If noise_type = gaussian, this is the variance.

        """
        if seed is None:
            seed = np.random.randint(1000)
        self.seed = seed
        self.neurons_num = neurons
        self.dimensions = dimensions
        self.array_size = array_size
        self.radius = radius
        self.noise = noise
        self.decoder_noise=decoder_noise
        self.dt = dt
        self.noise_type = noise_type

        # make sure that eval_points is the right shape
        if eval_points is not None:
            eval_points = np.array(eval_points)
            if len(eval_points.shape) == 1:
                eval_points.shape = [1, eval_points.shape[0]]
        self.eval_points = eval_points

        self.cache_key = cache.generate_ensemble_key(neurons, dimensions, 
                     tau_rc, tau_ref, max_rate, intercept, radius, encoders, 
                     decoder_noise, eval_points, noise, seed)

        # create the neurons
        # TODO: handle different neuron types,
        # which may have different parameters to pass in
        self.neurons = neuron.types[neuron_type](
            (array_size, self.neurons_num),
            tau_rc=tau_rc, tau_ref=tau_ref, dt=dt)

        # compute alpha and bias
        self.srng = RandomStreams(seed=seed)
        max_rates = self.srng.uniform(
            size=(self.array_size, self.neurons_num),
            low=max_rate[0], high=max_rate[1])  
        threshold = self.srng.uniform(
            size=(self.array_size, self.neurons_num),
            low=intercept[0], high=intercept[1])
        alpha, self.bias = theano.function(
            [], self.neurons.make_alpha_bias(max_rates, threshold))()

        # force to 32 bit for consistency / speed
        self.bias = self.bias.astype('float32')
                
        # compute encoders
        self.encoders = self.make_encoders(encoders=encoders)
        # combine encoders and gain for simplification
        self.encoders = (self.encoders.T * alpha.T).T
        self.shared_encoders = theano.shared(self.encoders, 
            name='ensemble.shared_encoders')
        
        # make default origin
        self.origin = {}
        self.add_origin('X', func=None, eval_points=self.eval_points) 

        # dictionary of accumulators tracking terminations
        # with different pstc values
        self.accumulators = {}
        
        # list of learned terminations on ensemble
        self.learned_terminations = []

    def add_filtered_input(self, pstc, decoded_input=None,
                           encoded_input=None, learn_input=None):
        """Accounts for a new termination that takes the given input
        (a theano object) and filters it with the given pstc.

        Adds its contributions to the set of decoded, encoded,
        or learn input with the same pstc. Decoded inputs
        are represented signals, encoded inputs are
        decoded_output * weight matrix, learn input is
        activities * weight_matrix.

        Can only have one of decoded OR encoded OR learn input != None.

        :param float pstc: post-synaptic time constant
        :param decoded_input:
            theano object representing the decoded output of
            the pre population multiplied by this termination's
            transform matrix
        :param encoded_input:
            theano object representing the encoded output of
            the pre population multiplied by a connection weight matrix
        :param learn_input:
            theano object representing the learned output of
            the pre population multiplied by a connection weight matrix
        
        """
        # make sure one and only one of
        # (decoded_input, encoded_input, learn_input) is specified
        if decoded_input is not None:
            assert (encoded_input is None) and (learn_input is None)
        elif encoded_input is not None:
            assert (decoded_input is None) and (learn_input is None)
        elif learn_input is not None:
            assert (decoded_input is None) and (encoded_input is None)
        else:
            assert False

        # make sure there's an accumulator for given pstc
        if pstc not in self.accumulators:
            self.accumulators[pstc] = Accumulator(self, pstc)

        # add this termination's contribution to
        # the set of terminations with the same pstc
        if decoded_input:
            # rescale decoded_input by this neuron's radius
            # to put us in the right range
            self.accumulators[pstc].add_decoded_input(
                TT.true_div(decoded_input, self.radius))
        elif encoded_input:
            self.accumulators[pstc].add_encoded_input(encoded_input)
        elif learn_input:
            self.accumulators[pstc].add_learn_input(learn_input)

    def add_learned_termination(self, pre, error, pstc, weight_matrix=None,
                                learned_termination_class=hPESTermination):
        """Adds a learned termination to the ensemble.

        Accounting for the additional input_current is still done
        through the accumulator, but a learned_termination object
        is also created and attached to keep track of the pre and post
        (self) spike times, and adjust the weight matrix according
        to the specified learning rule.

        :param Ensemble pre: the pre-synaptic population
        :param Ensemble error: the Origin that provides the error signal
        :param list weight_matrix:
            the initial connection weights with which to start
        
        """
        #TODO: is there ever a case we wouldn't want this?
        assert error.dimensions == self.dimensions * self.array_size

        # generate an initial weight matrix if none provided,
        # random numbers between -.001 and .001
        if weight_matrix is None:
            weight_matrix = np.random.uniform(
                size=(self.array_size * pre.array_size,
                      self.neurons_num, pre.neurons_num),
                low=-.001, high=.001)
        else:
            # make sure it's an np.array
            #TODO: error checking to make sure it's the right size
            weight_matrix = np.array(weight_matrix) 

        learned_term = learned_termination_class(
            pre, self, error, weight_matrix)

        learn_projections = [TT.dot(
            pre.neurons.output[learned_term.pre_index(i)],  
            learned_term.weight_matrix[i % self.array_size]) 
            for i in range(self.array_size * pre.array_size)]

        # now want to sum all the output to each of the post ensembles 
        # going to reshape and sum along the 0 axis
        learn_output = TT.sum( 
            TT.reshape(learn_projections, 
            (pre.array_size, self.array_size, self.neurons_num)), axis=0)
        # reshape to make it (array_size x neurons_num)
        learn_output = TT.reshape(learn_output, 
            (self.array_size, self.neurons_num))

        # add learn output to the accumulator to handle
        # the input_current from this connection during simulation
        self.add_filtered_input(pstc=pstc, learn_input=learn_output)
        self.learned_terminations.append(learned_term)
        return learned_term

    def add_origin(self, name, func, eval_points=None):
        """Create a new origin to perform a given function
        on the represented signal.

        :param string name: name of origin
        :param function func:
            desired transformation to perform over represented signal
        :param list eval_points:
            specific set of points to optimize decoders over for this origin
        """
        if eval_points == None:
            eval_points = self.eval_points
        self.origin[name] = ensemble_origin.EnsembleOrigin(
            self, func, eval_points=eval_points)

    def make_encoders(self, encoders=None):
        """Generates a set of encoders.

        :param int neurons: number of neurons 
        :param int dimensions: number of dimensions
        :param theano.tensor.shared_randomstreams snrg:
            theano random number generator function
        :param list encoders:
            set of possible preferred directions of neurons

        """
        if encoders is None:
            # if no encoders specified, generate randomly
            encoders = self.srng.normal(
                (self.array_size, self.neurons_num, self.dimensions))
        else:
            # if encoders were specified, cast list as array
            encoders = np.array(encoders).T
            # repeat array until 'encoders' is the same length
            # as number of neurons in population
            encoders = np.tile(encoders,
                (self.neurons_num / len(encoders) + 1)
                               ).T[:self.neurons_num, :self.dimensions]
            encoders = np.tile(encoders, (self.array_size, 1, 1))

        # normalize encoders across represented dimensions 
        norm = TT.sum(encoders * encoders, axis=[2], keepdims=True)
        encoders = encoders / TT.sqrt(norm)        

        return theano.function([], encoders)()

    def update(self):
        """Compute the set of theano updates needed for this ensemble.

        Returns a dictionary with new neuron state,
        termination, and origin values.

        """
        
        ### find the total input current to this population of neurons

        # apply respective biases to neurons in the population 
        J = np.array(self.bias)
        # set up matrix to store accumulated decoded input,
        # same size as decoded_input
        X = np.zeros((self.array_size, self.dimensions))
    
        for a in self.accumulators.values(): 
            if hasattr(a, 'new_decoded_input'):
                # if there's a decoded input in this accumulator,
                # add its values to the total decoded input
                X += a.new_decoded_input 
            if hasattr(a, 'new_encoded_input'):
                # if there's an encoded input in this accumulator
                # add its values directly to the input current
                J += a.new_encoded_input 
            if hasattr(a, 'new_learn_input'):
                # if there's a learn input in this accumulator
                # add its values directly to the input current 
                J += a.new_learn_input

        # onlf do this if X is a theano object (i.e. there was decoded_input)
        if hasattr(X, 'type'):
            # add to input current for each neuron as
            # represented input signal x preferred direction
            #TODO: use TT.batched_dot function here instead?
            J = [J[i] + TT.dot(self.shared_encoders[i], X[i].T)
                 for i in range(self.array_size)]

        # if noise has been specified for this neuron,
        # add Gaussian white noise with variance self.noise to the input_current
        if self.noise: 
            # generate random noise values, one for each input_current element, 
            # with standard deviation = sqrt(self.noise=std**2)
            # When simulating white noise, the noise process must be scaled by
            # sqrt(dt) instead of dt. Hence, we divide the std by sqrt(dt).
            if self.noise_type.lower() == 'gaussian':
                J += self.srng.normal(
                    size=self.bias.shape, std=np.sqrt(self.noise/self.dt))
            elif self.noise_type.lower() == 'uniform':
                J += self.srng.uniform(
                    size=self.bias.shape, 
                    low=-self.noise/np.sqrt(self.dt), 
                    high=self.noise/np.sqrt(self.dt))

        # pass that total into the neuron model to produce
        # the main theano computation
        # updates is an ordered dictionary of theano variables to update
        updates = self.neurons.update(J)
        
        for a in self.accumulators.values(): 
            # also update the filtered decoded and encoded
            # internal theano variables for the accumulators
            if hasattr(a, 'new_decoded_input'):
                # if there's a decoded input in this accumulator,
                # add accumulated decoded inputs to theano variable updates
                updates[a.decoded_input] = a.new_decoded_input.astype('float32')
            if hasattr(a, 'new_encoded_input'):
                # if there's an encoded input in this accumulator,
                # add accumulated encoded inputs to theano variable updates
                updates[a.encoded_input] = a.new_encoded_input.astype('float32')
            if hasattr(a, 'new_learn_input'):
                # if there's a learn input in this accumulator,
                # add accumulated learn inputs to theano variable updates
                updates[a.learn_input] = a.new_learn_input.astype('float32')

        for l in self.learned_terminations:
            # also update the weight matrices on learned terminations
            updates.update(l.update())

        # and compute the decoded origin decoded_input from the neuron output
        for o in self.origin.values():
            # in the dictionary updates, set each origin's
            # output decoded_input equal to the
            # self.neuron.output() we just calculated
            updates.update(o.update(updates[self.neurons.output]))

        return updates
