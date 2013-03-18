
from theano.tensor.shared_randomstreams import RandomStreams
from theano import tensor as TT
import theano
import numpy
import numpy.random

import neuron
import ensemble_origin
from learned_termination import hPESTermination

#TODO: straighten out this business for generating network arrays! looks like only one set of encoders is generated...????

def make_encoders(neurons, dimensions, srng, encoders=None):
    """Generates a set of encoders

    :param int neurons: number of neurons 
    :param int dimensions: number of dimensions
    :param theano.tensor.shared_randomstreams snrg: theano random number generator function
    :param list encoders: set of possible preferred directions of neurons
    """
    if encoders is None: # if no encoders specified
        encoders = srng.normal((neurons, dimensions)) # generate randomly
    else:    
        encoders = numpy.array(encoders) # if encoders were specified, cast list as array
        # repeat array until 'encoders' is the same length as number of neurons in population
        encoders = numpy.tile(encoders, (neurons / len(encoders) + 1, 1))[:neurons, :dimensions]
       
    # normalize encoders across represented dimensions 
    norm = TT.sum(encoders * encoders, axis=[1], keepdims=True)
    encoders = encoders / TT.sqrt(norm)        

    return theano.function([], encoders)()

   
class Accumulator:
    def __init__(self, ensemble, pstc):
        """A collection of terminations in the same population, all sharing the same time constant        
        Stores the decoded_input accumulated across these terminations, i.e. their summed contribution to the represented signal
        Also stores the direct_input value, which is direct current input when connections are added with a weight matrix specified

        :param Ensemble ensemble: the ensemble this set of terminations is attached to
        :param float pstc: post-synaptic time constant on filter
        """
        self.ensemble = ensemble
    
        self.decay = numpy.exp(-self.ensemble.neurons.dt / pstc) # time constant for filter
        self.decoded_total = None # the theano object representing the sum of the decoded inputs to this filter
        self.encoded_total = None # the theano object representing the sum of the encoded inputs to this filter
        self.learn_total = None # the theano object representing the sum of the learned inputs to this filter

        # decoded_input should be dimensions * array_size because we account for the transform matrix here, so different array networks get different input
        self.decoded_input = theano.shared(numpy.zeros(self.ensemble.dimensions * self.ensemble.array_size).astype('float32')) # the initial filtered decoded input 
        # encoded_input, however, is the same for all networks in the arrays, connecting directly to the neurons, so only needs to be size neurons_num
        self.encoded_input = theano.shared(numpy.zeros(self.ensemble.neurons_num).astype('float32')) # the initial filtered encoded input 
        # learn_input, is different for all networks in the arrays, connecting directly to the neurons, so it needs to be size neurons_num * array_size
        self.learn_input = theano.shared(numpy.zeros(self.ensemble.neurons_num * self.ensemble.array_size).astype('float32')) # the initial filtered encoded input 

    def add_decoded_input(self, decoded_input):
        """Add to the current set of decoded inputs (with the same post-synaptic time constant pstc) an additional input
        self.new_decoded_input is the calculation of the contribution of all of the decoded input with the same filtering 
        time constant to the ensemble, input current then calculated as the sum of all decoded_input x ensemble.encoders

        :param decoded_input: theano object representing the output of the pre population multiplied by this termination's transform matrix
        """
        if self.decoded_total is None: self.decoded_total = decoded_input # initialize internal value storing decoded input value to neurons
        else: self.decoded_total = self.decoded_total + decoded_input # add to the decoded input to neurons 

        self.new_decoded_input = self.decay * self.decoded_input + (1 - self.decay) * self.decoded_total # the theano object representing the filtering operation        

    def add_encoded_input(self, encoded_input): 
        """Add to the current set of encoded input (with the same post-synaptic time constant pstc) an additional input
        self.new_encoded_input is the calculation of the contribution of all the encoded input with the same filtering 
        time constant to the ensemble, where the encoded_input is exactly the input current to each neuron in the ensemble
        
        :param encoded_input: theano object representing the decoded output of every neuron of the pre population x a connection weight matrix
        """
        if self.encoded_total is None: self.encoded_total = encoded_input # initialize internal value storing encoded input (current) to neurons 
        else: self.encoded_total = self.encoded_total + encoded_input # add input encoded input (current) to neurons

        # flatten because a col + a vec gives a matrix type, but it's actually just a vector still
        self.new_encoded_input = TT.flatten(self.decay * self.encoded_input + (1 - self.decay) * self.encoded_total) # the theano object representing the filtering operation        
        
    def add_learn_input(self, learn_input): 
        """Add to the current set of learn input (with the same post-synaptic time constant pstc) an additional input
        self.new_learn_input is the calculation of the contribution of all the learn input with the same filtering 
        time constant to the ensemble, where the learn_input is exactly the input current to each neuron in the ensemble
        
        :param learn_input: theano object representing the current output of every neuron of the pre population x a connection weight matrix
        """
        if self.learn_total is None: self.learn_total = learn_input # initialize internal value storing learned encoded input (current) to neurons 
        else: self.learn_total = self.learn_total + learn_input # add input learn input (current) to neurons

        # flatten because a col + a vec gives a matrix type, but it's actually just a vector still
        self.new_learn_input = TT.flatten(self.decay * self.learn_input + (1 - self.decay) * self.learn_total) # the theano object representing the filtering operation        

class Ensemble:
    def __init__(self, neurons, dimensions, tau_ref=0.002, tau_rc=0.02, 
                 max_rate=(200,300), intercept=(-1.0,1.0), radius=1.0, 
                 encoders=None, seed=None, neuron_type='lif', dt=0.001, 
                 array_size=1, eval_points=None, noise=None, noise_type='uniform'):
        """Create an population of neurons with NEF parameters on top
        
        :param int neurons: number of neurons in this population
        :param int dimensions: number of dimensions in signal these neurons represent 
        :param float tau_ref: refractory period of neurons in this population
        :param float tau_rc: RC constant 
        :param tuple max_rate: lower and upper bounds on randomly generate firing rates for neurons in this population
        :param tuple intercept: lower and upper bounds on randomly generated x offset
        :param float radius: the range of input values (-radius:radius) this population is sensitive to 
        :param list encoders: set of possible preferred directions of neurons
        :param int seed: seed value for random number generator
        :param string neuron_type: type of neuron model to use, options = {'lif'}
        :param float dt: time step of neurons during update step
        :param int array_size: number of sub-populations - for network arrays
        :param list eval_points: specific set of points to optimize decoders over by default for this ensemble
        :param float noise: noise parameter for this ensemble for noise added to input current, sampled at every timestep
                            if noise_type = uniform, this is the lower and upper bound on the distribution
                            if noise_type = gaussian, this is the variance
        :param string noise_type: the type of noise added to the input current, options = {'uniform', 'gaussian'}
                                  default is 'uniform' to match the Nengo implementation
        """
        self.seed = seed
        self.neurons_num = neurons
        self.dimensions = dimensions
        self.array_size = array_size
        self.radius = radius
        self.eval_points = eval_points
        self.noise = noise
        self.dt = dt
        self.noise_type = noise_type
        if self.noise: # if a noise variance was specified
            self.srng = RandomStreams(seed=self.seed) # setup theano random number generator to generate noise
        
        # create the neurons
        # TODO: handle different neuron types, which may have different parameters to pass in
        self.neurons = neuron.names[neuron_type]((array_size, self.neurons_num), 
                                                 tau_rc=tau_rc, tau_ref=tau_ref, dt=dt)
        
        # compute alpha and bias
        srng = RandomStreams(seed=seed) # set up theano random number generator
        max_rates = srng.uniform([self.neurons_num], low=max_rate[0], high=max_rate[1])  
        threshold = srng.uniform([self.neurons_num], low=intercept[0], high=intercept[1])
        alpha, self.bias = theano.function([], self.neurons.make_alpha_bias(max_rates, threshold))()
        self.bias = self.bias.astype('float32') # force to 32 bit for consistency / speed
                
        # compute encoders
        self.encoders = make_encoders(self.neurons_num, dimensions, srng, encoders=encoders)
        self.encoders = (self.encoders.T * alpha).T # combine encoders and gain for simplification
        
        self.origin = {} # make origin dictionary
        self.add_origin('X', func=None, eval_points=self.eval_points) # make default origin
        
        self.accumulators = {} # dictionary of accumulators tracking terminations with different pstc values
        self.learned_terminations = [] # list of learned terminations on ensemble
    
    def add_filtered_input(self, pstc, decoded_input=None, encoded_input=None, learn_input=None):
        """Accounts for a new termination that takes the given input 
        (a theano object) and filters it with the given pstc.

        Adds its contributions to the set of decoded, encoded, or learn input with the 
        same pstc. Decoded inputs are represented signals, encoded inputs are
        decoded_output * weight matrix, learn input is activities * weight_matrix.
        Can only have decoded OR encoded OR learn input != None.

        :param float pstc: post-synaptic time constant
        :param decoded_input: theano object representing the decoded output of 
            the pre population multiplied by this termination's transform matrix
        :param encoded_input: theano object representing the encoded output of 
            the pre population multiplied by a connection weight matrix
        :param learn_input: theano object representing the learned output of 
            the pre population multiplied by a connection weight matrix
        """
        # make sure one and only one of (decoded_input, encoded_input, learn_input) is specified
        if decoded_input: assert (encoded_input is None) and (learn_input is None)
        elif encoded_input: assert (decoded_input is None) and (learn_input is None)
        elif learn_input: assert (decoded_input is None) and (encoded_input is None)
        assert (decoded_input) or (encoded_input) or (learn_input) 

        if pstc not in self.accumulators: # make sure there's an accumulator for given pstc
            self.accumulators[pstc] = Accumulator(self, pstc)

        # add this termination's contribution to the set of terminations with the same pstc
        if decoded_input: 
            # rescale decoded_input by this neurons radius to put us in the right range
            self.accumulators[pstc].add_decoded_input(TT.true_div(decoded_input, self.radius)) 
        elif encoded_input: 
            self.accumulators[pstc].add_encoded_input(encoded_input)
        elif learn_input:
            self.accumulators[pstc].add_learn_input(learn_input)
   
    #TODO: make this support specifying error origins 
    def add_learned_termination(self, pre, error, pstc, weight_matrix=None,
                                learned_termination_class=hPESTermination):
        """Adds a learned termination to the ensemble.

        Accounting for the additional input_current is still done through the 
        accumulator, but a learned_termination object is also created and
        attached to keep track of the pre and post (self) spike times, and 
        adjust the weight matrix according to the specified learning rule.
    
        :param Ensemble pre: the pre-synaptic population
        :param Ensemble error: the population that provides the error signal
        :param list weight_matrix: the initial connection weights with which to start
        """
        # generate an initial weight matrix if none provided, random numbers between -.001 and .001
        if weight_matrix is None: 
            weight_matrix = numpy.random.uniform(
                size=(self.neurons_num * self.array_size, pre.neurons_num * pre.array_size), 
                low=-.001, high=.001)
        else:
            weight_matrix = numpy.array(weight_matrix) # make sure it's an np.array

        learned_term = learned_termination_class(pre, self, error, weight_matrix)
        learn_output = TT.dot(pre.neurons.output, learned_term.weight_matrix.T)

        # add learn output to the accumulator to handle the input_current from this connection during simulation
        self.add_filtered_input(pstc=pstc, learn_input=learn_output)
        self.learned_terminations.append(learned_term)
        return learned_term
        
    def add_origin(self, name, func, eval_points=None):
        """Create a new origin to perform a given function over the represented signal
        
        :param string name: name of origin
        :param function func: desired transformation to perform over represented signal
        :param list eval_points: specific set of points to optimize decoders over for this origin
        """
        if eval_points == None: eval_points = self.eval_points
        self.origin[name] = ensemble_origin.EnsembleOrigin(self, func, eval_points=eval_points)    

    def update(self):
        """Compute the set of theano updates needed for this ensemble
        Returns dictionary with new neuron state, termination, and origin values
        """
        
        # find the total input current to this population of neurons
        input_current = numpy.tile(self.bias, (self.array_size, 1)) # apply respective biases to neurons in the population 
        X = numpy.zeros(self.dimensions * self.array_size) # set up matrix to store accumulated decoded input, same size as decoded_input
    
        for a in self.accumulators.values(): 
            if hasattr(a, 'new_decoded_input'): # if there's a decoded input in this accumulator,
                X += a.new_decoded_input # add its values to the total decoded input
            if hasattr(a, 'new_encoded_input'): # if there's an encoded input in this accumulator
                # encoded input is the same to every array network
                input_current += a.new_encoded_input # add its values directly to the input current 
            if hasattr(a, 'new_learn_input'): # if there's a learn input in this accumulator
                # learn input is self.neurons_num x self.array_size, need to reshape
                input_current += a.new_learn_input.reshape((self.array_size, self.neurons_num))

        #TODO: optimize for when nothing is added to X (ie there are no decoded inputs)
        X = X.reshape((self.array_size, self.dimensions)) # reshape decoded input for network arrays
        # find input current caused by decoded input signals 
        input_current += TT.dot(X, self.encoders.T) # calculate input_current for each neuron as represented input signal x preferred direction

        # if noise has been specified for this neuron, add Gaussian white noise with variance self.noise to the input_current
        if self.noise: 
            # generate random noise values, one for each input_current element, 
            # with standard deviation = sqrt(self.noise=std**2)
            # When simulating white noise, the noise process must be scaled by
            # sqrt(dt) instead of dt. Hence, we divide the std by sqrt(dt).
            if self.noise_type.lower() == 'gaussian':
                input_current += self.srng.normal(size=input_current.shape, 
                                                  std=numpy.sqrt(self.noise/self.dt))
            elif self.noise_type.lower() == 'uniform':
                input_current += self.srng.uniform(size=input_current.shape, 
                                                   low=-self.noise/numpy.sqrt(self.dt), 
                                                   high=self.noise/numpy.sqrt(self.dt))
        
        # pass that total into the neuron model to produce the main theano computation
        updates = self.neurons.update(input_current) # updates is an ordered dictionary of theano internal variables to update

        for a in self.accumulators.values(): 
            # also update the filtered decoded and encoded internal theano variables for the accumulators
            if hasattr(a, 'new_decoded_input'): # if there's a decoded input in this accumulator,
                updates[a.decoded_input] = a.new_decoded_input.astype('float32') # add accumulated decoded inputs to theano internal variable updates
            if hasattr(a, 'new_encoded_input'): # if there's an encoded input in this accumulator,
                updates[a.encoded_input] = a.new_encoded_input.astype('float32') # add accumulated encoded inputs to theano internal variable updates
            if hasattr(a, 'new_learn_input'): # if there's a learn input in this accumulator,
                updates[a.learn_input] = a.new_learn_input.astype('float32') # add accumulated learn inputs to theano internal variable updates

        for l in self.learned_terminations:
            # also update the weight matrices on learned terminations
            updates.update(l.update())

        # and compute the decoded origin decoded_input from the neuron output
        for o in self.origin.values():
            # in the dictionary updates, set each origin's output decoded_input equal to the self.neuron.output() we just calculated
            updates.update(o.update(updates[self.neurons.output]))
        
        return updates    
