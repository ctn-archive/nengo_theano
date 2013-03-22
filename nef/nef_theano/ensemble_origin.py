import collections

import theano
from theano import tensor as TT
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

from . import neuron
from . import cache
from .origin import Origin

class EnsembleOrigin(Origin):
    def __init__(self, ensemble, func=None, eval_points=None):
        """The output from a population of neurons (ensemble),
        performing a transformation (func) on the represented value.

        :param Ensemble ensemble:
            the Ensemble to which this origin is attached
        :param function func:
            the transformation to perform to the ensemble's
            represented values to get the output value
        
        """
        self.ensemble = ensemble
        self.decoders = self.compute_decoders(func, eval_points)
        # decoders is array_size * neurons_num * func_dimensions, 
        # initial value should have array_size values * func_dimensions
        initial_value = np.zeros(
            self.ensemble.array_size * self.decoders.shape[2]) 
        Origin.__init__(self, func=func, initial_value=initial_value)
    
    def compute_decoders(self, func, eval_points=None):     
        """Compute decoding weights.

        Calculate the scaling values to apply to the output
        of each of the neurons in the attached population
        such that the weighted summation of their output
        generates the desired decoded output.

        Decoder values computed as D = (A'A)^-1 A'X_f
        where A is the matrix of activity values of each 
        neuron over sampled X values, and X_f is the vector
        of desired f(x) values across sampled points.

        :param function func: function to compute with this origin
        :param list eval_points:
            specific set of points to optimize decoders over 
        """
        if eval_points == None:  
            # generate sample points from state space randomly
            # to minimize decoder error over in decoder calculation
            #TODO: have num_samples be more for higher dimensions?
            # 5000 maximum (like Nengo)?
            self.num_samples = 500
            eval_points = self.make_samples()

        else:
            # otherwise reset num_samples, and make sure eval_points
            # is in the right form
            # (rows are input dimensions, columns different samples)
            eval_points = np.array(eval_points)
            if len(eval_points.shape) == 1:
                eval_points.shape = [1, eval_points.shape[0]]
            self.num_samples = eval_points.shape[1]

        # compute the target_values at the sampled points 
        if func is None:
            # if no function provided, use identity function as default
            target_values = eval_points 
        else:
            # otherwise calculate target_values using provided function
            
            # scale all our sample points by ensemble radius,
            # calculate function value, then scale back to unit length

            # this ensures that we accurately capture the shape of the
            # function when the radius is > 1 (think for example func=x**2)
            target_values = (np.array([func(s * self.ensemble.radius)
                                      for s in eval_points.T])
                             / self.ensemble.radius)
            if len(target_values.shape) < 2:
                target_values.shape = target_values.shape[0], 1
            target_values = target_values.T
        
        # replicate attached population of neurons into array of ensembles,
        # one ensemble per sample point
        # set up matrix to store decoders,
        # should be (array_size * neurons_num * dim_func) 
        decoders = np.zeros((self.ensemble.array_size,
                             self.ensemble.neurons_num,
                             target_values.shape[0]))

        for index in range(self.ensemble.array_size): 
            key=self.ensemble.cache_key+'_%d'%index
            data = cache.get_gamma_inv(key)
            if data is not None:
                Ginv, A = data
            else:
                print 'recomputing',key
                # compute the input current for every neuron and every sample point
                J = np.dot(self.ensemble.encoders[index], eval_points)
                J += self.ensemble.bias[index][:, np.newaxis]

                # so in parallel we can calculate the activity
                # of all of the neurons at each sample point 
                neurons = self.ensemble.neurons.__class__(
                    size=(self.ensemble.neurons_num, self.num_samples), 
                    tau_rc=self.ensemble.neurons.tau_rc,
                    tau_ref=self.ensemble.neurons.tau_ref)

                # run the neuron model for 1 second,
                # accumulating spikes to get a spike rate
                #TODO: is this long enough? Should it be less?
                # If we do less, we may get a good noise approximation!
                A = neuron.accumulate(J, neurons)

                # compute Gamma and Upsilon
                G = np.dot(A, A.T) # correlation matrix
                
                #TODO: optimize this so we're not doing
                # the full eigenvalue decomposition
                #TODO: add NxS method for large N?
                #TODO: compare below with pinv rcond
                
                #TODO: check the decoder_noise math, and maybe add on to the
                #      diagonal of G?

                # eigh for symmetric matrices, returns
                # evalues w and normalized evectors v
                w, v = np.linalg.eigh(G)

                dnoise = self.ensemble.decoder_noise * self.ensemble.decoder_noise

                # formerly 0.1 * 0.1 * max(w), set threshold
                limit = dnoise * max(w) 
                for i in range(len(w)):
                    if w[i] < limit:
                        # if < limit set eval = 0
                        w[i] = 0
                    else:
                        # prep for upcoming Ginv calculation
                        w[i] = 1.0 / w[i]
                # w[:, np.newaxis] gives transpose of vector,
                # np.multiply is very fast element-wise multiplication
                Ginv = np.dot(v, np.multiply(w[:, np.newaxis], v.T)) 
                
                #Ginv=np.linalg.pinv(G, rcond=.01)  
                cache.set_gamma_inv(key, (Ginv, A))


            U = np.dot(A, target_values.T)
            
            # compute decoders - least squares method 
            decoders[index] = np.dot(Ginv, U) / (self.ensemble.neurons.dt)

        return decoders.astype('float32')

    def make_samples(self):
        """Generate sample points uniformly distributed within the sphere.
        
        Returns float array of sample points.
        
        """
        srng = RandomStreams(seed=self.ensemble.seed)
        samples = srng.normal((self.num_samples, self.ensemble.dimensions))
        
        # normalize magnitude of sampled points to be of unit length
        norm = TT.sum(samples * samples, axis=[1], keepdims=True) 
        samples = samples / TT.sqrt(norm)

        # generate magnitudes for vectors from uniform distribution
        scale = (srng.uniform((self.num_samples,))
                 ** (1.0 / self.ensemble.dimensions))

        # scale sample points
        samples = samples.T * scale 
        
        return theano.function([], samples)()

    def update(self, spikes):
        """the theano computation for converting neuron output
        into a decoded value.
        
        returns a dictionary with the decoded output value

        :param array spikes:
            theano object representing the instantaneous spike raster
            from the attached population

        """

        # assuming decoders are numpy array and trying to put them into
        # a theano variable instead to see what happens (#HACK)

        # XXX shared_decoders are *NOT* aliased to self.decoders in
        #     any way
        shared_decoders = theano.shared(self.decoders.astype(spikes.dtype))

        # multiply the output by the attached ensemble's radius
        # to put us back in the right range
        decoded_output = TT.concatenate(
            [TT.flatten(TT.dot(spikes[i], shared_decoders[i]))
             for i in range(self.ensemble.array_size)])
        decoded_output = TT.mul(
            decoded_output, self.ensemble.radius).astype('float32')
        return collections.OrderedDict({self.decoded_output: decoded_output})
