import ensemble
import input
import simplenode
import probe
import origin

from theano import tensor as TT
import theano
import numpy
import random
import collections

class Network:
    def __init__(self, name, seed=None):
        """Wraps a Nengo network with a set of helper functions for simplifying the creation of Nengo models.

        :param string name: create and wrap a new Network with the given *name*.  
        :param int seed:    random number seed to use for creating ensembles.  This one seed is used only to
                            start the random generation process, so each neural group created will be different.        
        """
        self.name = name
        self.dt = 0.001
        self.run_time = 0.0    
        self.seed = seed        
        self.nodes = {} # all the nodes in the network, indexed by name
        self.theano_tick = None # the function call to run the theano portions of the model one timestep
        self.tick_nodes = [] # the list of nodes who have non-theano code that must be run each timestep
        self.random = random.Random()
        if seed is not None:
            self.random.seed(seed)
          
    def add(self, node):
        """Add an arbitrary non-theano node to the network.

        Used for inputs, SimpleNodes, and Probes. These nodes will be added to
        the Theano graph if the node has an "update()" function, but will also
        be triggered explicitly at every tick via the node's "theano_tick()"
        function.
        
        :param Node node: the node to add to this network
        """
        self.theano_tick = None # remake theano_tick function, in case the node has Theano updates 
        self.tick_nodes.append(node)
        self.nodes[node.name] = node

    def compute_transform(self, dim_pre, dim_post, weight=1, index_pre=None, index_post=None):
        """Helper function used by :func:`nef.Network.connect()` to create the 
        *dim_post* by *dim_pre* transform matrix. Values are either 0 or *weight*.  
        *index_pre* and *index_post* are used to determine which values are 
        non-zero, and indicate which dimensions of the pre-synaptic ensemble 
        should be routed to which dimensions of the post-synaptic ensemble.

        :param integer dim_pre: first dimension of transform matrix
        :param integer dim_post: second dimension of transform matrix
        :param float weight: the non-zero value to put into the matrix
        :param index_pre: the indexes of the pre-synaptic dimensions to use
        :type index_pre: list of integers or a single integer
        :param index_post: the indexes of the post-synaptic dimensions to use
        :type index_post: list of integers or a single integer
        :returns: a two-dimensional transform matrix performing the requested routing        
        """
        transform = [[0] * dim_pre for i in range(dim_post)] # create a matrix of zeros

        # default index_pre/post lists set up *weight* value on diagonal of transform
        # if dim_post != dim_pre, then values wrap around when edge hit
        if index_pre is None: index_pre = range(dim_pre) 
        elif isinstance(index_pre, int): index_pre = [index_pre] 
        if index_post is None: index_post = range(dim_post) 
        elif isinstance(index_post, int): index_post = [index_post]

        for i in range(max(len(index_pre), len(index_post))):
            pre = index_pre[i % len(index_pre)]
            post = index_post[i % len(index_post)]
            transform[post][pre] = weight
        return transform
        
    def connect(self, pre, post, pstc=0.01, transform=None, weight=1, index_pre=None, index_post=None, func=None):
        """Connect two nodes in the network.
        Note: cannot specify (transform) AND any of (weight, index_pre, index_post) 

        *pre* and *post* can be strings giving the names of the nodes, or they
        can be the nodes themselves (FunctionInputs and NEFEnsembles are
        supported). They can also be actual Origins or Terminations, or any
        combination of the above. 

        If transform is not None, it is used as the transformation matrix for
        the new termination. You can also use *weight*, *index_pre*, and *index_post*
        to define a transformation matrix instead.  *weight* gives the value,
        and *index_pre* and *index_post* identify which dimensions to connect.
        transform can be of several sizes: 
        post.dimensions x pre.dimensions - specify where decoded signal dimensions project
        post.neurons x pre.dimensions - overwrites post encoders, i.e. inhibitory connections
        post.neurons x pre.neurons - fully specify the connection weight matrix 

        If *func* is not None, a new Origin will be created on the pre-synaptic
        ensemble that will compute the provided function. The name of this origin 
        will taken from the name of the function, or *origin_name*, if provided.  If an
        origin with that name already exists, the existing origin will be used 
        rather than creating a new one.

        :param string pre: Name of the node to connect from.
        :param string post: Name of the node to connect to.
        :param float pstc: post-synaptic time constant for the neurotransmitter/receptor on this connection
        :param transform: The linear transfom matrix to apply across the connection.
                          If *transform* is T and *pre* represents ``x``, then the connection
                          will cause *post* to represent ``Tx``.  Should be an N by M array,
                          where N is the dimensionality of *post* and M is the dimensionality of *pre*.
        :type transform: array of floats                              
        :param index_pre: The indexes of the pre-synaptic dimensions to use.
                          Ignored if *transform* is not None. See :func:`nef.Network.compute_transform()`
        :param float weight: scaling factor for a transformation defined with *index_pre* and *index_post*.
                             Ignored if *transform* is not None. See :func:`nef.Network.compute_transform()`
        :type index_pre: List of integers or a single integer
        :param index_post: The indexes of the post-synaptic dimensions to use.
                           Ignored if *transform* is not None. See :func:`nef.Network.compute_transform()`
        :type index_post: List of integers or a single integer 
        :param function func: function to be computed by this connection.  If None, computes ``f(x)=x``.
                              The function takes a single parameter x which is the current value of
                              the *pre* ensemble, and must return wither a float or an array of floats.
        :param string origin_name: Name of the origin to check for / create to compute the given function.
                                   Ignored if func is None.  If an origin with this name already
                                   exists, the existing origin is used instead of creating a new one.
        """
        pre = self.get_object(pre) # get pre Node object from node dictionary
        post = self.get_object(post) # get post Node object from node dictionary
      
        # reset timer in case the model has been run previously, as adding a new node means we have to rebuild the theano function 
        self.theano_tick = None  
    
        # check to see if there is a transform
        # if there is, and it's [0] dimension is post.neurons_num or post.neurons_num * post.array_size then assume encoded connection
        # otherwise it's a decoded connection
     
        if transform is not None: 
            # make sure contradicting things aren't simultaneously specified
            assert (weight == 1) and (index_pre is None) and (index_post is None)

            transform = numpy.array(transform)
            # check to see if it's an encoded connection
            if transform.shape[0] != post.dimensions:
                #TODO: optimization: move this transform hstack instead to a theano TT function on encoded_output instead so the 
                # dot product for encoded output is quicker when network array w same projection to each ensemble
                # just have to make sure that encoded output ends up being (post.neurons_num x post.array_size)
                #if transform.shape[0] == post.neurons_num: 
                #    transform = numpy.vstack([transform]*post.array_size)
                #assert transform.shape[0] == post.neurons_num * post.array_size
                assert func == None # can't specify a function with an encoded connection
                # also can't get encoded output from Input or SimpleNode objects
                assert not (isinstance(pre, input.Input) or isinstance(pre, simplenode.SimpleNode)) 

                # get the instantaneous spike raster from the pre population
                neuron_output = pre.neurons.output 
                # the encoded input to the next population is the spikes x weight matrix
                # dot product is opposite order than for decoded_output because of neurons.output shape
                encoded_output = TT.dot(neuron_output, transform)
                
                # pass in the pre population encoded output function to the post population, connecting them for theano
                post.add_filtered_input(pstc=pstc, encoded_input=encoded_output)
                return
        
        # if we're doing a decoded connection 
        if not isinstance(pre, origin.Origin): # see if pre is the origin we want to connect to or not
            # if pre is not an origin, find the origin the projection originates from
            origin_name = 'X' # take default identity decoded output from pre population
            if func is not None: # if we're supposed to compute a function on this connection create an origin to do it
                origin_name = func.__name__ # set name as the function being calculated
                #TODO: better analysis to see if we need to build a new origin (rather than just relying on the name)
                if origin_name not in pre.origin: # if an origin for this function hasn't already been created
                    pre.add_origin(origin_name, func) # create origin with to perform desired func
            pre = pre.origin[origin_name]
        else: # if pre is an origin, make sure a function wasn't given 
            assert func == None # can't specify a function for an already created origin

        decoded_output = pre.decoded_output
        dim_pre = pre.dimensions

        if transform is None: # compute transform matrix if not given
            transform = self.compute_transform(dim_pre=dim_pre, dim_post=post.dimensions * post.array_size, 
                                weight=weight, index_pre=index_pre, index_post=index_post)

        # apply transform matrix, directing pre dimensions to specific post dimensions
        decoded_output = TT.dot(transform, decoded_output)

        # pass in the pre population decoded output function to the post population, connecting them for theano
        post.add_filtered_input(pstc=pstc, decoded_input=decoded_output) 

    def get_object(self, name):
        """This is a method for parsing input to return the proper object
        The only thing we need to check for here is a :, indicating an origin.

        :param string name: the name of the desired object
        """
        assert isinstance(name, str)
        split = name.split(':') # separate into node and origin, if specified

        if len(split) == 1: # no origin specified
            return self.nodes[name]

        elif len(split) == 2: # origin specified
            node = self.nodes[split[0]]
            return node.origin[split[1]]
       
    def learn(self, pre, post, error, pstc=0.01, weight_matrix=None):
        """Add a connection with learning between pre and post, modulated by error

        :param Ensemble pre: the pre-synaptic population
        :param Ensemble post: the post-synaptic population
        :param Ensemble error: the population that provides the error signal
        :param list weight_matrix: the initial connection weights with which to start
        """
        pre = self.get_object(pre)
        post = self.get_object(post)
        error = self.get_object(error)
        return post.add_learned_termination(pre, error, pstc, weight_matrix)

    def make(self, name, *args, **kwargs): 
        """Create and return an ensemble of neurons. Note that all ensembles are actually arrays of length 1        
        :returns: the newly created ensemble      

        :param string name: name of the ensemble (must be unique)
        :param int seed: random number seed to use.  Will be passed to both random.seed() and ca.nengo.math.PDFTools.setSeed().
                         If this is None and the Network was constructed with a seed parameter, a seed will be randomly generated.
        """
        if 'seed' not in kwargs.keys(): # if no seed provided, get one randomly from the rng
            kwargs['seed'] = self.random.randrange(0x7fffffff)
    
        self.theano_tick=None  # just in case the model has been run previously, as adding a new node means we have to rebuild the theano function
        e = ensemble.Ensemble(*args, **kwargs) 

        self.nodes[name] = e # store created ensemble in node dictionary
        return e

    def make_array(self, name, neurons, array_size, dimensions=1, **kwargs): 
        """Generate a network array specifically, for legacy code \ non-theano API compatibility
        """
        return self.make(name=name, neurons=neurons, dimensions=dimensions, array_size=array_size, **kwargs)
    
    def make_input(self, *args, **kwargs): 
        """ # Create an input and add it to the network
        """
        i = input.Input(*args, **kwargs)
        self.add(i)
        return i

    def make_probe(self, target, name=None, dt_sample=0.01, **kwargs):
        """Add a probe to measure the given target.
        
        :param target: a Theano shared variable to record
        :param name: the name of the probe
        :param dt_sample: the sampling frequency of the probe
        :returns The Probe object
        """
        i = 0
        while name is None or self.nodes.has_key(name):
            i += 1
            name = ("Probe%d" % i)

        p = probe.Probe(name, self, target, dt_sample, **kwargs)
        self.add(p)
        return p
            
    def make_theano_tick(self):
        """Generate the theano function for running the network simulation
        :returns theano function 
        """
        updates = collections.OrderedDict() # dictionary for all variables and the theano description of how to compute them 

        for node in self.nodes.values(): # for every node in the network
            if hasattr(node, 'update'): # if there is some variable to update 
                updates.update(node.update()) # add it to the list of variables to update every time step

        theano.config.compute_test_value = 'warn' # for debugging
        return theano.function([], [], updates=updates) # create graph and return optimized update function

    def run(self, time):
        """Run the simulation. If called twice, the simulation will continue for *time* more seconds.  
        Note that the ensembles are simulated at the dt timestep specified when they are created.
        
        :param float time: the amount of time (in seconds) to run
        """         
        # if theano graph hasn't been calculated yet, retrieve it
        if self.theano_tick is None: self.theano_tick = self.make_theano_tick() 

        for i in range(int(time / self.dt)):
            t = self.run_time + i * self.dt # get current time step
            # run the non-theano nodes
            for node in self.tick_nodes:    
                node.t = t
                node.theano_tick()
            # run the theano nodes
            self.theano_tick()    
           
        self.run_time += time # update run_time variable
