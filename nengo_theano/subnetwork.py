class SubNetwork(object):
    """A container for sectioning models into modular sections.
    """
    def __init__(self, name, network):
        """
        :param string name: the name of the subnetwork
        :param Network network: the containing network
        """
        self.name = name
        self.network = network
    
    def connect(self, pre, post, *args, **kwargs):
        """Connect two nodes inside the subnetwork
        :param pre: the pre-synaptic node
        :param post: the post-synaptic node
        """
        return self.network.connect('%s.%s'%(self.name, pre), 
                                     '%s.%s'%(self.name, post), *args, **kwargs)
    def connect_neurons(self, pre, post, *args, **kwargs):
        """Connect to post-synaptic neuorns inside the subnetwork
        :param pre: the pre-synaptic node
        :param post: the post-synaptic node
        """
        return self.network.connect_neurons('%s.%s'%(self.name, pre), 
                                     '%s.%s'%(self.name, post), *args, **kwargs)
    def learn(self, pre, post, error, *args, **kwargs):
        """Create a learning connection inside the subnetwork
        :param pre: the pre-synaptic node
        :param Ensemble post: the host of the learned_termination
        :param error: the node with the origin with the error signal
        """
        return self.network.learn('%s.%s'%(self.name, pre), 
                                   '%s.%s'%(self.name, post),
                                   '%s.%s'%(self.name, error), *args, **kwargs)

    def make(self, name, *args, **kwargs):
        """Create an ensemble inside the subnetwork
        :param string name: the name of the ensemble 
        """
        return self.network.make('%s.%s'%(self.name, name), *args, **kwargs)
    
    def make_array(self, name, *args, **kwargs):
        """Create an array inside the subnetwork
        :param string name: the name of the ensemble 
        """
        return self.network.make_array('%s.%s'%(self.name, name), *args, **kwargs)
        
    def make_input(self, name, *args, **kwargs):
        """Create an input inside the subnetwork
        :param string name: the name of the ensemble 
        """
        return self.network.make_input('%s.%s'%(self.name, name), *args, **kwargs)

    def make_subnetwork(self, name, *args, **kwargs):
        """Create a subnetwork inside the subnetwork
        :param string name: the name of the ensemble 
        """
        return self.network.make_subnetwork('%s.%s'%(self.name, name), *args, **kwargs)

    def theano_tick(self): 
        """Call the theano tick function for the network
        """
        self.network.theano_tick()
        
    def set_alias(self, alias, node):
        """Adds a named shortcut to an existing node within this network to be
        used to simplify connect() calls.
        
        :param string alias: the new short name to create
        :param string node: the existing node name
        """    
        
        self.network.set_alias('%s.%s'%(self.name, alias), '%s.%s'%(self.name, node))    
