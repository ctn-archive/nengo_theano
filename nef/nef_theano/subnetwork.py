class SubNetwork(object):
    def __init__(self, name, network):
        self.name = name
        self.network = network
    
    def make(self, name, *args, **kwargs):
        return self.network.make('%s.%s'%(self.name, name), *args, **kwargs)
    
    def make_array(self, name, *args, **kwargs):
        return self.network.make_array('%s.%s'%(self.name, name), *args, **kwargs)
        
    def make_input(self, name, *args, **kwargs):
        return self.network.make_input('%s.%s'%(self.name, name), *args, **kwargs)

    def make_subnetwork(self, name, *args, **kwargs):
        return self.network.make_subnetwork('%s.%s'%(self.name, name), *args, **kwargs)

    def connect(self, pre, post, *args, **kwargs):
        return self.network.connect('%s.%s'%(self.name, pre), 
                                     '%s.%s'%(self.name, post), *args, **kwargs)

    def learn(self, pre, post, error, *args, **kwargs):
        return self.network.learn('%s.%s'%(self.name, pre), 
                                   '%s.%s'%(self.name, post),
                                   '%s.%s'%(self.name, error), *args, **kwargs)

