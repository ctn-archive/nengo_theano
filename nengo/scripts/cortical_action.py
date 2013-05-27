from .. import nef_theano as nef
import random

def make_cortical_action(net, name, neurons, dimensions, action_vals=None, 
                         scale=2.0, mode='spiking'): 
    """A function that makes a subnetwork to calculate the weighted 
    value of a specified cortical action.
   
    :param Network net: the network to add it to 
    :param string name: the name of the abs val subnetwork
    :param int neurons: the number of neurons per ensemble
    :param int dimensions: the dimensionality of the input
    :param list action_vals: list of actions values for each dimension
    :param float scale: if no action_vals specified, randomly create
        action values between [-scale/2, scale/2]
    :returns: action_vals, so you can access them if they were generated
    """
    def product(x): return x[0] * x[1]

    cortical_action = net.make_subnetwork(name)

    # if no action values specified, randomly create some
    if (action_vals==None): 
        # generate values and scale to put in the right range
        action_vals = [random.random() * scale - scale/2
            for d in range(dimensions)]

    cortical_action.make_input('action', value=action_vals)

    # create input relay
    cortical_action.make('input', neurons=1, 
        dimensions=dimensions, mode='direct') 
    # create output
    cortical_action.make('output', neurons=1, 
        dimensions=dimensions, mode='direct') 
    # create multiplication population
    cortical_action.make('multiplication', neurons=neurons, 
        array_size=dimensions, dimensions=2, radius=1.5,
        encoders=[[1,1],[-1,1],[1,-1],[-1,-1]], mode=mode) 
    
    # connect input to mult
    cortical_action.connect('input', 'multiplication', 
        index_post=range(0, 2*dimensions,2), pstc=0) 
    # connect action to mult
    cortical_action.connect('action', 'multiplication', 
        index_post=range(1, 2*dimensions,2)) 
    
    # connect mult to output and perform multiplication
    cortical_action.connect('multiplication', 'output', 
        func=product) 

    return action_vals

def test_cortical_action():

    net = nef.Network('Cortical action test')

    import numpy as np
    input = np.array([-.2, .5, -.8])
    action_vals = np.array([-.4, .35, .78])

    net.make_input('input', value=input)

    make_cortical_action(net, 'ca0', neurons=200, dimensions=3, 
        action_vals=action_vals) #, mode='direct')
    ca1_a_vals = make_cortical_action(net, 'ca1', neurons=100, 
        dimensions=3, mode='direct')

    net.connect('input', 'ca0.input')
    net.connect('input', 'ca1.input')

    im_probe = net.make_probe('input')
    ca0_probe = net.make_probe('ca0.output')
    ca1_probe = net.make_probe('ca1.output')

    net.run(1)

    import matplotlib.pyplot as plt
    plt.ion(); plt.close()
    a = 'input = ', input
    plt.subplot(311); plt.title(a)
    plt.plot(im_probe.get_data())
    b = 'answer = ', input * action_vals
    plt.subplot(312); plt.title(b)
    plt.plot(ca0_probe.get_data())
    c = 'answer = ', input * ca1_a_vals 
    plt.subplot(313); plt.title(c)
    plt.plot(ca1_probe.get_data())
    plt.tight_layout()

#test_cortical_action()
