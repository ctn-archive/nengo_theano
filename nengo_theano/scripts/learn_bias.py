import nengo_theano as nef

def make(net, name, neurons, min_value=-.1):
    """Simple function, if input is negative, output a small 
    positive value, to be added to the learning signal to keep the
    signal positive.
   
    :param Network net: the network to add it to 
    :param string name: the name of the abs val subnetwork
    :param int neurons: the number of neurons per ensemble
    """
    learn_bias = net.make_subnetwork(name)

    learn_bias.make('input', neurons=1, dimensions=1, mode='direct')
    learn_bias.make('pos_inhib', neurons=neurons, dimensions=1, 
        encoders=[[1]], intercept=(min_value,1))
    learn_bias.make('output', neurons=neurons, dimensions=1)

    inhib_matrix = [[-20]] * neurons

    # connect with small pstc from relay node
    learn_bias.connect('input', 
                       'pos_inhib', pstc=1e-6)
    learn_bias.connect('input', 
                       'output', weight=-1)
    learn_bias.connect_neurons(
        'pos_inhib', 
        'output', weight_matrix=inhib_matrix)

def test_learn_bias():
    import time
    import math

    start_time = time.time()
    net = nef.Network('LB test')

    net.make_input('input', values=math.sin)

    make(net, 'learn_bias', neurons=100)

    net.connect('input', 
                'learn_bias.input')

    in_probe = net.make_probe('input')
    output_probe = net.make_probe('learn_bias.output')

    build_time = time.time()
    print "build time: ", build_time - start_time

    net.run(10)

    print "sim time: ", time.time() - build_time

    import matplotlib.pyplot as plt
    plt.subplot(211); plt.title('input')
    plt.plot(in_probe.get_data())
    plt.subplot(212); plt.title('output')
    plt.plot(output_probe.get_data())
    plt.tight_layout()
    plt.show()

#test_learn_bias()
