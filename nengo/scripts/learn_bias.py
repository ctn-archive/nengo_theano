from .. import nef_theano as nef

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
    learn_bias.connect('input', 'pos_inhib', pstc=1e-6)
    learn_bias.connect('input', 'output', weight=-1)
    learn_bias.connect('pos_inhib', 'output', transform=inhib_matrix)
