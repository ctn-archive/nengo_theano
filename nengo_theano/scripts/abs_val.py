from .. import nef_theano as nef

def make_abs_val(net, name, neurons, dimensions, intercept=[0, 1]):
    """A function that makes a subnetwork to calculate the absolute
    value of the input vector, and adds it to the network.
   
    :param Network net: the network to add it to 
    :param string name: the name of the abs val subnetwork
    :param int neurons: the number of neurons per ensemble
    :param int dimensions: the dimensionality of the input
    :param intercept: the range of represented values
    :param intercept type: int or list
        if int or list length 1, then value is the lower bound, 
        and the upper bound is set to 1
    """
    
    abs_val = net.make_subnetwork(name)

    # create input relay
    abs_val.make('input', neurons=1, dimensions=dimensions, mode='direct') 
    # create output relay
    abs_val.make('output', neurons=1, dimensions=dimensions, mode='direct') 

    num_samples = 500
    dx = (intercept[1] - intercept[0]) / num_samples
    eval_points_pos = [x*dx + intercept[0] for x in range(num_samples)]
    eval_points_neg = [-x*dx + intercept[0] for x in range(num_samples)]
 
    # for each dimension in the input signal
    for d in range(dimensions): 
        # create a population for the positive and negative parts of the signal
        abs_val.make('abs_pos%d'%d, neurons=neurons, 
            dimensions=1, encoders=[[1]], intercept=intercept,
            eval_points=eval_points_pos)
        abs_val.make('abs_neg%d'%d, neurons=neurons, 
            dimensions=1, encoders=[[-1]], intercept=intercept,
            eval_points=eval_points_neg)

        # connect to input, pstc = 0 so no delay introduced by relay populations
        abs_val.connect('input', 'abs_pos%d'%d, index_pre=d, pstc=1e-6)
        abs_val.connect('input', 'abs_neg%d'%d, index_pre=d, pstc=1e-6)
    
        # connect to output, making the negative values positive
        abs_val.connect('abs_pos%d'%d, 'output', index_post=d)
        abs_val.connect('abs_neg%d'%d, 'output', index_post=d, weight=-1)

def test_abs_val():

    net = nef.Network('Abs val test')

    import numpy as np
    input = np.array([-.2, .5, -.8])
    net.make_input('input', value=input)

    make_abs_val(net, 'abs_val', neurons=neurons, 
        dimensions=3, intercept=(.2, 1))

    net.connect('input', 'abs_val.input')

    im_probe = net.make_probe('abs_val.input')
    av0_pos_probe = net.make_probe('abs_val.abs_pos0')
    av0_neg_probe = net.make_probe('abs_val.abs_neg0')
    av1_pos_probe = net.make_probe('abs_val.abs_pos1')
    av1_neg_probe = net.make_probe('abs_val.abs_neg1')
    av2_pos_probe = net.make_probe('abs_val.abs_pos2')
    av2_neg_probe = net.make_probe('abs_val.abs_neg2')
    av_probe = net.make_probe('abs_val.output')

    net.run(1)

    import matplotlib.pyplot as plt
    a = 'input = ', input
    plt.subplot(411); plt.title(a)
    plt.plot(im_probe.get_data())
    plt.subplot(412); plt.title('abs_val_neg')
    plt.plot(av0_pos_probe.get_data())
    plt.plot(av1_pos_probe.get_data())
    plt.plot(av2_pos_probe.get_data())
    plt.legend(['input0','input1','input2'])
    plt.subplot(413); plt.title('abs_val_neg')
    plt.plot(av0_neg_probe.get_data())
    plt.plot(av1_neg_probe.get_data())
    plt.plot(av2_neg_probe.get_data())
    plt.legend(['input0','input1','input2'])
    input[np.abs(input) <= .2] = 0
    b = 'answer = ', np.abs(input)
    plt.subplot(414); plt.title(b)
    plt.plot(av_probe.get_data())
    plt.tight_layout()
    plt.show()

#test_abs_val()
