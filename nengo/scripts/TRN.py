from .. import nef_theano as nef
import random

import abs_val

def make_TRN(net, name, neurons, dimensions, dim_fb_err, radius=1.5, 
             tau_inhib=0.005, inhib_scale1=10, inhib_scale2=1):
    """A function that makes a subnetwork to calculate the weighted 
    value of a specified cortical action.
   
    :param Network net: the network to add it to 
    :param string name: the name of the abs val subnetwork
    :param int neurons: the number of neurons per ensemble
    :param int dimensions: the dimensionality of the input
    :param int dim_fb_err: the dimensionality of the feedback error signal
    :param float radius: the radius of the population calculating 
        the derivative
    :param float tau_inhib: the time constant on the inhibitory connection
    :param float inhib_scale1: the strength of the inhibitory connection
        preventing output from projecting
    :param float inhib_scale2: the strength of the inhibitory connection
        inhibiting the population inhibiting output
    """
    # to shift thalamic range [-1 to 1]
    def subone(x): return [i - 1 for i in x] 

    # for our error based inhibition signal, if there's 
    # any error signal at all, output 1
    def step(x):
        if sum(x) > .05: return 1
        else: return 0

    TRN = net.make_subnetwork(name)

    # create input relay for 
    TRN.make('input_cortex', neurons=1, dimensions=dimensions, mode='direct') 
    # create input relay for input signal from thalamus
    ### NOTE: important to connect to this relay with weight=2
    TRN.make('input_thalamus', neurons=1, dimensions=dimensions, mode='direct')
    # create input relay for feedback error signal
    TRN.make('input_error', neurons=1, dimensions=dim_fb_err, mode='direct')
    # create output relay
    TRN.make('output', neurons=1, dimensions=dimensions, mode='direct')
     
    # create a subnetwork to calculate the absolute value of error input
    abs_val.make_abs_val(net=TRN, name='abs_error', neurons=neurons, 
        dimensions=dim_fb_err, intercept=(.05,1))
    
     # create output relays
    for d in range(dimensions):
        TRN.make('output %d'%d, neurons=neurons, dimensions=1)

    # create error populations
    TRN.make('err_step', neurons=1, dimensions=1, mode='direct')
    TRN.make('err_sum', neurons=1, dimensions=1, mode='direct')

    # track the derivative of sum, and only let output relay the 
    # input if the derivative is below a given threshold

    # create a subnetwork to calculate the absolute value 
    abs_val.make_abs_val(net=TRN, name='abs_val_deriv', 
        neurons=neurons, dimensions=dimensions, intercept=(.1,1))

    # find derivatives and send to absolute value network

    # create population to calculate the derivative
    TRN.make_array('derivative', neurons=neurons, 
        array_size=dimensions, dimensions=2, radius=radius) 
    # set up recurrent connection
    TRN.connect('derivative', 'derivative', 
        index_pre=range(0,2*dimensions,2), 
        index_post=range(1,2*dimensions,2), pstc=0.05)
    # connect deriv input 
    TRN.connect('input_cortex', 'derivative', 
        index_post=range(0,2*dimensions,2), pstc=1e-6)
    # connect deriv output
    TRN.connect('derivative', 'abs_val_deriv.input',
        index_pre=range(0,2*dimensions,2), weight=-1)
    TRN.connect('derivative', 'abs_val_deriv.input',
        index_pre=range(1,2*dimensions,2))
    
    # create integrator that saturates quickly in response to any derivative 
    # signal and inhibits output, decreases in response to a second input 
    # the idea being that you connect system feedback error up to the second 
    # input so that BG contribution is inhibited unless there's FB error
    TRN.make('integrator', neurons=neurons, dimensions=1, intercept=(.1,1))
    # hook up to hold current value
    TRN.connect('integrator', 'integrator', weight=1.1) 

    # connect it up, pstc low so relays don't cause delay
    for d in range(dimensions):
        # set up communication channel
        TRN.connect('input_cortex', 'output %d'%d, pstc=1e-6, index_pre=d) 
        # set up thalamic input
        TRN.connect('input_thalamus', 'output %d'%d, pstc=1e-6, 
            index_pre=d)#, func=subone) 
        TRN.connect('output %d'%d, 'output', index_post=d)
    # saturate integrator if there is a strong derivative
    TRN.connect('abs_val_deriv.output', 'integrator', weight=10) 
        
    # set up inhibitory matrices
    inhib_matrix1 = [[-inhib_scale1]] * neurons 
    inhib_matrix2 = [[-inhib_scale2]] * neurons

    for d in range(dimensions):
        TRN.connect('integrator', 'output %d'%d, 
            transform=inhib_matrix1, pstc=tau_inhib)

    TRN.connect('input_error', 'abs_error.input')
    # sum error signals
    TRN.connect('abs_error.output', 'err_step') 
    # calculate step function
    TRN.connect('err_step', 'err_sum', func=step)
    # connect up error relay to integrator with high pstc 
    # value to get high pass filter behavior
    TRN.connect('err_sum', 'integrator', transform=inhib_matrix2, pstc=1)

def test_TRN():
    import time

    start_time = time.time()
    net = nef.Network('TRN test')
 
    def cx_func(x):
        return [math.sin(x), -math.sin(x), math.cos(x)]

    net.make_input('input_cortex', value=[.2, .5, -.3])
    net.make_input('input_thalamus', value=[.5,.6,1])
    net.make_input('input_error', value=[0, 0])

    make_TRN(net, 'TRN', neurons=100, dimensions=3, 
        dim_fb_err=2) #, mode='direct')

    net.connect('input_cortex', 'TRN.input_cortex')
    net.connect('input_thalamus', 'TRN.input_thalamus')#, weight=2)
    net.connect('input_error', 'TRN.input_error')

    output_probe = net.make_probe('TRN.output')
    avout_probe = net.make_probe('TRN.abs_val_deriv.output')
    int_probe = net.make_probe('TRN.integrator')

    build_time = time.time()
    print "build time: ", build_time - start_time

    net.run(1)

    print "sim time: ", time.time() - build_time

    import matplotlib.pyplot as plt
    plt.plot(output_probe.get_data())
    plt.subplot(312); plt.title('TRN abs val output')
    plt.plot(avout_probe.get_data()); plt.ylim([-1,1])
    plt.subplot(313); plt.title('TRN integrator')
    plt.plot(int_probe.get_data())
    plt.tight_layout()
    plt.show()

test_TRN()
