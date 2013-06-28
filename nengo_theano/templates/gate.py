import nengo_theano as nef

def make(net, name, gated, neurons, pstc=0.01):
    """
    """
    def addOne(x):
        return [x[0]+1]            

    net.make(name, neurons=neurons, dimensions=1, 
        intercept=(-0.7, 0), encoders=[[-1]])

    output = net.get_object(gated)

    weights = [[-10]] * output.neurons_num

    net.connect_neurons(name, gated, weight_matrix=weights, 
        pstc=pstc, func=addOne)

def test_gate(): 
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    
    import nengo_theano as nef

    net = nef.Network('Gate Test')
    net.make_input('in1', values=.5)
    net.make_input('in2', values=math.sin)

    net.make('A', neurons=300, dimensions=1)

    make(net=net, name='gate', gated='A',
        neurons=100, pstc=.01)

    net.connect('in1', 'A', pstc=1e-6)
    net.connect('in2', 'gate', pstc=1e-6)

    timesteps = 1000
    dt_step = 0.01
    t = np.linspace(dt_step, timesteps*dt_step, timesteps)
    pstc = 0.01

    I1p = net.make_probe('in1', dt_sample=dt_step, pstc=pstc)
    I2p = net.make_probe('in2', dt_sample=dt_step, pstc=pstc)
    gate_probe = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
    gated_probe1 = net.make_probe('gate', dt_sample=dt_step, pstc=pstc)
    gated_probe2 = net.make_probe('gate:addOne', dt_sample=dt_step, pstc=pstc)

    print "starting simulation"
    net.run(timesteps*dt_step)

    # plot the results
    plt.ioff(); plt.close(); 
    plt.subplot(3,1,1); plt.title('Input')
    plt.plot(t, I1p.get_data(), 'x'); plt.plot(t, I2p.get_data(), 'x')
    plt.subplot(3,1,2)
    plt.plot(gate_probe.get_data()); plt.title('A')
    plt.subplot(3,1,3); plt.title('Gate')
    plt.plot(gated_probe1.get_data())
    plt.plot(gated_probe2.get_data())
    plt.tight_layout()
    plt.show()

#test_gate()
