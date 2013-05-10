from .. import nef_theano as nef

def make(net, name='Thalamus', neurons=50, dimensions=2, 
         inhib_scale=3, tau_inhib=.005):

    # create subnetwork to house the template
    th_net = net.make_subnetwork('Thalamus')

    # set up input relay
    th_net.make('input', neurons=1, dimensions=dimensions, mode='direct')
    # set up output relay
    th_net.make('output', neurons=1, dimensions=dimensions, mode='direct')

    # create a network array
    th_net.make(name=name, neurons=neurons, dimensions=1,
        array_size=dimensions, max_rate=(100,300), intercept=(-1, 0), 
        radius=1, encoders=[[1]])

    # setup inhibitory scaling matrix
    inhib_scaling_matrix = [[0]*dimensions for i in range(dimensions)]
    for i in range(dimensions):
        inhib_scaling_matrix[i][i] = -inhib_scale

    # setup inhibitory matrix
    inhib_matrix = []
    for i in range(dimensions):
        inhib_matrix_part = [[inhib_scaling_matrix[i]] * neurons]
        inhib_matrix.append(inhib_matrix_part[0])

    th_net.connect('input', 'Thalamus', transform=inhib_matrix, pstc=tau_inhib)

    def addOne(x): return [x[0]+1] 
    th_net.connect('Thalamus', 'output', func=addOne)
    
    return th_net

def test_thalamus():
    import numpy as np
    import matplotlib.pyplot as plt
    import math

    from .. import nef_theano as nef
    from .. import templates

    net = nef.Network('Thalamus Test')
    def func(x):
        return [math.sin(x), .5,.2]
    net.make_input('in', value=func)
    templates.thalamus.make(net=net, name='Thalamus', 
        neurons=300, dimensions=3, inhib_scale=3)

    net.connect('in', 'Thalamus.input')

    timesteps = 1000
    dt_step = 0.01
    t = np.linspace(dt_step, timesteps*dt_step, timesteps)
    pstc = 0.01

    Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
    Thp = net.make_probe('Thalamus.Thalamus:addOne', dt_sample=dt_step, pstc=pstc)

    print "starting simulation"
    net.run(timesteps*dt_step)

    # plot the results
    plt.ioff(); plt.close(); 
    plt.subplot(2,1,1)
    plt.plot(t, Ip.get_data(), 'x'); plt.title('Input')
    plt.subplot(2,1,2)
    plt.plot(Thp.get_data()); plt.title('Thalamus.output')
    plt.tight_layout()
    plt.show()

