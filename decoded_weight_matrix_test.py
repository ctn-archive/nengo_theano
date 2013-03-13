"""This is a test file to test the decoded_weight_matrix parameter on addTermination, 
   here we test by creating inhibitory connections.

   Need to test
        - 1) inhibitory to ensemble connection
        - 2) inhibitory to network array connection
        - 3) inhibitory with scalar value
"""

import nef_theano as nef
import numpy as np
import matplotlib.pyplot as plt

neurons = 100
dimensions = 1
array_size = 3
inhib_scale = 10

net=nef.Network('WeightMatrix Test')
net.make_input('in1', 1, zero_after=2.5)
net.make_input('in2', [1, .5, 0])
net.make('A', neurons=neurons, dimensions=dimensions, intercept=(.1, 1))
net.make('B', neurons=neurons, dimensions=dimensions) # for test 1
net.make('B2', neurons=neurons, dimensions=dimensions, array_size=array_size) # for test 2 
#net.make('B3', neurons=neurons, dimensions=dimensions) # for test 3

# setup inhibitory scaling matrix
inhib_matrix = [[-10] * dimensions] * neurons 

# define our transform and connect up! 
net.connect('in1', 'A')
net.connect('in2', 'B', index_pre=0)
net.connect('in2', 'B2')
net.connect('A', 'B', decoded_weight_matrix=inhib_matrix)
net.connect('A', 'B2', decoded_weight_matrix=inhib_matrix) 
#net.connect('A', 'B3', decoded_weight_matrix=[-10]) 

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe(net.nodes['in1'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
I2p = net.make_probe(net.nodes['in2'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe(net.nodes['A'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe(net.nodes['B'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
B2p = net.make_probe(net.nodes['B2'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

plt.ion(); plt.close(); 
plt.subplot(611); plt.title('Input1')
plt.plot(Ip.get_data()); 
plt.subplot(612); plt.title('Input2')
plt.plot(I2p.get_data()); 
plt.subplot(613); plt.title('A = In1')
plt.plot(Ap.get_data())
plt.subplot(614); plt.title('B = In2(0) inhib by A')
plt.plot(Bp.get_data())
plt.subplot(615); plt.title('B2 = In2, network array inhib by A')
plt.plot(B2p.get_data())
#plt.subplot(616); plt.title('B3 = In2(0), inhib by scalar from A')
#plt.plot(B3vals)
