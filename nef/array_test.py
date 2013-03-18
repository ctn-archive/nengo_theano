"""This is a file to test the network array function, both with make_array, 
   and by using the array_size parameter in the network.make command"""

import nef_theano as nef
import numpy as np
import matplotlib.pyplot as plt

net=nef.Network('Array Test', seed=5)
net.make_input('in', [-1,0,0,0,0,1], zero_after=1.0)
net.make_array('A', neurons=50, array_size=3, dimensions=2, neuron_type='lif')
net.make('A2', neurons=50, array_size=2, dimensions=3, neuron_type='lif')
net.make('B', 200, 6, neuron_type='lif')
net.make('B2', 50, dimensions=1, array_size=6, neuron_type='lif')

net.connect('in', 'A')
net.connect('in', 'A2')
net.connect('in', 'B')
net.connect('in', 'B2')

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe(net.nodes['in'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe(net.nodes['A'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
A2p = net.make_probe(net.nodes['A2'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe(net.nodes['B'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
B2p = net.make_probe(net.nodes['B2'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

# plot the results
plt.ion(); plt.close(); 
plt.subplot(5,1,1)
plt.plot(t, Ip.get_data(), 'x'); plt.title('Input')
plt.subplot(5,1,2)
plt.plot(Ap.get_data()); plt.title('A, array_size=3, dim=2')
plt.subplot(5,1,3)
plt.plot(A2p.get_data()); plt.title('A2, array_size=2, dim=3')
plt.subplot(5,1,4)
plt.plot(Bp.get_data()); plt.title('B, array_size=1, dim=6')
plt.subplot(5,1,5)
plt.plot(B2p.get_data()); plt.title('B2, array_size=6, dim=1')
