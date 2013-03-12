"""This is a test file to test the noise parameter on ensemble"""

import nef_theano as nef
import numpy as np
import matplotlib.pyplot as plt
import math

net=nef.Network('Noise Test')
net.make_input('in', value=math.sin)
net.make('A', neurons=300, dimensions=1, noise=1)
net.make('A2', neurons=300, dimensions=1, noise=100)
net.make('B', neurons=300, dimensions=2, noise=1000, noise_type='Gaussian')
net.make('C', neurons=100, dimensions=1, array_size=3, noise=10)

net.connect('in', 'A')
net.connect('in', 'A2')
net.connect('in', 'B')
net.connect('in', 'C')

timesteps = 500
# setup arrays to store data gathered from sim
Invals = np.zeros((timesteps, 1))
Avals = np.zeros((timesteps, 1))
A2vals = np.zeros((timesteps, 1))
Bvals = np.zeros((timesteps, 2))
Cvals = np.zeros((timesteps, 3))

print "starting simulation"
for i in range(timesteps):
    net.run(0.01)
    Invals[i] = net.nodes['in'].decoded_output.get_value() 
    Avals[i] = net.nodes['A'].origin['X'].decoded_output.get_value() 
    A2vals[i] = net.nodes['A2'].origin['X'].decoded_output.get_value() 
    Bvals[i] = net.nodes['B'].origin['X'].decoded_output.get_value() 
    Cvals[i] = net.nodes['C'].origin['X'].decoded_output.get_value() 

# plot the results
plt.ion(); plt.close()
plt.subplot(411); plt.title('Input')
plt.plot(Invals)
plt.subplot(412); plt.hold(1)
plt.plot(Avals); plt.plot(A2vals)
plt.legend(['A noise = 1', 'A2 noise = 100'])
plt.subplot(413); plt.title('B noise = 1000, type = gaussian')
plt.plot(Bvals)
plt.subplot(414); plt.title('C')
plt.plot(Cvals)
