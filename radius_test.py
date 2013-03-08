import nef_theano as nef
import math
import numpy as np
import matplotlib.pyplot as plt

def sin3(x):
    return math.sin(x) * 3
net=nef.Network('Encoder Test')
net.make_input('in', value=sin3)
net.make('A', 1000, 1, radius=5)
net.make('B', 300, 1, radius=.5)
net.make('C', 1000, 1, radius=10)
net.make('D', 300, 1, radius=6)

net.connect('in', 'A')
net.connect('A', 'B')
def pow(x):
    return [xval**2 for xval in x]
net.connect('A', 'C', func=pow)
def mult(x):
    return [xval*2 for xval in x]
net.connect('A', 'D', func=mult)

timesteps = 500
Fvals = np.zeros((timesteps,1))
Avals = np.zeros((timesteps,1))
Bvals = np.zeros((timesteps,1))
Cvals = np.zeros((timesteps,1))
Dvals = np.zeros((timesteps,1))
for i in range(timesteps):
    net.run(0.01)
    #print net.nodes['A'].origin['X'].projected_value.get_value(), net.nodes['B'].origin['X'].projected_value.get_value(), net.nodes['C'].origin['X'].projected_value.get_value()
     #net.nodes['B'].accumulator[0.01].projected_value.get_value(), net.nodes['C'].accumulator[0.01].projected_value.get_value()
    Fvals[i] = net.nodes['in'].projected_value.get_value() #net.nodes['A'].accumulator[0.01].projected_value.get_value()
    Avals[i] = net.nodes['A'].origin['X'].projected_value.get_value() #net.nodes['B'].accumulator[0.01].projected_value.get_value() # get the post-synaptic values because they're already filtered
    Bvals[i] = net.nodes['B'].origin['X'].projected_value.get_value() #net.nodes['C'].accumulator[0.01].projected_value.get_value()
    Cvals[i] = net.nodes['C'].origin['X'].projected_value.get_value() # net.nodes['D'].accumulator[0.01].projected_value.get_value() 
    Dvals[i] = net.nodes['D'].origin['X'].projected_value.get_value() # net.nodes['D'].accumulator[0.01].projected_value.get_value() 

plt.ion(); plt.clf(); plt.hold(1);
plt.plot(Fvals)
plt.plot(Avals)
plt.plot(Bvals)
plt.plot(Cvals)
plt.plot(Dvals)
plt.legend(['Input','A','B','C'])