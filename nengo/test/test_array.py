"""This is a file to test the network array function, both with make_array, 
and by using the array_size parameter in the network.make command.

"""

import unittest

# Attempt to import nef from nef_theano, if that fails assume this script is being run
# from nengo itself. This is to support running this unittest from nengo itself
try:
    from .. import nef_theano as nef
    is_nef_theano = True
except ImportError:
    import nef
    is_nef_theano = False

if is_nef_theano:
    import numpy as np
else:
    import numeric as np

class TestArray(unittest.TestCase):
    def setUp(self):
        self.timesteps = 200
        self.dt = 0.01

        self.neurons = 40

        self.net = nef.Network('Array Test', seed = 50)
        self.net.make_input('in', np.arange(-1, 1, .34), zero_after = 1.0)
        #self.net.make_input('in', value = 1, zero_after = 1.0)
        self.net.make_array('A', neurons = self.neurons, array_size = 1, dimensions = 6)
        self.net.make('A2', neurons = self.neurons, array_size = 2, dimensions = 3)
        self.net.make('B', neurons = self.neurons, array_size = 3, dimensions = 2)
        self.net.make('B2', neurons = self.neurons, array_size = 6, dimensions = 1)

        self.net.connect('in', 'A')
        self.net.connect('in', 'A2')
        self.net.connect('in', 'B')
        self.net.connect('in', 'B2')
           
        

    def test_representation(self):
    
    def plot():
        
import matplotlib.pyplot as plt



timesteps  =  200
dt_step  =  0.01
t  =  np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc  =  0.01

Ip  =  net.make_probe(
    net.nodes['in'].origin['X'].decoded_output, dt_sample = dt_step, pstc = pstc)
Ap  =  net.make_probe(
    net.nodes['A'].origin['X'].decoded_output, dt_sample = dt_step, pstc = pstc)
A2p  =  net.make_probe(
    net.nodes['A2'].origin['X'].decoded_output, dt_sample = dt_step, pstc = pstc)
Bp  =  net.make_probe(
    net.nodes['B'].origin['X'].decoded_output, dt_sample = dt_step, pstc = pstc)
B2p  =  net.make_probe(
    net.nodes['B2'].origin['X'].decoded_output, dt_sample = dt_step, pstc = pstc)

print "starting simulation"
net.run(timesteps*dt_step)

# plot the results
plt.ioff(); plt.close(); 
plt.subplot(5,1,1); plt.ylim([-1.5,1.5])
plt.plot(t, Ip.get_data(), 'x'); plt.title('Input')
plt.subplot(5,1,2); plt.ylim([-1.5,1.5])
plt.plot(Ap.get_data()); plt.title('A, array_size = 1, dim = 6')
plt.subplot(5,1,3); plt.ylim([-1.5,1.5])
plt.plot(A2p.get_data()); plt.title('A2, array_size = 2, dim = 3')
plt.subplot(5,1,4); plt.ylim([-1.5,1.5])
plt.plot(Bp.get_data()); plt.title('B, array_size = 3, dim = 2')
plt.subplot(5,1,5); plt.ylim([-1.5,1.5])
plt.plot(B2p.get_data()); plt.title('B2, array_size = 6, dim = 1')
plt.tight_layout()
plt.show()

timesteps = 200
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
A2p = net.make_probe('A2', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
B2p = net.make_probe('B2', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

# plot the results
plt.ioff(); plt.close(); 
plt.subplot(5,1,1); plt.ylim([-1.5,1.5])
plt.plot(t, Ip.get_data(), 'x'); plt.title('Input')
plt.subplot(5,1,2); plt.ylim([-1.5,1.5])
plt.plot(Ap.get_data()); plt.title('A, array_size=1, dim=6')
plt.subplot(5,1,3); plt.ylim([-1.5,1.5])
plt.plot(A2p.get_data()); plt.title('A2, array_size=2, dim=3')
plt.subplot(5,1,4); plt.ylim([-1.5,1.5])
plt.plot(Bp.get_data()); plt.title('B, array_size=3, dim=2')
plt.subplot(5,1,5); plt.ylim([-1.5,1.5])
plt.plot(B2p.get_data()); plt.title('B2, array_size=6, dim=1')
plt.tight_layout()
plt.show()
