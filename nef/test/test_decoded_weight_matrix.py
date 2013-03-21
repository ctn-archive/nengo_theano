"""This is a test file to test the decoded_weight_matrix parameter on
addTermination. Here we test by creating inhibitory connections.

TODO:
  1. inhibitory to ensemble connection with T = (neurons x dimensions)
  2. inhibitory to network array connection with T = (neurons x dimensions)
  3. inhibitory to network array with T = (array_size x neurons x dimensions)
  4. inhibitory to network array with T = (array_size * neurons x dimensions)
"""

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

neurons = 100
dimensions = 1
array_size = 3
inhib_scale = 10

net = nef.Network('WeightMatrix Test')
net.make_input('in1', 1, zero_after=2.5)
net.make_input('in2', [1, .5, 0])
net.make('A', neurons=neurons, dimensions=dimensions, intercept=(.1, 1))
net.make('B', neurons=neurons, dimensions=dimensions) # for test 1
net.make('B2', neurons=neurons, dimensions=dimensions, array_size=array_size) # for test 2 
net.make('B3', neurons=neurons, dimensions=dimensions, array_size=array_size) # for test 3 
net.make('B4', neurons=neurons, dimensions=dimensions, array_size=array_size) # for test 4

# setup inhibitory scaling matrix
inhib_matrix_1 = [[-10] * dimensions] * neurons # for test 1 and 2
inhib_matrix_2 = [[[-10] * dimensions] * neurons] * array_size # for test 3 
inhib_matrix_3 = [[-10] * dimensions] * neurons * array_size # for test 4

# define our transform and connect up! 
net.connect('in1', 'A')
net.connect('in2', 'B', index_pre=0)
net.connect('in2', 'B2')
net.connect('in2', 'B3')
net.connect('in2', 'B4')
net.connect('A', 'B', transform=inhib_matrix_1) # for test 1
net.connect('A', 'B2', transform=inhib_matrix_1) # for test 2
net.connect('A', 'B3', transform=inhib_matrix_2) # for test 3
net.connect('A', 'B4', transform=inhib_matrix_3) # for test 4

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe(net.nodes['in1'].origin['X'].decoded_output,
                    dt_sample=dt_step, pstc=pstc)
I2p = net.make_probe(net.nodes['in2'].origin['X'].decoded_output,
                     dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe(net.nodes['A'].origin['X'].decoded_output,
                    dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe(net.nodes['B'].origin['X'].decoded_output,
                    dt_sample=dt_step, pstc=pstc)
B2p = net.make_probe(net.nodes['B2'].origin['X'].decoded_output,
                     dt_sample=dt_step, pstc=pstc)
B3p = net.make_probe(net.nodes['B3'].origin['X'].decoded_output,
                     dt_sample=dt_step, pstc=pstc)
B4p = net.make_probe(net.nodes['B4'].origin['X'].decoded_output,
                     dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps*dt_step)

plt.ioff(); plt.close(); 
plt.subplot(711); plt.title('Input1')
plt.plot(Ip.get_data()); 
plt.subplot(712); plt.title('Input2')
plt.plot(I2p.get_data()); 
plt.subplot(713); plt.title('A = In1')
plt.plot(Ap.get_data())
plt.subplot(714); plt.title('B = In2(0) inhib by A')
plt.plot(Bp.get_data())
plt.subplot(715); plt.title('B2 = In2, network array inhib by A')
plt.plot(B2p.get_data())
plt.subplot(716); plt.title('B3 = In2(0), inhib by scalar from A')
plt.plot(B3p.get_data())
plt.subplot(717); plt.title('B4 = In2(0), inhib by scalar from A')
plt.plot(B4p.get_data())
plt.tight_layout()
plt.show()
