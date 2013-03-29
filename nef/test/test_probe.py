"""This is a file to test the probe class, and it's ability to record data
and write to file.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

from .. import nef_theano as nef

net = nef.Network('Probe Test')
net.make_input('in', math.sin)
net.make('A', 50, 1)
net.make('B', 5, 1)

net.connect('in', 'A')
net.connect('in', 'B')

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step)
BpSpikes = net.make_probe('B', data_type='spikes', dt_sample=dt_step)

print "starting simulation"
net.run(timesteps*dt_step)

# plot the results
plt.ioff(); plt.close(); 
plt.subplot(3,1,1)
plt.plot(t, Ip.get_data(), 'x'); plt.title('Input')
plt.subplot(3,1,2)
plt.plot(Ap.get_data()); plt.title('A')
plt.subplot(3,1,3); plt.hold(1)
plt.plot(Bp.get_data())
for row in BpSpikes.get_data().T: 
    plt.plot(row[0]); 
plt.title('B')
plt.show()
