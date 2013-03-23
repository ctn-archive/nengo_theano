"""This is a file to test the encoders parameter on ensembles"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

build_time_start = time.time()

net = nef.Network('Encoder Test')
net.make_input('in', math.sin)
net.make('A', 100, 1)
net.make('B', 100, 1, encoders=[[1]], intercept=(0, 1.0))

net.connect('in', 'A')
net.connect('A', 'B')

timesteps = 1000
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01
Ip = net.make_probe(
    net.nodes['in'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe(
    net.nodes['A'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe(
    net.nodes['B'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)

build_time_end = time.time()

print "starting simulation"
net.run(timesteps * dt_step)

sim_time_end = time.time()
print "\nBuild time: %0.10fs" % (build_time_end - build_time_start)
print "Sim time: %0.10fs" % (sim_time_end - build_time_end)

plt.ioff(); plt.close()
plt.subplot(311); plt.title('Input')
plt.plot(t, Ip.get_data())
plt.subplot(312); plt.title('A')
plt.plot(t, Ap.get_data())
plt.subplot(313); plt.title('B')
plt.plot(t, Bp.get_data())
plt.tight_layout()
plt.show()

