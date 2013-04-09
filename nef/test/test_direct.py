"""This is a file to test the direct mode on ensembles"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

build_time_start = time.time()

net = nef.Network('Direct Mode Test')
net.make_input('in', math.sin)
net.make('A', 100, 1)
net.make('B', 1, 1, mode='direct')
net.make('C', 100, 1)

net.connect('in', 'A')
net.connect('A', 'B')
net.connect('B', 'C')

timesteps = 1000
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01
Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
Cp = net.make_probe('C', dt_sample=dt_step, pstc=pstc)

build_time_end = time.time()

print "starting simulation"
net.run(timesteps * dt_step)

sim_time_end = time.time()
print "\nBuild time: %0.10fs" % (build_time_end - build_time_start)
print "Sim time: %0.10fs" % (sim_time_end - build_time_end)

plt.ioff(); plt.close()
plt.subplot(411); plt.title('Input')
plt.plot(t, Ip.get_data())
plt.subplot(412); plt.title('A = spiking')
plt.plot(t, Ap.get_data())
plt.subplot(413); plt.title('B = direct')
plt.plot(t, Bp.get_data())
plt.subplot(414); plt.title('B = spiking')
plt.plot(t, Cp.get_data())
plt.tight_layout()
plt.show()

