"""This is a test file to test the func parameter on the connect method"""

import math
import matplotlib.pyplot as plt
import numpy as np
import time

import nengo_theano as nef

start_time = time.time()

net = nef.Network('Function Test')
net.make_input('in', values=math.sin)
net.make('A', neurons=250, dimensions=1)
net.make('B', neurons=250, dimensions=3)

# function example for testing
def square(x):
    return [-x[0]*x[0], -x[0], x[0]]

net.connect('in', 'A')
net.connect('A', 'B', func=square, pstc=0.1)

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.03

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)

build_time = time.time()
print "build time: ", build_time - start_time

print "starting simulation"
net.run(timesteps * dt_step)
print "sim time: ", time.time() - build_time

# plot the results
plt.ioff(); plt.clf(); plt.hold(1);
plt.plot(Ip.get_data())
plt.plot(Ap.get_data())
plt.plot(Bp.get_data())
plt.legend(['Input','A','B0','B1','B2'])
plt.tight_layout()
plt.show()
