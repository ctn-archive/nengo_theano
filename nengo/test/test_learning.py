"""This is a test file to test basic learning
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

neurons = 30  # number of neurons in all ensembles

start_time = time.time()
net = nef.Network('Learning Test')
net.make_input('in', value=0.8)
net.make('A', neurons=neurons, dimensions=1)
net.make('B', neurons=2*neurons, dimensions=1)
net.make('error1', neurons=neurons, dimensions=1)

net.learn(pre='A', post='B', error='error1')

net.connect('in', 'A')
net.connect('A', 'error1')
net.connect('B', 'error1', weight=-1)

t_final = 2
dt_step = 0.001
pstc = 0.03

Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe('B', dt_sample=dt_step, pstc=pstc)
E1p = net.make_probe('error1', dt_sample=dt_step, pstc=pstc)

build_time = time.time()
print "build time: ", build_time - start_time
net.run(t_final)
print "sim time: ", time.time() - build_time

plt.ioff(); plt.close()

t = np.linspace(0, t_final, len(Ap.get_data()))

plt.plot(t, Ap.get_data())
plt.plot(t, Bp.get_data())
plt.plot(t, E1p.get_data())
plt.legend(['A', 'B', 'error'])
plt.title('Normal learning')
plt.tight_layout()
plt.show()
