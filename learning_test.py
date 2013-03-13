"""This is a test file to test basic learning

   Tests with both a regular post ensemble and a network array
"""

import nef_theano as nef
import numpy as np
import math
import time

import matplotlib.pyplot as plt
plt.ion()

neurons = 10 # number of neurons in all ensembles

net=nef.Network('Learning Test')
net.make_input('in', value=0.8)
# net.make_input('in', value=math.sin)
timer = time.time()
net.make('A', neurons=neurons, dimensions=1)
net.make('B', neurons=neurons, dimensions=1)
# net.make('C', neurons=neurons, dimensions=1, array_size=3)
net.make('error1', neurons=neurons, dimensions=1)
# net.make('error2', neurons=neurons, dimensions=1)
print "Made populations:", time.time() - timer

L = net.learn(pre='A', post='B', error='error1')
# net.learn(pre='A', post='C', error='error2')

net.connect('in', 'A')
net.connect('A', 'error1')
# net.connect('A', 'error2')
net.connect('B', 'error1', weight=-1)
# net.connect('C', 'error2', weight=-1)

t_final = 10.
dt_step = 0.01
pstc = 0.03

Ip = net.make_probe(net.nodes['in'].decoded_output, dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe(net.nodes['A'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
Bp = net.make_probe(net.nodes['B'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
# Cp = net.make_probe(net.nodes['C'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
E1p = net.make_probe(net.nodes['error1'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)
# E2p = net.make_probe(net.nodes['error2'].origin['X'].decoded_output, dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(t_final)

plt.figure(1)
plt.clf()

t = np.linspace(0, t_final, len(Ap.get_data()))

# plt.subplot(211)
plt.plot(t, Ap.get_data())
plt.plot(t, Bp.get_data())
plt.plot(t, E1p.get_data())
plt.legend(['A', 'B', 'error'])
plt.title('Normal learning')


# plt.plot(t, E2p.get_data())

# plt.subplot(312); plt.title('B')
# plt.plot(t, Bp.get_data())
# plt.subplot(313); plt.title('C')
# plt.plot(t, Cp.get_data())
