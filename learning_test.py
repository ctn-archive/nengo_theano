"""This is a test file to test basic learning"""

import nef_theano as nef
import numpy as np
import math
import time

import matplotlib.pyplot as plt
plt.ion()

neurons = 100 # number of neurons in all ensembles

net=nef.Network('Learning Test')
net.make_input('in', value=0.8)
# net.make_input('in', value=math.sin)
timer = time.time()
A = net.make('A', neurons=neurons, dimensions=1)
B = net.make('B', neurons=neurons, dimensions=1)
error = net.make('error', neurons=neurons, dimensions=1)
print "Made populations:", time.time() - timer

net.learn('A', 'B', 'error')

net.connect('in', 'A')
net.connect('A', 'error')
net.connect('B', 'error', weight=-1)

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
tau = 0.03

Ip = net.make_probe(net.nodes['in'].decoded_output, dt_sample=dt_step, tau=tau)
Ap = net.make_probe(net.nodes['A'].origin['X'].decoded_output, dt_sample=dt_step, tau=tau)
Bp = net.make_probe(net.nodes['B'].origin['X'].decoded_output, dt_sample=dt_step, tau=tau)
Ep = net.make_probe(net.nodes['error'].origin['X'].decoded_output, dt_sample=dt_step, tau=tau)

print "starting simulation"
net.run(timesteps*dt_step)

plt.figure(1)
plt.clf()
plt.plot(t, Ap.get_data())
plt.plot(t, Bp.get_data())
plt.plot(t, Ep.get_data())
plt.legend(['A', 'B', 'error'])


# setup arrays to store data gathered from sim
# Invals = np.zeros((timesteps, 1))
# Avals = np.zeros((timesteps, 1))
# Bvals = np.zeros((timesteps, 1))
# Evals = np.zeros((timesteps, 1))

# print "starting simulation"
# for i in range(timesteps):
#     net.run(dt_step)
#     Invals[i] = net.nodes['in'].decoded_output.get_value() 
#     Avals[i] = net.nodes['A'].origin['X'].decoded_output.get_value() 
#     Bvals[i] = net.nodes['B'].origin['X'].decoded_output.get_value() 
#     Evals[i] = net.nodes['error'].origin['X'].decoded_output.get_value()

# def lowpass_filter(signal, dt, tau):
#     vi = np.zeros_like(signal[0])
#     for i, point in enumerate(signal):
#         vi = vi + (dt/tau) * (point - vi)
#         signal[i] = vi

# tau = 0.3
# lowpass_filter(Avals, dt_step, tau)
# lowpass_filter(Bvals, dt_step, tau)
# lowpass_filter(Evals, dt_step, tau)

# plot the results

# plt.figure(1)
# plt.clf()
# plt.plot(t, Avals)
# plt.plot(t, Bvals)
# plt.plot(t, Evals)
# plt.legend(['A', 'B', 'error'])

