"""This is a test file to test the SimpleNode object and the ability
   to project in signals from an ensemble."""

import math
import random

import numpy as np
import matplotlib.pyplot as plt
import theano

import nengo_theano as nef

net = nef.Network('SimpleNode Test')

class TrainingInput(nef.simplenode.SimpleNode):
    def init(self):
        self.add_input(name='input_1', dimensions=4)
        self.b = self.input['input_1'].value * -1.0

    def origin_test1(self):
        return 3.0 * self.b

    def origin_regular(self):
        return [math.sin(self.t*100)]
    
    def reset(self, **kwargs):
        nef.SimpleNode.reset(self, **kwargs)

net.add(TrainingInput('input'))

net.make_input('in', values=[1.7685, -.3, .4, -.5])

net.make('A', neurons=100, array_size=2, dimensions=2)

net.connect('in', 'A')
net.connect('A', 'input:input_1')

timesteps = 500
dt_step = 0.001
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01
Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
Inp = net.make_probe('input:test1', dt_sample=dt_step, pstc=pstc)
Inp2 = net.make_probe('input:regular', dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps * dt_step)

# plot the results
plt.ioff(); plt.close(); 
plt.subplot(311); plt.title('A'); 
plt.plot(Ap.get_data())
plt.subplot(312); plt.title('input'); 
plt.plot(Inp.get_data()[:,0])
plt.subplot(313); plt.title('10'); 
plt.plot(Inp2.get_data())
plt.tight_layout()
plt.show()
