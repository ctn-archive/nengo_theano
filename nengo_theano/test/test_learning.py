"""This is a test file to test basic learning
"""

import math
import time

import numpy as np
import matplotlib.pyplot as plt

import nengo_theano as nef

neurons = 30  # number of neurons in all ensembles

start_time = time.time()
net = nef.Network('Learning Test')

import random
class TrainingInput(nef.SimpleNode):
    def init(self):
        self.input_vals = np.arange(-1, 1, .2)
        self.period_length = 10
        self.choose_time = 0.0
    def origin_ILinput(self):
        if (self.t >= self.choose_time):
            # choose an input randomly from the set
            self.index = random.randint(0,9) 
            # specify the correct response for this input
            if (self.index < 5): self.correct_response = [.6]
            else: self.correct_response = [0.2]
            # update the time to next change the input again
            self.choose_time = self.t + self.period_length
        return [self.input_vals[self.index]]
    def origin_goal(self):
        return self.correct_response
    def reset(self, randomize=False):
        self.choose_time = 0.0
        nef.SimpleNode.reset(self, randomize)
net.add(TrainingInput('SNinput'))

#net.make_input('in', values=0.8)
net.make('A', neurons=neurons, dimensions=1)
net.make('B', neurons=2*neurons, dimensions=1)
net.make('error1', neurons=neurons, dimensions=1, mode='direct')

net.learn(pre='A', post='B', error='error1', rate=5e-3)

net.connect('SNinput:ILinput', 'A')
net.connect('A', 'error1')
net.connect('B', 'error1', weight=-1)

t_final = 10
dt_step = 0.001
pstc = 0.03

Ip = net.make_probe('SNinput:ILinput', dt_sample=dt_step, pstc=pstc)
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
