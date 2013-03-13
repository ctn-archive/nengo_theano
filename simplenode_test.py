"""This is a test file to test the SimpleNode object"""

import nef_theano as nef
import numpy as np
import matplotlib.pyplot as plt
import math
import random

net=nef.Network('SimpleNode Test')

class TrainingInput(nef.simplenode.SimpleNode):
    def init(self):
        self.input_vals = np.arange(-1, 1, .2)
        self.period_length = 2
        self.choose_time = 0.0

    def origin_test1(self):
        if (self.t >= self.choose_time):
            self.index = random.randint(0,9) # choose an input randomly from the set
            if (self.index < 5): # specify the correct response for this input
                self.correct_response = [.5]
            else:
                self.correct_response = [-.5]
            # update the time to next change the input again
            self.choose_time = self.t + self.period_length
        return [self.input_vals[self.index]]

    def origin_test2(self):
        return self.correct_response

    def origin_test3(self):
        return [.93, -1, -.1]

    def reset(self, **kwargs):
        self.choose_time = 0.0
        nef.SimpleNode.reset(self, **kwargs)

net.add(TrainingInput('SNinput'))

net.make('A', neurons=300, dimensions=1)
net.make('B', neurons=300, dimensions=1)
net.make('C', neurons=300, dimensions=3)

net.connect('SNinput', 'A', origin_name='test1')
net.connect('SNinput', 'B', origin_name='test2')
net.connect('SNinput', 'C', origin_name='test3')

timesteps = 500
# setup arrays to store data gathered from sim
In1vals = np.zeros((timesteps, 1))
In2vals = np.zeros((timesteps, 1))
In3vals = np.zeros((timesteps, 3))
Avals = np.zeros((timesteps, 1))
Bvals = np.zeros((timesteps, 1))
Cvals = np.zeros((timesteps, 3))

print "starting simulation"
for i in range(timesteps):
    net.run(0.01)
    In1vals[i] = net.nodes['SNinput'].origin['test1'].decoded_output.get_value() 
    In2vals[i] = net.nodes['SNinput'].origin['test2'].decoded_output.get_value() 
    In3vals[i] = net.nodes['SNinput'].origin['test3'].decoded_output.get_value() 
    Avals[i] = net.nodes['A'].origin['X'].decoded_output.get_value() 
    Bvals[i] = net.nodes['B'].origin['X'].decoded_output.get_value()
    Cvals[i] = net.nodes['C'].origin['X'].decoded_output.get_value()

# plot the results
plt.ion(); plt.close(); 
plt.subplot(411); plt.title('SNinput'); 
plt.hold(1)
plt.plot(In1vals); plt.plot(In2vals); plt.plot(In3vals)
plt.legend(['test1','test2','test3'])
plt.subplot(412); plt.title('A'); 
plt.plot(Avals)
plt.subplot(413); plt.title('B'); 
plt.plot(Bvals)
plt.subplot(414); plt.title('C'); 
plt.plot(Cvals)
