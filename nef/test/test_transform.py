"""This is a test file to test the transform parameter on the connect function.
The transform matrix is post * pre dimensions"""

import math

import numpy as np
import matplotlib.pyplot as plt

from .. import nef_theano as nef

def func(x): 
    return [math.sin(x), -math.sin(x)]

net = nef.Network('Transform Test')
net.make_input('in', value=func)
net.make('A', 300, 3)

# define our transform and connect up! 
transform = [[0, 1], [1, 0], [1, -1]]
net.connect('in', 'A', transform=transform)

timesteps = 500
dt_step = 0.01
t = np.linspace(dt_step, timesteps*dt_step, timesteps)
pstc = 0.01
Ip = net.make_probe(net.nodes['in'].origin['X'].decoded_output,
                    dt_sample=dt_step, pstc=pstc)
Ap = net.make_probe(net.nodes['A'].origin['X'].decoded_output,
                    dt_sample=dt_step, pstc=pstc)

print "starting simulation"
net.run(timesteps * dt_step)

plt.ioff(); plt.clf(); 
plt.subplot(411); plt.title('Input')
plt.plot(Ip.get_data()); plt.legend(['In(0)','In(1)'])
plt.subplot(412); plt.title('A(0) = In(1)')
plt.plot(Ap.get_data()[:,0])
plt.subplot(413); plt.title('A(1) = In(0)')
plt.plot(Ap.get_data()[:,1])
plt.subplot(414); plt.title('A(2) = In(0) - In(1)')
plt.plot(Ap.get_data()[:,2])
plt.tight_layout()
plt.show()
