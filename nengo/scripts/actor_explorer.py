import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time

from .. import nef_theano as nef

import abs_val; reload(abs_val)
import cortical_action; reload(cortical_action)
import dot_product; reload(dot_product)
import learn_bias; reload(learn_bias)
import TRN; reload(TRN)

from ..templates import basalganglia; reload(basalganglia)
from ..templates import thalamus; reload(thalamus)

# Set random seed
#======================
seed = random.randint(0, sys.maxint)
random.seed(seed)
print "seed: ", seed

# Constants 
#======================
N1 = 100 # number of neurons 
d2 = 1 # number of dimensions in each primitive
d3 = d2 # number of dimensions in the goal / actual states 
inhib_scale = 500 # weighting on inhibitory matrix
tau_inhib = 0.05

testing = 1 # set testing = True
N2 = N1; mode = 'spiking'
if (testing): N2 = 1; mode = 'direct'

# Create the network object
#======================
start_time = time.time()
net = nef.Network('Actor Explorer') 

# Create function input
#======================
# default competitor value found from trial and error
net.make_input('default_competitor', values=[1.5]) 

class TrainingInput(nef.SimpleNode):
    def init(self):
        self.input_vals = np.arange(-1, 1, .2)
        self.period_length = .5
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

# Cortical actions and ILP 
#===================
neurons_explorer = 50
cortical_action.make_cortical_action(net=net, name='action', 
    neurons=N2, dimensions=d2, action_vals=[1], mode='direct') 
# create modulating ILP populations
net.make('ILP', neurons=2*N1, dimensions=1) 
net.make('ILP_relay', neurons=neurons_explorer, dimensions=1)

net.connect('SNinput:ILinput', 
            'ILP')
net.connect('ILP', 
            'ILP_relay')

# Explorer
#==================
net.make('explorer', neurons=neurons_explorer, dimensions=1)

# Actor
#==================
net.make('actor', neurons=N1, dimensions=1)#, radius=.5)
net.make(name='learn_signal_actor', neurons=N1, dimensions=1)
    #, noise=10)

# actor learning connections, slower than explorer, still fast
net.learn(
    pre='ILP', post='actor',
    error='learn_signal_actor', 
    rate=5e-6, supervision_ratio=1) 

# Basal ganglia
#==================
# +1 for default_competitor 
net.make('bg_input', neurons=N2, dimensions=1+1, mode=mode) 

# Make a basal ganglia model for weighting the primitives
basalganglia.make(net=net, name="Basal Ganglia", 
    dimensions=1+1, neurons=50) # +1 for default_competitor 
# Make a thalamus model for flipping the BG inhibitory output 
# into modulation values for the primitive weights
thalamus.make(net=net, name='Thalamus', 
    neurons=50, dimensions=1+1, inhib_scale=1) # +1 for default_competitor  
# BG input
net.connect('actor', 
            'bg_input', 
            index_post=0) 
net.connect('explorer', 
            'bg_input', index_post=0) 
net.connect('default_competitor', 
            'bg_input', index_post=1)
net.connect('bg_input', 
            'Basal Ganglia.input')
# BG output
net.connect('Basal Ganglia.output', 
            'Thalamus.input')
net.connect('Thalamus.output',
            'action.input', 
            index_pre=0)

# Error signals
#==================
net.make('error', neurons=N2, dimensions=d3, mode=mode)
# for projecting fb error into action space
dot_product.make(net=net, name='error_projection', 
    neurons=N2, dimensions1=1, dimensions2=d3, mode='direct') 
net.make('error_actor_gate', neurons=N2, dimensions=2, mode=mode)

# explorer learning connections, really fast
net.learn(
    pre='ILP_relay', post='explorer',
    error='error_projection.output', 
    rate=7e-5, supervision_ratio=1) 
# also add in a positive bias signal whenever actor + explorer value < 0
learn_bias.make(net=net, name='explorer_learn_bias', neurons=50)
net.connect('explorer',
            'explorer_learn_bias.input')
net.connect('actor',
            'explorer_learn_bias.input')
net.connect('explorer_learn_bias.output',
            'error_projection.output')

net.connect('SNinput:goal', 
            'error')
net.connect('error', 
            'error_projection.input_vector')
net.connect('action.action', 
            'error_projection.input_matrix')

net.connect('error',
            'error_actor_gate',
            index_post=0)
net.connect('explorer', 
            'error_actor_gate', 
            index_post=1)

def error_thresh(x): 
    if abs(x[0]) < .1: return x[1]
    return 0
net.connect('error_actor_gate', 
            'learn_signal_actor',
            func=error_thresh)

# System output signal
#===================
# where all the cortical action output is summed
net.make('actual', neurons=N2, dimensions=d2, mode=mode) 

net.connect('action.output', 
            'actual') 
net.connect('actual', 
            'error', 
            weight=-1) 

build_time = time.time()
print "build time: ", build_time - start_time

# Record data
#====================
dt = .001
#SNin_probe = net.make_probe('SNinput:ILinput')
SNgoal_probe = net.make_probe('SNinput:goal', dt_sample=dt*10)
exp_probe = net.make_probe('explorer', dt_sample=dt*10)
act_probe = net.make_probe('actor', dt_sample=dt*10)
#lsa_probe = net.make_probe('learn_signal_actor', dt_sample=dt*10)
out_probe = net.make_probe('action.output', dt_sample=dt*10)
err_probe = net.make_probe('error', dt_sample=dt*10)
lb_probe = net.make_probe('explorer_learn_bias.output', dt_sample=dt*10)

# Run the system
#====================
runtime = 100
net.run(runtime)
print "simulated %.2f seconds in %.2fs real time"%(runtime, time.time() - build_time)

# Plot results
#====================
'''plt.subplot(611); plt.title('input'); plt.ylim([-1,1])
plt.plot(SNin_probe.get_data())
plt.subplot(612); plt.title('goal'); plt.ylim([-1,1])
plt.plot(SNgoal_probe.get_data())
plt.subplot(613); plt.title('explorer'); plt.ylim([-1,1])
plt.plot(exp_probe.get_data())
plt.subplot(614); plt.title('actor'); plt.ylim([-1,1])
plt.plot(act_probe.get_data())
plt.subplot(615); plt.title('learn_signal_actor'); plt.ylim([-1,1])
plt.plot(lsa_probe.get_data())
plt.subplot(616); plt.title('output'); plt.ylim([-1,1])
plt.plot(out_probe.get_data())
plt.tight_layout()
plt.show()'''
plt.plot()
plt.show()
