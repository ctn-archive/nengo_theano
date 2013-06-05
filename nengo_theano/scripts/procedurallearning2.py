import matplotlib.pyplot as plt
import numpy as np
import random
import sys
import time

import nengo_theano as nef

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
d1 = 1 # number of actions 
d2 = 1 # number of dimensions in each primitive
d3 = d2 # number of dimensions in the goal / actual states 
inhib_scale = 10 # weighting on inhibitory matrix
tau_inhib=0.005 # pstc for inhibitory connection

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
        self.input_vals = arange(-1, 1, .2)
        self.period_length = 2
        self.choose_time = 0.0
    def origin_ILinput(self):
        if (self.t >= self.choose_time):
            self.index = random.randint(0,9) # choose an input randomly from the set
            if (self.index < 5): # specify the correct response for this input
                self.correct_response = [.5]
            else:
                self.correct_response = [-.5]
            # update the time to next change the input again
            self.choose_time = self.t + self.period_length
        return [self.input_vals[self.index]]
    def origin_goal(self):
        return self.correct_response
    def origin_constant(self):
        return [.93]
    def reset(self, randomize=False):
        self.choose_time = 0.0
        nef.SimpleNode.reset(self, randomize)

net.add(TrainingInput('SNinput'))

action_vals = [[2]]
net.make_input('default_competitor', value=[1.5]) # value found from trial and error

# Cortical actions and ILP
#===================
for d in range(d1): 
    cortical_action.make(net=net, name='action %d'%d, 
        neurons=N1, dimensions=d2, action_vals=action_vals[d])) 
    # create modulating ILP populations
    net.make('ILP %d'%d, neurons=10*N1, dimensions=1) 
# population representing the input to the system
net.make('input', neurons=N1, dimensions=d3) 
net.make('bg_input', neurons=10*N1, dimensions=d1)
# +1 for default_competitor 
net.make('bg_input_for_real', neurons=d1*N1, dimensions=d1+1) 
# where all the cortical action output is summed
net.make('actual', neurons=N1, dimensions=d2) 

# Create networks for dot product and weighted summation
# relay population (for preserving dot1 output)
net.make(name='ILP multiplexer', neurons=2*d1*N1, dimensions=d1+1) 
# between error and primitives
net.add(make_dot(name='error_projection', neurons=N1, dimensions1=d1, dimensions2=d3)) 

# Create population for calculating system feedback error 
error_bg= net.make('error_bg', N1, d3)

# Make a basal ganglia model for weighting the primitives
bg = nef.templates.basalganglia.make(net, name="Basal Ganglia", dimensions=d1+1, neurons=50) # +1 for default_competitor 
# Make a thalamus model for flipping the BG inhibitory output into modulation values for the primitive weights
thalamus = nef.templates.thalamus.make(net=net, name='Thalamus', neurons=50, dimensions=d1+1, inhib_scale=1) # +1 for default_competitor  

# Create a modulated connection between the 'pre' and 'post' ensembles.
net.make(name='learn_signal_bg', neurons=d1*N1, dimensions=d1, noise=10, noise_frequency=1000) # the noise boosts learning speed
# basal ganglia learn connections
learning.make(net, errName='learn_signal_bg', N_err=N1, preName='ILP multiplexer', postName='bg_input', rate=4e-6, supervisionRatio=1) 
# add inhibitory connection to learn signal to prevent saliency modification when BG output inhibited
inhib_matrix = [[-inhib_scale]] * N1 * d1 
net.get('learn_signal_bg').addTermination('inhibition', inhib_matrix, tau_inhib, False)

# Set up TRN to gate error signal when cortex derivative is high
net.add(make_TRN('TRN', neurons=N1, dimensions=d1, dim_fb_err=d3))
for d in range(d1):
    # cortical learn connections 
    learning.make(net, errName='TRN.output %d'%d, N_err=1, preName='input', postName='ILP %d'%d, rate=4e-10, supervisionRatio=.5) 


# Connect up network
#===================
net.connect(net.get('SNinput').getOrigin('ILinput'), 'input')
net.connect(net.get('SNinput').getOrigin('goal'), 'error_bg')
for d in range(d1):
    net.connect('action %d.output'%d, 'actual') 
    net.connect('ILP %d'%d, 'TRN.input', weight=-1, index_post=d)
    net.connect('ILP %d'%d, 'action %d.input'%d)
    net.connect('TRN.output %d'%d, 'action %d.input'%d, pstc=.5) 
    net.connect('ILP %d'%d, 'ILP multiplexer')

# weight = 2, then -1 in TRN to get -1 to 1 range
net.connect(thalamus.getOrigin('xBiased'), 'TRN.thalamic input', index_pre=range(d1), weight=2) 
# connect up TRN integrator population to also inhibit BG learning
net.connect('TRN.integrator', net.get('learn_signal_bg').getTermination('inhibition'))
# connect up feedback error to disinhibit thalamic output
net.connect('error_bg', 'TRN.int-inhib.input', weight=1)

# set up error_bg
net.connect('actual', 'error_bg', weight=-1) 
net.connect('error_bg', 'error_projection.input2')
for d in range(d1):
    net.connect('action %d.action'%d, 'error_projection.input1', index_post=range(d2*d,d2*d+d2))

# connect up bg
net.connect('bg_input', 'bg_input_for_real', index_post=range(d1)) 
net.connect('default_competitor', 'bg_input_for_real', index_post=d1)
net.connect('bg_input_for_real', bg.getTermination('input'))
net.connect(bg.getOrigin('output'), thalamus.getTermination('bg_input'))

net.connect('error_projection.output', 'learn_signal_bg') # connect up error signal to bg learn signal

