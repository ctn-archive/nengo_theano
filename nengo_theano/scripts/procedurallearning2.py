import nef.nef_theano as nef
#from ..nef_theano import hpes_termination as learning
from nef.nef_theano.hPES_termination import hPESTermination as learning
import random
from datetime import datetime 
import sys
from numpy import arange

seed = random.randint(0, sys.maxint)

# constants / parameter setup etc
N1 = 100 # number of neurons
d1 = 1 # number of actions 
d2 = 1 # number of dimensions in each primitive
d3 = d2 # number of dimensions in the goal / actual states 
pstc = 0.01 # post-synaptic time constant
inhib_scale = 10 # weighting on inhibitory matrix
tau_inhib=0.005 # pstc for inhibitory connection

# Create the network object
net = nef.Network('Procedural_Learning', seed=seed) 
random.seed(seed)
print 'seed: ', seed

def make_TRN(name, neurons, dimensions, dim_fb_err, radius=1, tau_inhib=0.005, inhib_scale1=10, inhib_scale2=1):

    TRN = nef.Network(name)

    TRN.make('input', neurons=N1, dimensions=dimensions) # create population to be out
    TRN.make('thalamic input', neurons=N1, dimensions=dimensions) # create population to be out
    TRN.add(make_abs_val(name='abs_val_input', neurons=neurons, dimensions=dimensions)) # create a subnetwork to calculate the absolute value 
    for d in range(dimensions):
        TRN.make('output %d'%d, neurons=neurons, dimensions=1)#, intercept=(.05,1)) # create output relays

    # now we track the derivative of sum, and only let output relay the input
    # if the derivative is below a given threshold
    TRN.make('derivative', neurons=radius*neurons, dimensions=2, radius=radius) # create population to calculate the derivative
    TRN.connect('derivative', 'derivative', index_pre=0, index_post=1, pstc=0.05) # set up recurrent connection
    TRN.add(make_abs_val(name='abs_val_deriv', neurons=neurons, dimensions=1, intercept=(.1,1))) # create a subnetwork to calculate the absolute value 
    
    # create integrator that saturates quickly in response to any derivative signal and inhibits output, decreases in response to a second input 
    # the idea being that you connect system feedback error up to the second input so that BG contribution is inhibited unless there's FB error
    TRN.make('integrator', neurons=neurons, dimensions=1, intercept=(.1,1))
    TRN.connect('integrator', 'integrator', weight=1) # hook up to hold current value

    # connect it up!
    def subone(x): 
        for i in range(len(x)):
            x[i] = x[i] - 1
        return x # to shift thalamic range - 1 to 1
    for d in range(dimensions):
        TRN.connect('input', 'output %d'%d, pstc=1e-6, index_pre=d) # set up communication channel
        TRN.connect('thalamic input', 'output %d'%d, pstc=1e-6, index_pre=d, func=subone) # set up thalamic input 
    TRN.connect('input', 'abs_val_input.input', pstc=1e-6)
    TRN.connect('abs_val_input.output', 'derivative', index_post=0)
    
    def sub(x): return [x[0] - x[1]]
    TRN.connect('derivative', 'abs_val_deriv.input', func=sub)
    TRN.connect('abs_val_deriv.output', 'integrator', weight=10) # saturate integrator if there's any derivative
        
    # set up inhibitory matrix
    inhib_matrix1 = [[-inhib_scale1]] * neurons 
    inhib_matrix2 = [[-inhib_scale2]] * neurons * dim_fb_err
    TRN.get('integrator').addTermination('inhibition', inhib_matrix2, 1, False) # 1 is the pstc value
    for d in range(dimensions):
        TRN.get('output %d'%d).addTermination('inhibition', inhib_matrix1, tau_inhib, False)
        TRN.connect('integrator', TRN.get('output %d'%d).getTermination('inhibition'))
 
    TRN.add(make_abs_val(name='int-inhib', neurons=neurons, dimensions=dim_fb_err, intercept=(.05,1))) # create a subnetwork to calculate the absolute value 
    #TRN.make('err_step', neurons=1000, dimensions=1, encoders=[[1]]) # this is to standardize the amount of time it takes to inhibit the integrator, regardless of error magnitude
    TRN.make('err_step', neurons=N1, dimensions=1)
    #def step(x): # if err_step has anything to project, send 1 instead
    #    return 1
    def step(x):
        if x > 0: return 1
        else: return 0
    TRN.make('err_relay', neurons=N1, dimensions=1)
    TRN.connect('int-inhib.output', 'err_step')
    TRN.connect('err_step', 'err_relay', func=step)
    TRN.connect('err_relay', TRN.get('integrator').getTermination('inhibition'))

    return TRN

# Create function input
#======================

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

# Create populations
#===================
for d in range(d1): 
    net.add(make_cortical_action('action %d'%d, neurons=N1, dimensions=d2, action_vals=action_vals[d])) # create cortical actions
    net.make('ILP %d'%d, neurons=10*N1, dimensions=1) # create modulating ILP populations
net.make('input', neurons=N1, dimensions=d3) # population representing the input to the system
net.make('bg_input', neurons=10*N1, dimensions=d1)
net.make('bg_input_for_real', neurons=d1*N1, dimensions=d1+1) # +1 for default_competitor 
net.make('actual', neurons=N1, dimensions=d2) # where all the cortical action output is summed

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

# run and save info to file!
#===========================
'''trials = 500
log_dir = "scripts/results" # set up directory to save to 
for i in range(trials):

    log_name = "results-" + datetime.now().strftime("%Y%m%d-%H_%M_%S") + ".csv" # set filename
    logNode = nef.Log(net, "log", dir=log_dir, filename=log_name, interval=0.001) # setup data logger

    # add all nodes that we want data from
    logNode.add('input', origin='X')
    #logNode.add('SNinput', origin='constant')
    logNode.add('actual', origin='X')
    logNode.add('error_bg', origin='X')
    #logNode.add_spikes('bg_input') # record the neural activity of the striatum
    for d in range(d1):
        logNode.add('TRN.output %d'%d, origin='X') # bg contributed weights
        logNode.add('ILP %d'%d, origin='X') # cortex projected weights  
        #logNode.add_spikes('ILP %d'%d)

    net.network.simulator.run(0, 3, .001, False) # run the simulation
    net.network.removeStepListener(logNode) # remove the logNode to close the file we're writing to

    learned_weights = {} # create dictionary to store learned weights 
    for d in range(d1): # store all the learned weights
        learned_weights['ILP %d'%d] = net.get('ILP %d'%d).getTermination('input_00').getTransform()
    learned_weights['bg_input'] = net.get('bg_input').getTermination('ILP multiplexer_00').getTransform()
    
    net.network.simulator.resetNetwork(False, False) # reset the network

    for d in range(d1): # load in all the learned weights
        net.get('ILP %d'%d).getTermination('input_00').setTransform(learned_weights['ILP %d'%d], False)
    net.get('bg_input').getTermination('ILP multiplexer_00').setTransform(learned_weights['bg_input'], False)

    #print "trial: %d"%i
    '''

