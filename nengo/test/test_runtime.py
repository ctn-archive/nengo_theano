"""This test file is for checking the run time of the theano code."""

import math
import time

from nengo import nef_theano as nef
from nengo.nef_theano.simulator import Simulator
from nengo.nef_theano.simulator_ocl import SimulatorOCL

net=nef.Network('Runtime Test', seed=123)
net.make_input('in', value=math.sin)
net.make('A', 1000, 1)
net.make('B', 1000, 1)
net.make('C', 1000, 1)
net.make('D', 1000, 1)

# some functions to use in our network
def pow(x):
    return [xval**2 for xval in x]

def mult(x):
    return [xval*2 for xval in x]

net.connect('in', 'A')
net.connect('A', 'B')
net.connect('A', 'C', func=pow)
net.connect('A', 'D', func=mult)
net.connect('D', 'B', func=pow) # throw in some recurrency whynot

if 0:
    start_time = time.time()
    print "starting simulation (net.run)"
    net.run(0.5)
    print "runtime: ", time.time() - start_time, "seconds"

if 0:
    sim = Simulator(net)
    start_time = time.time()
    print "starting simulation (Simulator)"
    sim.run(0.5)
    print "runtime: ", time.time() - start_time, "seconds"

if 1:
    sim2 = SimulatorOCL(net, profiling=True)
    start_time = time.time()
    print "starting simulation (OCL with profiling)"
    sim2.run(0.5)
    print "runtime: ", time.time() - start_time, "seconds"
    foo = [(t, n) for (n, t) in sim2.t_used.items()]
    foo.sort()
    foo.reverse()
    t_total = 0
    for t, n in foo:
        print t * 1e-9, n
        t_total += t * 1e-9
    print 'total time in OCL:', t_total

if 1:
    sim3 = SimulatorOCL(net, profiling=False)
    start_time = time.time()
    print "starting simulation (OCL)"
    sim3.run(0.5)
    print "runtime: ", time.time() - start_time, "seconds"
