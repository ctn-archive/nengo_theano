"""This is a file to test the probe class, and it's ability to record data
and write to file.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time

from .. import nef_theano as nef

def test_probe(simcls=None, show=False):

    build_time_start = time.time()

    net = nef.Network('Probe Test')
    net.make_input('in', math.sin)
    net.make('A', 50, 1)
    net.make('B', 5, 1)

    net.connect('in', 'A')
    net.connect('in', 'B')

    timesteps = 100
    dt_step = 0.01
    t = np.linspace(dt_step, timesteps*dt_step, timesteps)
    pstc = 0.01

    Ip = net.make_probe('in', dt_sample=dt_step, pstc=pstc)
    Ap = net.make_probe('A', dt_sample=dt_step, pstc=pstc)
    Bp = net.make_probe('B', dt_sample=dt_step)
    BpSpikes = net.make_probe('B', data_type='spikes', dt_sample=dt_step)

    build_time_end = time.time()

    print "Starting simulation"
    if simcls is None:
        net.run(timesteps * dt_step)
    else:
        sim = simcls(net)
        sim.run(timesteps * dt_step)

    sim_time_end = time.time()
    print "\nBuild time: %0.10fs" % (build_time_end - build_time_start)
    print "Sim time: %0.10fs" % (sim_time_end - build_time_end)

    assert Ip.get_data().shape == (100, 1), Ip.get_data().shape
    assert Ap.get_data().shape == (100, 1)
    assert Bp.get_data().shape == (100, 1)
    assert BpSpikes.get_data().shape == (100, 1, 5)

    ip_mse = np.mean((Ip.get_data() - np.sin(t)) ** 2)
    print 'Ip MSE', ip_mse
    assert ip_mse < 0.15  # getting .124 May 9 2013

    ap_mse = np.mean((Ap.get_data() - np.sin(t)) ** 2)
    print 'Ap MSE', ap_mse
    assert ap_mse < 0.15  # getting .127 May 9 2013

    plt.ioff(); plt.close(); 
    plt.subplot(3,1,1)
    plt.plot(Ip.get_data(), 'x'); plt.title('Input')
    plt.plot(np.sin(t))
    plt.subplot(3,1,2)
    plt.plot(Ap.get_data()); plt.title('A')
    plt.plot(np.sin(t))
    plt.subplot(3,1,3); plt.hold(1)
    plt.plot(Bp.get_data())
    for row in BpSpikes.get_data().T: 
        plt.plot(row[0]); 
    plt.title('B')
    if show:
        plt.show()
    else:
        plt.close()

def test_probe_sim():
    from nengo.nef_theano import simulator
    test_probe(simulator.Simulator)

def test_probe_simocl():
    from nengo.nef_theano import simulator_ocl
    test_probe(simulator_ocl.SimulatorOCL)
