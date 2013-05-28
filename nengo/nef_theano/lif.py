from _collections import OrderedDict

import numpy as np
import theano
from theano import tensor as TT

import neuron

class LIF_Op(theano.Op):
    def __init__(self, tau_ref, tau_rc, upsample=1):
        self.tau_ref = tau_ref
        self.tau_rc = tau_rc
        self.upsample = upsample

    def __hash__(self):
        return hash((type(self), self.tau_ref, self.tau_rc, self.upsample))

    def __eq__(self, other):
        return (type(self) == type(other)
                and self.tau_ref == other.tau_ref
                and self.tau_rc == other.tau_rc
                and self.upsample == other.upsample)

    def make_node(self, voltage, refractory_time, input_current, dt):
        orig_inputs = [voltage, refractory_time, input_current, dt]
        tsor_inputs = map(theano.tensor.as_tensor_variable, orig_inputs)

        new_voltage = voltage.type()
        new_refractory_time = refractory_time.type()
        spiked = voltage.type()  # XXX should be ints?
        outputs = [new_voltage, new_refractory_time, spiked]

        return theano.Apply(self, tsor_inputs, outputs)

    def perform(self, node, inputs, outstor):
        voltage, refractory_time, J, dt = inputs

        tau_rc = self.tau_rc
        tau_ref = self.tau_ref

        dt = dt / self.upsample
        spiked = (voltage != voltage)  # -- bool array of False

        for ii in xrange(self.upsample):

            dV = dt / tau_rc * (J - voltage)

            # increase the voltage, ignore values below 0
            v = np.maximum(voltage + dV, 0)

            # handle refractory period
            post_ref = 1.0 - (refractory_time - dt) / dt

            # set any post_ref elements < 0 = 0, and > 1 = 1
            v *= np.clip(post_ref, 0, 1)

            # determine which neurons spike
            # if v > 1 set spiked = 1, else 0
            spiked = np.bitwise_or(spiked, v > 1)

            # adjust refractory time (neurons that spike get
            # a new refractory time set, all others get it reduced by dt)

            # linearly approximate time since neuron crossed spike threshold
            overshoot = (v - 1) / dV
            spiketime = dt * (1.0 - overshoot)

            # adjust refractory time (neurons that spike get a new
            # refractory time set, all others get it reduced by dt)
            refractory_time = np.where(spiked,
                spiketime + tau_ref, refractory_time - dt)

            voltage = v * (1.0 - spiked.astype(v.dtype))

        outstor[0][0] = voltage.astype(node.outputs[0].dtype)
        outstor[1][0] = refractory_time.astype(node.outputs[1].dtype)
        outstor[2][0] = spiked.astype(node.outputs[2].dtype)


class LIFNeuron(neuron.Neuron):
    def __init__(self, size, tau_rc=0.02, tau_ref=0.002):
        """Constructor for a set of LIF rate neuron.

        :param int size: number of neurons in set
        :param float dt: timestep for neuron update function
        :param float tau_rc: the RC time constant
        :param float tau_ref: refractory period length (s)

        """
        neuron.Neuron.__init__(self, size)
        self.tau_rc = tau_rc
        self.tau_ref  = tau_ref
        self.voltage = theano.shared(
            np.zeros(size).astype('float32'), name='lif.voltage')
        self.refractory_time = theano.shared(
            np.zeros(size).astype('float32'), name='lif.refractory_time')

    #TODO: make this generic so it can be applied to any neuron model
    # (by running the neurons and finding their response function),
    # rather than this special-case implementation for LIF

    def make_alpha_bias(self, max_rates, intercepts):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons

        """
        x = 1.0 / (1 - TT.exp(
                (self.tau_ref - (1.0 / max_rates)) / self.tau_rc))
        alpha = (1 - x) / (intercepts - 1.0)
        j_bias = 1 - alpha * intercepts
        return alpha, j_bias

    # TODO: have a reset() function at the ensemble and network level
    #that would actually call this
    def reset(self):
        """Resets the state of the neuron."""
        neuron.Neuron.reset(self)

        self.voltage.set_value(np.zeros(self.size).astype('float32'))
        self.refractory_time.set_value(np.zeros(self.size).astype('float32'))

    def update(self, J, dt):
        """Theano update rule that implementing LIF rate neuron type
        Returns dictionary with voltage levels, refractory periods,
        and instantaneous spike raster of neurons.

        :param float array J:
            the input current for the current time step
        :param float dt: the timestep of the update
        """
        op = LIF_Op(tau_ref=self.tau_ref, tau_rc=self.tau_rc)
        new_v, new_rt, spiked = op(
                self.voltage, self.refractory_time,
                input_current=J, dt=dt)
        
        return OrderedDict([
            (self.voltage, new_v),
            (self.refractory_time, new_rt),
            (self.output, spiked)])

neuron.types['lif'] = LIFNeuron

