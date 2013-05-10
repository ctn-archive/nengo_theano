"""

TODO
----
* generate kernels for correct dtypes
* create meta-information dictionary to go with ocl_vars to at least mark constants
* double-buffer the simulator
* use int8 spikes
* use float16 for many things,
"""

import re

from _collections import OrderedDict
import theano
import numpy as np
import pyopencl as cl
import simulator
import lif
import probe

from ocl.array import Array, to_device, empty
from ocl.gemv_batched import plan_map_gemv
from ocl.elemwise import plan_copy
from ocl.dot import plan_dot
from ocl.plan import Plan

ocl_perform = {}
ocl_alloc = {}


# -- decorator to register an OCL "perform" method for Theano Op `op_cls`
def perform(op_cls):
    def deco(f):
        ocl_perform[op_cls] = f
        return f
    return deco


# -- decorator to register an OCL "alloc" method for Theano Op `op_cls`
def alloc(op_cls):
    def deco(f):
        ocl_alloc[op_cls] = f
        return f
    return deco


class UnAllocatedOutput(object):
    """Singleton to stand for an un-allocated output """


class SimulatorOCL(object):
    """
    Simulator that uses OpenCL instead of numpy to evaluate the Theano "step"
    function and run the simulator for the network.

    This class draws on alternate implementations for the Ops in the step
    function. It
    """
    def __init__(self, network, context=None, profiling=False):
        self.network = network
        if self.network.tick_nodes:
            raise ValueError('Simulator does not support',
                             ' networks with tick_nodes')
        if context is None:
            context = cl.create_some_context()
        self.context = context
        self.profiling = profiling
        if profiling:
            self.queue = cl.CommandQueue(context,
                properties=cl.command_queue_properties.PROFILING_ENABLE)
            self.t_used = {}
        else:
            self.queue = cl.CommandQueue(context)

        # dictionary for all variables
        # and the theano description of how to compute them 
        updates = OrderedDict()

        # for every node in the network
        for node in self.network.nodes.values():
            # if there is some variable to update
            if hasattr(node, 'update'):
                # add it to the list of variables to update every time step
                updates.update(node.update(self.network.dt))

        # create graph and return optimized update function
        # -- use py linker to avoid wasting time compiling C code
        updates_items = updates.items()
        self.step = theano.function([simulator.simulation_time], [],
            updates=updates_items, 
            mode=theano.Mode(
                optimizer='default',
                linker=theano.gof.vm.VM_Linker(use_cloop=False, allow_gc=False),
                ))

        # -- replace the theano function with a new list of thunks
        self.nodes = self.step.fn.nodes

        # -- allocate two plans and vars for double-buffered updates
        self.n_steps = -1
        self._plans = ([], [])
        self._ocl_vars = ({}, {})
        self.constant_vars = {}
        self._node_plans = ({}, {})
        self._probed_vars = ({}, {})
        self._probed_vals = ({}, {})
        self._simtime = [
            cl.array.zeros(self.queue, (), dtype='float32'),
            cl.array.zeros(self.queue, (), dtype='float32'),
        ]


        # -- allocate workspace for the first plan (plan 0)
        self.ocl_vars = self._ocl_vars[0]
        for node in self.nodes:
            for vv in node.inputs:
                if vv in self._ocl_vars[0]:
                    continue
                if vv in self.constant_vars:
                    continue
                if hasattr(vv, 'data'):
                    self.constant_vars[vv] = vv.data
                elif vv.name == 'simulation_time':
                    self._ocl_vars[0][vv] = self._simtime[0]
                elif vv.owner is None:
                    val = vv.get_value(borrow=True)
                    self._ocl_vars[0][vv] = to_device(self.queue, val)
                    self.queue.finish()
            ocl_alloc[type(node.op)](self.queue, self, node)
            for vout in node.outputs:
                if vout in self._ocl_vars[0]:
                    if self._ocl_vars[0][vout] is not UnAllocatedOutput:
                        assert self._ocl_vars[0][vout].ndim == vout.ndim, node.op
                        assert self._ocl_vars[0][vout].dtype == vout.dtype, node.op
                else:
                    assert vout in self.constant_vars

        # -- set up the outputs from plan 0 as the inputs for plan 1
        # -- and the outputs from plan 1 as the inputs for plan 0
        for (ivar, ovar) in self.step.fn.updated_vars.items():
            if ovar not in self._ocl_vars[0]:
                # -- this can happen if `ovar` is not
                #    implicated in any computation except
                #    the updating of this variable
                #    and perhaps being updated itself.
                assert ovar.owner is None
                if hasattr(vv, 'data'):
                    self.constant_vars[ovar] = ovar.data
                    raise NotImplementedError('copy const into ocl ivar')
                else:
                    val = ovar.get_value(borrow=True)
                    self._ocl_vars[0][ovar] = to_device(self.queue, val)
                    self.queue.finish()
            self._ocl_vars[1][ivar] = self._ocl_vars[0][ovar]
            if ivar not in self._ocl_vars[0]:
                # -- if ivar is not an input to anything, then this is the
                #    first time we've seen it
                val = ivar.get_value(borrow=True)
                self._ocl_vars[0][ivar] = to_device(self.queue, val)
                self.queue.finish()

            self._ocl_vars[1][ovar] = self._ocl_vars[0][ivar]

        # -- allocate workspace for the second plan (plan 1)
        self.ocl_vars = self._ocl_vars[1]
        for node in self.nodes:
            for vv in node.inputs:
                if vv in self._ocl_vars[1]:
                    continue
                if vv in self.constant_vars:
                    continue
                assert not hasattr(vv, 'data')
                if vv.name == 'simulation_time':
                    self._ocl_vars[1][vv] = self._simtime[1]
                elif vv.owner is None:
                    # -- vv is a shared var that isn't updated
                    self._ocl_vars[1][vv] = self._ocl_vars[0][vv]
            if any(vv not in self.ocl_vars for vv in node.outputs):
                ocl_alloc[type(node.op)](self.queue, self, node)
                for vout in node.outputs:
                    if vout in self._ocl_vars[1]:
                        assert self._ocl_vars[1][vout].ndim == vout.ndim, node.op
                        assert self._ocl_vars[1][vout].dtype == vout.dtype, node.op
                    else:
                        assert vout in self.constant_vars
        del self.ocl_vars


        # -- build plans for evaluating ocl_vals[0]
        for node in self.nodes:
            self.ocl_vars = self._ocl_vars[0]
            self.probed_vars = self._probed_vars[0]
            self.probed_vals = self._probed_vals[0]
            plans = ocl_perform[type(node.op)](self.queue, self, node)
            for plan in plans:
                plan.node = node
            self._plans[0].extend(plans)
            self._node_plans[0][node] = plans

        # -- build plans for evaluating ocl_vals[0]
        for node in self.nodes:
            self.ocl_vars = self._ocl_vars[1]
            self.ocl_vars = self._ocl_vars[1]
            self.probed_vars = self._probed_vars[1]
            self.probed_vals = self._probed_vals[1]
            plans = ocl_perform[type(node.op)](self.queue, self, node)
            for plan in plans:
                plan.node = node
            self._plans[1].extend(plans)
            self._node_plans[1][node] = plans
        del self.ocl_vars
        del self.probed_vars
        del self.probed_vals
        self.queue.finish()


    def copy_to_shared(self):
        """
        Copy data from the theano graph's shared variables into self.ocl_vars
        """

        ocl_vars = self._ocl_vars[self.n_steps % 2]
        probed_vals = self._probed_vals[self.n_steps % 2]
        for (ivar, ovar) in self.step.fn.updated_vars.items():
            if ocl_vars[ivar] is UnAllocatedOutput:
                if ovar.owner and isinstance(ovar.owner.op, probe.Scribe):
                    x = ovar.owner.inputs[0]
                    last_update, vals = probed_vals[x]
                    ovar.owner.inputs[1].set_value(vals)
                    ovar.owner.inputs[2].set_value(len(vals))
            else:
                try:
                    nparray = ocl_vars[ivar].get(self.queue)
                except AssertionError:
                    print ocl_vars[ivar].structure
                    raise
                ivar.set_value(nparray, borrow=True)
        self.queue.finish()

    def copy_from_shared(self):
        """
        Copy data from self.ocl_vars into the theano graph's shared variables
        """
        ocl_vars = self._ocl_vars[self.n_steps % 2]
        for (ivar, ovar) in self.step.fn.updated_vars.items():
            nparray = ivar.get_value(borrow=True)
            assert nparray.dtype == ocl_vars[ivar].dtype
            assert nparray.shape == tuple(ocl_vars[ivar].shape)
            ocl_vars[ivar].set(nparray, self.queue)
        self.queue.finish()

    def run_steps(self, N, sync_w_theano_shared_vars=True,
                  run_theano_too=False):
        """
        Run the simulator for N steps of duration `self.dt`
        """
        if sync_w_theano_shared_vars and self.n_steps > -1:
            self.copy_from_shared()

        if 0:
          for ivar in self._ocl_vars[0]:
            print ivar, self._ocl_vars[0][ivar].get().max(),
            print self._ocl_vars[1][ivar].get().max()
            if self.step.fn.storage_map[ivar][0] is not None:
                print '  ---> ', self.step.fn.storage_map[ivar][0].max()


        if run_theano_too:
            storage_map = self.step.fn.storage_map
            for i in xrange(N):
                self.n_steps += 1
                ocl_vars = self._ocl_vars[self.n_steps % 2]
                node_plans = self._node_plans[self.n_steps % 2]
                for jj, (node, thunk) in enumerate(
                        zip(self.step.fn.nodes, self.step.fn.thunks)):

                    def any_inaccuracy(msg, theano_vars):
                        inaccuracy = False
                        seen = set()
                        for ovar in theano_vars:
                            if ovar in seen:
                                continue
                            if ovar in ocl_vars:
                                refval = storage_map[ovar][0]
                                try:
                                    oclval = ocl_vars[ovar].get()
                                except (AssertionError, ValueError):
                                    print ocl_vars[ovar].structure
                                    raise
                                assert refval.dtype == oclval.dtype
                                assert refval.shape == oclval.shape
                                if not np.allclose(refval, oclval,
                                                   atol=1e-4, rtol=1e-4):
                                    print msg, self.n_steps, 'Node', node, 'messed up', ovar
                                    print '  stats', refval.max(), refval.min(), refval.mean(),
                                    print 'vs', oclval.max(), oclval.min(), oclval.mean()
                                    print '  diff abs', np.max(abs(refval - oclval)),
                                    print 'rel', np.max(abs(refval - oclval) / abs(refval + oclval + 1e-12))
                                    inaccuracy=True
                            seen.add(ovar)
                        return inaccuracy

                    if any_inaccuracy('pre', node.inputs):
                        raise RuntimeError('Inaccurate computations')

                    # -- run the theano thunk
                    thunk()

                    # -- run the ocl equivalent
                    for p in node_plans[node]:
                        p._fn(*p._fn_args)
                    self.queue.finish()

                    vars_that_should_match = node.inputs + node.outputs
                    for opos, iposlist in getattr(node.op, 'destroy_map', {}).items():
                        for ipos in reversed(sorted(iposlist)):
                            vars_that_should_match.pop(ipos)
                    if any_inaccuracy('post', vars_that_should_match):
                        print node_plans[node]
                        print p._fn
                        print p.text
                        print p._fn_args
                        print vars_that_should_match
                        print [ocl_vars[vv].data for vv in node.inputs if vv in ocl_vars]
                        print [ocl_vars[vv].data for vv in node.outputs]
                        raise RuntimeError('Inaccurate computations')
                    else:
                        print '.',
                print 'done pass', self.n_steps
                # -- apply updates to theano fn
                for (ivar, ovar) in self.step.fn.updated_vars.items():
                    storage_map[ivar][0] = storage_map[ovar][0]

        else:
            if self.profiling:
                for i in xrange(N):
                    self.n_steps += 1
                    evs = []
                    plans = self._plans[self.n_steps % 2]
                    for p in plans:
                        evs.append(p._fn(*p._fn_args))
                    self.queue.finish()
                    assert len(evs) == len(plans)
                    for e, p in zip(evs, plans):
                        self.t_used.setdefault(p.node, 0)
                        self.t_used[p.node] +=  e.profile.end - e.profile.start
            else:
                N2 = N // 2
                A = (self.n_steps + 1) % 2
                B = (self.n_steps + 2) % 2
                plans = self._plans[A] + self._plans[B]
                queues = [p._enqueue_args[0] for p in plans]
                kerns = [p._enqueue_args[1] for p in plans]
                gsize = [p._enqueue_args[2] for p in plans]
                lsize = [p._enqueue_args[3] for p in plans]
                dt = self.network.dt
                try:
                    for i in xrange(N2):
                        simtime = (self.n_steps + i) * dt
                        self._simtime[A] = simtime
                        self._simtime[B] = simtime + dt 
                        map(cl.enqueue_nd_range_kernel,
                            queues, kerns, gsize, lsize)
                        for vv, scribe_dt in self._probed_vars[A].items():
                            last_update, vals = self._probed_vals[A][vv]
                            if simtime - last_update > scribe_dt:
                                vals.append(self._ocl_vars[A][vv].get())
                                last_update = simtime
                            self._probed_vals[A][vv] = last_update, vals
                        for vv, scribe_dt in self._probed_vars[B].items():
                            last_update, vals = self._probed_vals[B][vv]
                            if simtime - last_update > scribe_dt:
                                vals.append(self._ocl_vars[B][vv].get())
                                last_update = simtime
                            self._probed_vals[B][vv] = last_update, vals
                    self.n_steps += N2 * 2
                    if N % 2:
                        raise NotImplementedError()
                except Exception, e:
                    e.args = e.args + ({'plan': p, 'node': p.node},)
                    raise

        if sync_w_theano_shared_vars:
            self.copy_to_shared()  # -- calls queue.finish
        else:
            self.queue.finish()

    def run(self, approx_sim_time, **kwargs):
        """
        Run the simulator for a number of steps that advances the total
        simulation time by approximately `approx_sim_time`
        """
        n_steps = int(approx_sim_time / self.network.dt)
        return self.run_steps(n_steps, **kwargs)


@alloc(simulator.MapGemv)
def ocl_map_gemv_a(queue, sim, node):
    # TODO: work in-place on Y_in if node.destroy_map is set
    try:
        Y = sim.ocl_vars[node.inputs[-1]]
    except KeyError:
        Y = sim.constant_vars[node.inputs[-1]]
    sim.ocl_vars[node.outputs[0]] = empty(queue, Y.shape, Y.dtype)


@perform(simulator.MapGemv)
def ocl_map_gemv_p(queue, sim, node):
    alpha, A, X, beta, Y_in = [sim.ocl_vars.get(vi) for vi in node.inputs]
    Y_out, = [sim.ocl_vars[vi] for vi in node.outputs]

    # XXX: following depends on constants alpha, beta
    falpha = float(node.inputs[0].data)
    fbeta = float(node.inputs[3].data)

    B, M, N = A.shape
    Bx, Nx = X.shape
    By, My = Y_out.shape
    assert Bx == By == B
    assert My == M
    assert Nx == N

    A_offsets = to_device(queue, np.arange(B) * M * N )
    X_offsets = to_device(queue, np.arange(B) * N )
    Y_offsets = to_device(queue, np.arange(B) * M)

    if Y_in is None:
        Y_in_val = sim.constant_vars[node.inputs[-1]]
        if np.all(Y_in_val == 0):
            fbeta = 0
        elif fbeta != 0:
            Y_in = to_device(queue, np.asarray(Y_in_val * fbeta))
            fbeta = 1

    return [plan_map_gemv(queue, falpha, A, X, fbeta, Y_out, Y_in)]


@alloc(theano.tensor.elemwise.DimShuffle)
def dimshuffle_a(queue, sim, node):
    # -- set up a view of X
    # XXX make sure that dimshuffle is inplace
    Xvar, = node.inputs
    X = sim.ocl_vars[Xvar]
    Yvar, = node.outputs

    Yshape = list(X.shape)
    Ystrides = list(X.strides)

    # -- drop
    for drop in reversed(node.op.drop):
        Yshape.pop(drop)
        Ystrides.pop(drop)

    # -- transpose
    Yshape = [Yshape[i] for i in node.op.shuffle]
    Ystrides = [Ystrides[i] for i in node.op.shuffle]

    # -- augment
    for augm in node.op.augment:
        Yshape.insert(augm, 1)
        Ystrides.insert(augm, X.dtype.itemsize)

    Y = Array(queue, data=X.data, dtype=X.dtype,
              shape=Yshape, strides=Ystrides)
    sim.ocl_vars[Yvar] = Y


@perform(theano.tensor.elemwise.DimShuffle)
def dimshuffle_p(queue, sim, node):
    return []


@alloc(theano.tensor.opt.Shape_i)
def shape_i_a(queue, sim, node):
    X = sim.ocl_vars[node.inputs[0]]
    sim.constant_vars[node.outputs[0]] = X.shape[node.op.i]

@perform(theano.tensor.opt.Shape_i)
def shape_i_p(queue, sim, node):
    return []

@alloc(theano.tensor.opt.MakeVector)
def make_vector_a(queue, sim, node):
    try:
        inputs = [sim.constant_vars[vv] for  vv in node.inputs]
        sim.constant_vars[node.outputs[0]] = np.asarray(inputs)
    except KeyError:
        theano.printing.debugprint(node.outputs)
        raise NotImplementedError('non-constant MakeVector')

@perform(theano.tensor.opt.MakeVector)
def make_vector_p(queue, sim, node):
    return []


@alloc(theano.tensor.basic.Reshape)
def reshape_a(queue, sim, node):
    X, shp = node.inputs
    Xval = sim.ocl_vars[X]
    # -- N.B. we only support fixed shapes currently
    shape_val = sim.constant_vars[shp]
    try:
        shape_val = [int(shape_val)]
    except:
        pass

    assert len(shape_val) == node.outputs[0].ndim
    assert node.outputs[0].dtype == node.inputs[0].dtype

    if np.prod(Xval.shape) == np.prod(shape_val) == 1:
        Yval = Array(queue, data=Xval.data, dtype=Xval.dtype,
                shape=list(shape_val),
                strides=[Xval.dtype.itemsize] * len(shape_val))
    else:
        theano.printing.debugprint(node.outputs)
        print 'X stats', Xval.shape, Xval.strides
        print 'target shape', shape_val
        raise NotImplementedError('MakeVector')
    sim.ocl_vars[node.outputs[0]] = Yval

@perform(theano.tensor.basic.Reshape)
def reshape_p(queue, sim, node):
    return []


def flatten_c_contig(Xval):
    # currently this is a little different from the c contiguous
    # flag logic in array.Array, so we redo it here
    need_stride = Xval.dtype.itemsize
    c_contig = True
    for si, ri in reversed(zip(Xval.shape, Xval.strides)):
        if si == 1:
            continue
        else:
            if ri == need_stride:
                need_stride *= si
            else:
                c_contig = False
    return c_contig


@alloc(theano.tensor.basic.Flatten)
def flatten_a(queue, sim, node):
    X,= node.inputs
    Xval = sim.ocl_vars[X]
    if flatten_c_contig(Xval):
        Yval = Array(queue, data=Xval.data, dtype=Xval.dtype,
                shape=[int(np.prod(Xval.shape))],
                strides=[Xval.dtype.itemsize])
    else:
        raise NotImplementedError()
    sim.ocl_vars[node.outputs[0]] = Yval

@perform(theano.tensor.basic.Flatten)
def flatten_p(queue, sim, node):
    X, = node.inputs
    Xval = sim.ocl_vars[X]
    Y, = node.outputs
    Yval = sim.ocl_vars[Y]
    if Xval.data is Yval.data:
        return []
    elif flatten_c_contig(Xval) and flatten_c_contig(Yval):
        Xtype = Xval.ocldtype
        Ytype = Yval.ocldtype
        _fn = cl.Program(queue.context, """
            __kernel void foo(
                __global const %(Xtype)s *X,
                __global %(Ytype)s *Y)
            {
                int ii = get_global_id(0);
                Y[ii] = X[ii];
            }
            """ % locals()).build().foo
        _fn_args = (queue, (Xval.size,), None, Xval.data, Yval.data)
        return [Plan(locals())]

    else:
        print Xval.shape, Yval.shape
        raise NotImplementedError('Flatten')

@alloc(theano.tensor.basic.Dot)
def dot_a(queue, sim, node):
    X, Y = node.inputs
    Z, = node.outputs

    Xval = sim.constant_vars[X]
    Yval = sim.ocl_vars[Y]

    if Xval.ndim == 0 and Yval.ndim == 2:
        assert X.ndim == 2 # XXX such garbage here, clean it up.
        Zshape = list(Yval.shape)
        Zdtype = np.dtype(node.outputs[0].dtype)
        Zstrides = [Yval.shape[1] * Zdtype.itemsize, Zdtype.itemsize]
    elif Xval.ndim == 2 and Yval.ndim == 2:
        Zshape = [Xval.shape[0], Yval.shape[1]]
        Zdtype = np.dtype(node.outputs[0].dtype)
        Zstrides = [Yval.shape[1] * Zdtype.itemsize, Zdtype.itemsize]
    else:
        raise NotImplementedError('dot with shapes', (Xval.shape, Yval.shape))

    size = Zstrides[0] * Zshape[0]
    Zdata = cl.Buffer(queue.context,
                      flags=cl.mem_flags.READ_WRITE,
                      size=int(size))
    Zval = Array(queue, data=Zdata, dtype=Zdtype,
                 shape=Zshape, strides=Zstrides)
    sim.ocl_vars[Z] = Zval

@perform(theano.tensor.basic.Dot)
def dot_p(queue, sim, node):
    X, Y = node.inputs
    Z, = node.outputs

    if X.ndim == 2 and Y.ndim == 2:
        if X in sim.constant_vars:
            # -- will fail for non 1x1 arrays
            X = float(sim.constant_vars[X])

            Yval = sim.ocl_vars[Y]
            Zval = sim.ocl_vars[Z]
            assert Yval.shape[0] == 1

            Ys0, Ys1 = Yval.itemstrides
            Zs0, Zs1 = Zval.itemstrides
            Ytype = Yval.ocldtype
            Ztype = Zval.ocldtype

            sumtype = Ztype # TODO: consider more precision here

            _fn = cl.Program(queue.context, """
                __kernel void foo(
                    __global const %(Ytype)s *Y,
                    __global %(Ztype)s *Z)
                {
                    int ii = get_global_id(0);
                    int jj = get_global_id(1);
                    Z[ii * %(Zs0)s + jj * %(Zs1)s] =
                        %(X)s * Y[ii * %(Ys0)s + jj * %(Ys1)s];
                }
                """ % locals()).build().foo

            _fn_args = (queue, Zval.shape, None, Yval.data, Zval.data)
        else:
            return plan_dot(queue,
                sim.ocl_vars[X], sim.ocl_vars[Y], sim.ocl_vars[Z])
    else:
        raise NotImplementedError('dot with shapes', (Xval.shape, Yval.shape))

    return [Plan(locals())]


@alloc(lif.LIF_Op)
def lif_a(queue, sim, node):
    v, rt, ic, dt = node.inputs
    dt = float(dt.value)

    nv = sim.ocl_vars[v].empty_like()
    nrt = sim.ocl_vars[rt].empty_like()
    # TDOO: use int8 for spikes
    spiked = empty(queue, nv.shape, dtype=np.float32)

    sim.ocl_vars[node.outputs[0]] = nv
    sim.ocl_vars[node.outputs[1]] = nrt
    sim.ocl_vars[node.outputs[2]] = spiked


@perform(lif.LIF_Op)
def lif_p(queue, sim, node):
    _v, _rt, _ic, _dt = node.inputs
    _ov, _ort, _os = node.outputs

    J = sim.ocl_vars[_ic]
    V = sim.ocl_vars[_v]
    RT = sim.ocl_vars[_rt]
    OV = sim.ocl_vars[_ov]
    ORT = sim.ocl_vars[_ort]
    OS = sim.ocl_vars[_os]
    
    dt = float(_dt.value)
    tau_rc = node.op.tau_rc
    tau_ref  = node.op.tau_ref
    V_threshold = 1.0
    tau_rc_inv = 1.0 / tau_rc

    upsample = node.op.upsample
    upsample_dt = dt / upsample
    upsample_dt_inv = 1.0 / upsample_dt

    Jtype = J.ocldtype
    Vtype = V.ocldtype
    RTtype = RT.ocldtype
    OStype = OS.ocldtype

    _fn = cl.Program(queue.context, """
        __kernel void foo(
            __global const %(Jtype)s *J,
            __global const %(Vtype)s *voltage,
            __global const %(RTtype)s *refractory_time,
            __global %(Vtype)s *out_voltage,
            __global %(RTtype)s *out_refractory_time,
            __global %(OStype)s *out_spiked
                     )
        {
            const %(RTtype)s dt = %(upsample_dt)s;
            const %(RTtype)s dt_inv = %(upsample_dt_inv)s;
            const %(RTtype)s tau_ref = %(tau_ref)s;
            const %(Vtype)s tau_rc_inv = %(tau_rc_inv)s;
            const %(Vtype)s V_threshold = %(V_threshold)s;

            const int gid = get_global_id(0);
            %(Vtype)s v = voltage[gid];
            %(RTtype)s rt = refractory_time[gid];
            %(Jtype)s input_current = J[gid];
            int spiked = 0;

            for (int ii = 0; ii < %(upsample)s; ++ii)
            {
              %(Vtype)s dV = dt * tau_rc_inv * (input_current - v);
              %(RTtype)s post_ref = 1.0 - (rt - dt) * dt_inv;
              v += dV;
              v = v > 0 ?
                  v * (post_ref < 0 ? 0.0 : post_ref < 1 ? post_ref : 1.0)
                  : 0;
              const int spiked_ii = v > V_threshold;
              %(Vtype)s overshoot = (v - V_threshold) / dV;
              %(RTtype)s spiketime = dt * (1.0 - overshoot);

              if (spiked_ii)
              {
                v = 0.0;
                rt = spiketime + tau_ref;
                spiked = 1;
              }
              else
              {
                rt -= dt;
              }
            }

            out_voltage[gid] = v;
            out_refractory_time[gid] = rt;
            out_spiked[gid] = spiked ? (%(OStype)s) 1 : (%(OStype)s) 0;
        }
        """ % locals()).build().foo
    v = sim.ocl_vars[_v]

    _fn_args = (queue, (v.size,), None,
        sim.ocl_vars[_ic].data,
        sim.ocl_vars[_v].data,
        sim.ocl_vars[_rt].data,
        sim.ocl_vars[_ov].data,
        sim.ocl_vars[_ort].data,
        sim.ocl_vars[_os].data,
        )
    return [Plan(locals())]


@alloc(theano.tensor.elemwise.Elemwise)
def elemwise_a(queue, sim, node):
    ocl_inputs = [sim.ocl_vars.get(vv) for vv in node.inputs]
    const_inputs = [sim.constant_vars.get(vv) for vv in node.inputs]
    for vv in node.outputs:
        shape = np.asarray([1] * vv.ndim)
        for vi in ocl_inputs:
            if vi is not None:
                assert len(shape) == len(vi.shape), (shape, vi.shape)
                shape = list(np.maximum(shape, vi.shape))
        for vi in const_inputs:
            if vi is not None and hasattr(vi, 'shape'):
                assert len(shape) == len(vi.shape)
                shape = list(np.maximum(shape, vi.shape))

        sim.ocl_vars[vv] = empty(queue,
                list(shape), np.dtype(vv.dtype))
        print shape

@perform(theano.tensor.elemwise.Elemwise)
def elemwise_p(queue, sim, node):
    if len(node.outputs) > 1:
        raise NotImplementedError()
    if node.outputs[0].ndim > 3:
        raise NotImplementedError()

    for ovar in node.outputs:
        for ivar in node.inputs:
            if ivar in sim.ocl_vars:
                assert sim.ocl_vars[ivar].data is not sim.ocl_vars[ovar].data

    c_body_inputs = {}
    for inum, ivar in enumerate(node.inputs + node.outputs):
        if ivar in sim.ocl_vars:
            varname = 'I%i' % inum
            indexes = []
            for jj in range(ivar.ndim):
                indexes.append('gid%i * I%i_s%i' % (jj, inum, jj))
            c_body_inputs[ivar] = '%s[%s]' % (varname, ' + '.join(indexes))
        else:
            c_body_inputs[ivar] = str(float(sim.constant_vars[ivar]))

    scalar_inputs = [theano.scalar.Scalar(dtype=vv.dtype)()
                     for vv in node.inputs]
    ctype_body = node.op.scalar_op.c_code(
        node.op.scalar_op(*scalar_inputs).owner,
        'name',
        [c_body_inputs[vv] for vv in node.inputs],
        [c_body_inputs[vv] for vv in node.outputs],
        {})

    # -- replace the numpy typedefs
    ctype_body = re.sub('npy_float64', 'double', ctype_body)
    ctype_body = re.sub('npy_float32', 'float', ctype_body)

    for inum, ivar in enumerate(node.inputs + node.outputs):
        if ivar in sim.ocl_vars:
            ival = sim.ocl_vars[ivar]
            for jj, sj in enumerate(ival.itemstrides):
                if ival.shape[jj] == 1:
                    sj = 0
                ctype_body = re.sub('I%i_s%i' % (inum, jj), str(sj),
                                    ctype_body)


    params = ['__global %s %s * I%s' % (
            'const' if ivar in node.inputs else '',
            sim.ocl_vars[ivar].ocldtype,
            inum)
        for inum, ivar in enumerate(node.inputs + node.outputs)
        if ivar in sim.ocl_vars]

    joined_params = ', '.join(params)

    text = """
        __kernel void foo(
            %(joined_params)s
                     )
        {
            const int gid0 = get_global_id(0);
            const int gid1 = get_global_id(1);
            const int gid2 = get_global_id(2);

            %(ctype_body)s
        }
        """ % locals()

    _fn = cl.Program(queue.context, text).build().foo
    _fn_args = (queue, sim.ocl_vars[node.outputs[0]].shape, None,)
    _fn_args = _fn_args + tuple([sim.ocl_vars[ivar].data
        for inum, ivar in enumerate(node.inputs + node.outputs)
        if ivar in sim.ocl_vars])
    return [Plan(locals())]

@alloc(theano.tensor.Rebroadcast)
def rebroadcast_a(queue, sim, node):
    sim.ocl_vars[node.outputs[0]] = sim.ocl_vars[node.inputs[0]]

@perform(theano.tensor.Rebroadcast)
def rebroadcast_p(queue, sim, node):
    return []


@alloc(probe.Scribe)
def scribe_a(queue, sim, node):
    x, buf, i, t, dt_sample = node.inputs
    #new_buf = sim.ocl_vars[buf].empty_like()
    new_i = sim.ocl_vars[i].empty_like()
    assert len(node.outputs) == 2
    sim.ocl_vars[node.outputs[0]] = UnAllocatedOutput
    sim.ocl_vars[node.outputs[1]] = UnAllocatedOutput

@perform(probe.Scribe)
def scribe_p(queue, sim, node):
    # Scribes are handled specially by the simulator
    # because generally, the simulator does not permit the size of Array
    # objects to change during simulation.
    # Scribe ops are a necessary violation of that general rule.

    # XXX Schedule the probes to run less often than the main simulator
    #     clock, if the dt_sample is sufficiently large
    dt_sample = float(sim.constant_vars[node.inputs[-1]])
    sim.probed_vars[node.inputs[0]] = dt_sample
    sim.probed_vals[node.inputs[0]] = [0, []]
    return []

