"""

TODO
----
* generate kernels for correct dtypes
* create meta-information dictionary to go with ocl_vars to at least mark constants
* double-buffer the simulator
* use int8 spikes
* use float16 for many things,
"""

from _collections import OrderedDict
import theano
import numpy as np
import pyopencl as cl
import simulator
import lif


from ocl.array import Array, to_device, empty
from ocl.gemv_batched import plan_map_gemv
from ocl.elemwise import plan_copy
from ocl.dot import plan_dot
from ocl.plan import Plan


ocl_perform = {}
ocl_alloc = {}


def perform(op_cls):
    def deco(f):
        ocl_perform[op_cls] = f
        return f
    return deco


def alloc(op_cls):
    def deco(f):
        ocl_alloc[op_cls] = f
        return f
    return deco


class SimulatorOCL(object):
    def __init__(self, network, context=None, profiling=False):
        self.network = network
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
        self.step = theano.function([], [], updates=updates.items())

        self.order = self.step.maker.fgraph.toposort()

        self.plans = []
        self.ocl_vars = {}
        self.constant_vars = {}

        for node in self.order:
            for vv in node.inputs:
                if vv in self.ocl_vars:
                    continue
                if vv in self.constant_vars:
                    continue
                if hasattr(vv, 'data'):
                    self.constant_vars[vv] = vv.data
                elif vv.owner is None:
                    val = vv.get_value(borrow=True)
                    # TODO: optional casting logic?
                    self.ocl_vars[vv] = to_device(self.queue, val)
            ocl_alloc[type(node.op)](self.queue, self, node)
            for vout in node.outputs:
                if vout in self.ocl_vars:
                    assert self.ocl_vars[vout].ndim == vout.ndim, node.op
                    assert self.ocl_vars[vout].dtype == vout.dtype, node.op
            plans = ocl_perform[type(node.op)](self.queue, self, node)
            for plan in plans:
                plan.node = node
            self.plans.extend(plans)

        # XXX create another program that does the update
        # in the other direction (double-buffer the ocl_vars)

    def run_steps(self, N):
        # -- copy from shared variable inputs into internal state dict
        # -- run N steps
        # -- copy from state to shared variables
        fns = [p._fn for p in self.plans]
        args = [p._fn_args for p in self.plans]
        fns_args_plans = zip(fns, args, self.plans)
        if self.profiling:
            for i in xrange(N):
                evs = []
                try:
                    for fn, arg, p in fns_args_plans:
                        evs.append(fn(*arg))
                except Exception, e:
                    e.args = e.args + ({'plan': p, 'node': node},)
                    raise
                self.queue.finish()
                assert len(evs) == len(self.plans)
                for e, p in zip(evs, self.plans):
                    self.t_used.setdefault(p.node, 0)
                    self.t_used[p.node] +=  e.profile.end - e.profile.start
        else:
            for i in xrange(N):
                try:
                    for fn, arg, p in fns_args_plans:
                        fn(*arg)
                except Exception, e:
                    e.args = e.args + ({'plan': p, 'node': node},)
                    raise
            self.queue.finish()


    def run(self, approx_sim_time):
        n_steps = int(approx_sim_time / self.network.dt)
        return self.run_steps(n_steps)



@alloc(simulator.MapGemv)
def ocl_map_gemv_a(queue, sim, node):
    try:
        Y = sim.ocl_vars[node.inputs[-1]]
    except KeyError:
        Y = sim.constant_vars[node.inputs[-1]]
    sim.ocl_vars[node.outputs[0]] = empty(queue.context, Y.shape, Y.dtype)


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
        Ystrides.insert(augm, 0)

    Y = Array(X.data, X.dtype, Yshape, Ystrides)
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
        Yval = Array(Xval.data, dtype=Xval.dtype,
                shape=list(shape_val),
                strides=[0] * len(shape_val))
    else:
        theano.printing.debugprint(node.outputs)
        print 'X stats', Xval.shape, Xval.strides
        print 'target shape', shape_val
        raise NotImplementedError('MakeVector')
    sim.ocl_vars[node.outputs[0]] = Yval

@perform(theano.tensor.basic.Reshape)
def reshape_p(queue, sim, node):
    return []

@alloc(theano.tensor.basic.Flatten)
def flatten_a(queue, sim, node):
    X,= node.inputs
    Xval = sim.ocl_vars[X]
    # XXX verify that strides match is correct
    Yval = Array(Xval.data, Xval.dtype,
            shape=[int(np.prod(Xval.shape))],
            strides=[0])
    sim.ocl_vars[node.outputs[0]] = Yval

@perform(theano.tensor.basic.Flatten)
def flatten_p(queue, sim, node):
    return []

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
    Zval = Array(Zdata, Zdtype, Zshape, Zstrides)
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
    spiked = empty(queue.context, nv.shape, dtype=np.float32)

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
            %(OStype)s spiked = 0;

            for (int ii = 0; ii < %(upsample)s; ++ii)
            {
              %(Vtype)s dV = dt * tau_rc_inv * (input_current - v);
              v += dV;
              %(RTtype)s post_ref = - rt * dt_inv;
              v = v > 0 ?
                  v * (post_ref < 0 ? 0 : post_ref < 1 ? post_ref : 1)
                  : 0;
              spiked |= v > V_threshold;
              %(Vtype)s overshoot = (v - V_threshold) / dV;
              %(RTtype)s spiketime = dt * (1.0 - overshoot);

              v = v * (1.0 - spiked);
              rt = spiked ? spiketime + tau_ref : rt - dt;
            }

            out_voltage[gid] = v;
            out_refractory_time[gid] = rt;
            out_spiked[gid] = spiked;
        }
        """ % locals()).build().foo
    v = sim.ocl_vars[_v]

    _fn_args = (queue, v.shape, None,
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

        sim.ocl_vars[vv] = empty(queue.context,
                list(shape), np.dtype(vv.dtype))

@perform(theano.tensor.elemwise.Elemwise)
def elemwise_p(queue, sim, node):
    # XXX
    return []

