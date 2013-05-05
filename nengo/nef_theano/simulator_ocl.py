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


class Array(object):
    def __init__(self, data, dtype, shape, strides):
        self.data = data
        self.dtype = dtype
        self.shape = shape
        self.strides = strides

    @property
    def ndim(self):
        return len(self.shape)

    def empty_like(self):
        buf = cl.Buffer(self.data.context,
            flags=cl.mem_flags.READ_WRITE,
            size=self.data.size)
        return Array(buf,
                dtype=self.dtype,
                shape=list(self.shape),
                strides=list(self.strides))


def to_device(queue, arr, flags=cl.mem_flags.READ_WRITE):
    arr = np.asarray(arr)
    buf = cl.Buffer(queue.context, flags, size=len(arr.data))
    cl.enqueue_copy(queue, buf, arr.data).wait()
    return Array(buf, arr.dtype, arr.shape, arr.strides)


def empty(context, shape, dtype, flags=cl.mem_flags.READ_WRITE,
        strides=None, order='C'):

    dtype = np.dtype(dtype)
    
    if strides is None:
        strides = [dtype.itemsize]
        for shp in shape:
            strides.append(strides[-1] * shp)
        size = strides[-1]
        strides = strides[:-1]
        if order == 'C':
            strides.reverse()
    buf = cl.Buffer(context, flags, size=size)
    return Array(buf, dtype, shape, strides)


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
    def __init__(self, network, context=None):
        self.network = network
        if context is None:
            context = cl.create_some_context()
        self.context = context
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

        self.thunks = []
        self.ocl_vars = {}
        self.constant_vars = {}

        for node in self.order:
            for vv in node.inputs:
                if vv in self.ocl_vars:
                    continue
                if vv in self.constant_vars:
                    continue
                try:
                    const_val = theano.tensor.basic.get_scalar_constant_value(vv)
                    self.constant_vars[vv] = const_val
                    continue
                except theano.tensor.basic.NotScalarConstantError:
                    pass
                if vv.owner is None:
                    if hasattr(vv, 'data'):
                        val = vv.data
                    else:
                        val = vv.get_value(borrow=True)
                    # XXX this logic should be optional
                    if val.dtype == 'float64':
                        val = val.astype('float32')
                    if val.dtype == 'int64':
                        val = val.astype('int32')
                    self.ocl_vars[vv] = to_device(self.queue, val)
            ocl_alloc[type(node.op)](self.queue, self, node)
            plans = ocl_perform[type(node.op)](self.queue, self, node)
            self.thunks.extend(plans)

        # XXX create another program that does the update
        # in the other direction (double-buffer the ocl_vars)

    def run_steps(self, N):
        # -- copy from shared variable inputs into internal state dict
        # -- run N steps
        # -- copy from state to shared variables
        for i in xrange(N):
            self.step()


    def run(self, approx_sim_time):
        n_steps = int(approx_sim_time / self.network.dt)
        return self.run_steps(n_steps)


class Plan(object):

    def __init__(self, dct):
        self.__dict__.update(dct)

    def __call__(self):
        self._fn(*self._fn_args)
        self._fn_args[0].finish()
#
# 
#

def gemv_batched_ref(context, B, M, N, alpha,
                             Aoffset, AsB, AsM, AsN,
                             XsN,
                             beta,
                             YsM,
                            ):
    return cl.Program(context, """
        __kernel void fn(__global const float *A_data,
                         __global const float *X_data,
                         __global const int *X_offsets,
                         __global float *Y_data,
                         __global const int *Y_offsets
                         )
        {
            const int bb = get_global_id(0);

            A_data += %(Aoffset)s + bb * %(AsB)s;
            X_data += X_offsets[bb];
            Y_data += Y_offsets[bb];

            for (int mm = 0; mm < %(M)s; ++mm)
            {
                float ksum = 0.0;
                for (int nn = 0; nn < %(N)s; ++nn)
                {
                    ksum += A_data[nn * %(AsN)s  + mm * %(AsM)s] * X_data[nn * %(XsN)s];
                }

                if (%(beta)s == 0)
                {
                    Y_data[%(YsM)s * mm] = %(alpha)s * ksum;
                }
                else
                {
                    Y_data[%(YsM)s * mm] = %(beta)s * Y_data[%(YsM)s * mm] + %(alpha)s * ksum;
                }
            }
        }
        """ % locals()).build().fn


def gemv_batched_parout_nolocal(context, B, M, N, alpha,
                             Aoffset, AsB, AsM, AsN,
                             XsN,
                             beta,
                             YsM,
                            ):
    return cl.Program(context, """
        __kernel void fn(__global const float *A_data,
                         __global const float *X_data,
                         __global const int *X_offsets,
                         __global float *Y_data,
                         __global const int *Y_offsets
                         )
        {
            const int mm0 = get_global_id(0);
            const int bb = get_global_id(1);

            A_data += %(Aoffset)s + bb * %(AsB)s;
            X_data += X_offsets[bb];
            Y_data += Y_offsets[bb];

            for (int mm = mm0; mm < %(M)s; mm += get_local_size(0))
            {
                float ksum = 0.0;
                for (int nn = 0; nn < %(N)s; ++nn)
                {
                    ksum += A_data[nn * %(AsN)s  + mm * %(AsM)s] * X_data[nn * %(XsN)s];
                }

                if (%(beta)s == 0)
                {
                    Y_data[%(YsM)s * mm] = %(alpha)s * ksum;
                }
                else
                {
                    Y_data[%(YsM)s * mm] = %(beta)s * Y_data[%(YsM)s * mm] + %(alpha)s * ksum;
                }
            }
        }
        """ % locals()).build().fn


def gemv_batched_parout_local(context, B, M, N, alpha,
                             Aoffset, AsB, AsM, AsN,
                             XsN,
                             beta,
                             YsM,
                            ):
    return cl.Program(context, """
        __kernel void fn(__global const float *A_data,
                         __global const float *X_data,
                         __global const int *X_offsets,
                         __global float *Y_data,
                         __global const int *Y_offsets,
                         __local float * Xbuf
                         )
        {
            const int bb = get_global_id(1);

            A_data += %(Aoffset)s + bb * %(AsB)s;
            X_data += X_offsets[bb];
            Y_data += Y_offsets[bb];
            __local float * Ybuf = Xbuf + %(N)s;

            for(int nn = get_local_id(0); nn < %(N)s; nn += get_local_size(0))
            {
                Xbuf[nn] = X_data[nn * %(XsN)s];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            for (int mm = get_local_id(0); mm < %(M)s; mm += get_local_size(0))
            {
                float tmp = 0.0;
                for (int nn = 0; nn < %(N)s; nn += 1)
                {
                    tmp += A_data[nn * %(AsN)s + mm * %(AsM)s] * Xbuf[nn];
                }

                if (%(beta)s != 0)
                {
                    Y_data[mm * %(YsM)s] = Y_data[mm * %(YsM)s] * %(beta)s
                        + %(alpha)s * tmp;
                }
                else
                {
                    Y_data[mm * %(YsM)s] = %(alpha)s * tmp;
                }
            }
        }
        """ % locals()).build().fn


def choose_gemv_batched_plan(
    BMN, alpha, Aparams, Xparams, beta, Yparams, queues,
    ):
    """
    For each i, compute

    Yi <- alpha * dot(Ai, Xi) + beta * Yi

    Where
    Yi = Y[Y_offsets[i]: Y_offsets[i] + YsM * M: YsM]
    Xi = X[X_offsets[i]: X_offsets[i] + XsN * N: XsN]
    Ai is an M x N matrix whose first element is at
        A[A_offsets[i]] and is strided on the dimension
        of size M by AsM, and is strided on the dimension
        of size N by AsN.

    """
    B, M, N = BMN
    A_buf, Aoffset, AsB, AsM, AsN = Aparams
    X_buf, X_offsets, XsN = Xparams
    Y_buf, Y_offsets, YsM = Yparams
    queue = queues[0]
    if np.float32 != A_buf.dtype:
        raise NotImplementedError('A dtype', A_buf.dtype)
    if np.float32 != X_buf.dtype:
        raise NotImplementedError('X dtype', X_buf.dtype)
    if np.float32 != Y_buf.dtype:
        raise NotImplementedError('Y dtype', Y_buf.dtype)

    # TODO: autotune decision/regression tree
    if M == 1:
        _fn = gemv_batched_ref(queue.context,
            B, M, N,
            alpha,
            Aoffset, AsB, AsM, AsN,
            XsN,
            beta,
            YsM)
        global_shape = (B,)
        local_shape = None
        _fn_args = (queue, global_shape, local_shape, A_buf.data,
                    X_buf.data, X_offsets.data,
                    Y_buf.data, Y_offsets.data)
    elif 0: # this is a good idea for A in Fortran order
        _fn = gemv_batched_parout_nolocal(queue.context,
            B, M, N,
            alpha,
            Aoffset, AsB, AsM, AsN,
            XsN,
            beta,
            YsM)
        mpergroup = min(queue.context.devices[0].max_work_group_size, M)
        global_shape = (mpergroup, B,)
        local_shape = (mpergroup, 1)
        _fn_args = (queue, global_shape, local_shape, A_buf.data,
                    X_buf.data, X_offsets.data,
                    Y_buf.data, Y_offsets.data)
    else: # this is a good idea for A in C order
        _fn = gemv_batched_parout_local(queue.context,
            B, M, N,
            alpha,
            Aoffset, AsB, AsM, AsN,
            XsN,
            beta,
            YsM)
        mpergroup = min(queue.context.devices[0].max_work_group_size, M)
        global_shape = (mpergroup, B,)
        local_shape = (mpergroup, 1)
        local_mem = cl.LocalMemory(4 * N)
        _fn_args = (queue, global_shape, local_shape, A_buf.data,
                    X_buf.data, X_offsets.data,
                    Y_buf.data, Y_offsets.data, local_mem)


    return Plan(locals())


def choose_copy(queue, src, dst):
    _fn = cl.Program(queue.context, """
        __kernel void fn(__global const float *src,
                         __global float *dst
                         )
        {
            dst[get_global_id(0)] = src[get_global_id(0)];
        }
        """ % locals()).build().fn
    _fn_args = (queue, (src.data.size,), None, src.data, dst.data)
    return Plan(locals())


@alloc(simulator.MapGemv)
def ocl_map_gemv(queue, sim, node):
    try:
        J = sim.ocl_vars[node.inputs[-1]]
    except KeyError:
        J = sim.constant_vars[node.inputs[-1]]
        J = np.asarray(J).reshape((1,) * node.inputs[-1].ndim)
        assert np.all(node.inputs[-1].broadcastable)
    sim.ocl_vars[node.outputs[0]] = empty(queue.context, J.shape, J.dtype)


@perform(simulator.MapGemv)
def ocl_map_gemv(queue, sim, node):
    alpha, A, X, beta, Y_in = [sim.ocl_vars.get(vi) for vi in node.inputs]
    Y_out, = [sim.ocl_vars[vi] for vi in node.outputs]

    if Y_in is None:
        Y_in_val = sim.constant_vars[node.inputs[-1]]

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

    rval = []
    if fbeta != 0:
        if Y_in is None:
            if np.any(Y_in_val != 0):
                raise NotImplementedError()
            fbeta = 0
        else:
            rval.append(choose_copy(queue, Y_in, Y_out))
    gemv_plan = choose_gemv_batched_plan(BMN=A.shape, alpha=falpha,
            Aparams=(A, 0, M * N, N, 1),
            Xparams=(X, X_offsets, M),
            beta=fbeta,
            Yparams=(Y_out, Y_offsets, N),
            queues=[queue])
    rval.append(gemv_plan)
    return rval


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
    X, shape = node.inputs
    Xval = sim.ocl_vars[X]
    shape_val = sim.constant_vars[shape]

    if np.prod(Xval.shape) == np.prod(shape_val) == 1:
        Yval = Array(Xval.data, Xval.dtype,
                shape=shape_val,
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
            shape=(np.prod(Xval.shape),),
            strides=[0])
    sim.ocl_vars[node.outputs[0]] = Yval

@perform(theano.tensor.basic.Flatten)
def flatten_p(queue, sim, node):
    return []

@alloc(theano.tensor.basic.Dot)
def flatten_a(queue, sim, node):
    X, Y,= node.inputs
    if X in sim.ocl_vars:
        Xval = sim.ocl_vars[X]
    else:
        Xval = sim.constant_vars[X]
        print Xval
    Yval = sim.ocl_vars[Y]

    print Xval.ndim
    print Yval.ndim

    A, B = Xval.shape
    C, D = Yval.shape

    Zval = empty(Xval.data.context, node.outputs[0].dtype,
            shape=(Xval.shape[0], Yval.shape[1]),
            strides=[0])
    sim.ocl_vars[node.outputs[0]] = Yval

@perform(theano.tensor.basic.Dot)
def flatten_p(queue, sim, node):
    return []


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

    dt = float(_dt.value)
    tau_rc = node.op.tau_rc
    tau_ref  = node.op.tau_ref
    V_threshold = 1.0
    tau_rc_inv = 1.0 / tau_rc

    upsample = node.op.upsample
    upsample_dt = dt / upsample
    upsample_dt_inv = 1.0 / upsample_dt

    # XXX GET DTYPES RIGHT
    _fn = cl.Program(queue.context, """
        __kernel void foo(
            __global const float *J,
            __global const float *voltage,
            __global const float *refractory_time,
            __global float *out_voltage,
            __global float *out_refractory_time,
            __global char *out_spiked
                     )
        {
            const float dt = %(upsample_dt)s;
            const float dt_inv = %(upsample_dt_inv)s;
            const float tau_ref = %(tau_ref)s;
            const float tau_rc_inv = %(tau_rc_inv)s;
            const float V_threshold = %(V_threshold)s;

            int gid = get_global_id(0);
            float v = voltage[gid];
            float rt = refractory_time[gid];
            float input_current = J[gid];
            char spiked = 0;

            for (int ii = 0; ii < %(upsample)s; ++ii)
            {
              float dV = dt * tau_rc_inv * (input_current - v);
              v += dV;
              float post_ref = - rt * dt_inv;
              v = v > 0 ?
                  v * (post_ref < 0 ? 0 : post_ref < 1 ? post_ref : 1)
                  : 0;
              spiked |= v > V_threshold;
              float overshoot = (v - V_threshold) / dV;
              float spiketime = dt * (1.0 - overshoot);

              v = v * (1.0 - spiked);
              rt = spiked ? spiketime + tau_ref : rt - dt;
            }

            out_voltage[gid] = v;
            out_refractory_time[gid] = rt;
            out_spiked[gid] = spiked;
        }
        """ % locals()).build().foo

    _v, _rt, _ic, _dt = node.inputs
    _ov, _ort, _os = node.outputs

    _fn_args = (queue, _v.shape, None,
        sim.ocl_vars[_ic].data,
        sim.ocl_vars[_v].data,
        sim.ocl_vars[_rt].data,
        sim.ocl_vars[_ov].data,
        sim.ocl_vars[_ort].data,
        sim.ocl_vars[_os].data,
        )
    return [Plan(locals())]



