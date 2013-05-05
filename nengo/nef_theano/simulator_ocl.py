"""

TODO
----
* move from cl.arra.Array to Buffers and strides
* dtype management
"""

from _collections import OrderedDict
import theano
import numpy as np
import pyopencl as cl

class Array(object):
    def __init__(self, data, dtype, shape, strides):
        self.data = data
        self.dtype = dtype
        self.shape = shape
        self.strides = strides

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



import simulator

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

        for node in self.order:
            for vv in node.inputs:
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
            ocl_alloc[type(node.op)](self.queue, self.ocl_vars, node)
            ocl_thunk = ocl_perform[type(node.op)](self.queue, self.ocl_vars, node)
            self.thunks.append(ocl_thunk)

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
def ocl_map_gemv(queue, ocl_vars, node):
    J = ocl_vars[node.inputs[-1]]
    ocl_vars[node.outputs[0]] = J.empty_like()


@perform(simulator.MapGemv)
def ocl_map_gemv(queue, ocl_vars, node):
    alpha, A, X, beta, Y_in = [ocl_vars[vi] for vi in node.inputs]
    Y_out, = [ocl_vars[vi] for vi in node.outputs]

    # XXX: following depends on constants alpha, beta
    falpha = float(node.inputs[0].data)
    fbeta = float(node.inputs[3].data)

    B, M, N = A.shape
    Bx, Nx = X.shape
    By, My = Y_in.shape
    assert Bx == By == B
    assert My == M
    assert Nx == N

    A_offsets = to_device(queue, np.arange(B) * M * N )
    X_offsets = to_device(queue, np.arange(B) * N )
    Y_offsets = to_device(queue, np.arange(B) * M)
    
    y_copy = choose_copy(queue, Y_in, Y_out)
    gemv = choose_gemv_batched_plan(BMN=A.shape, alpha=falpha,
            Aparams=(A, 0, M * N, N, 1),
            Xparams=(X, X_offsets, M),
            beta=fbeta,
            Yparams=(Y_out, Y_offsets, N),
            queues=[queue])
    return [y_copy, gemv]
