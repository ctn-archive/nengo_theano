import pyopencl as cl
from plan import Plan

def plan_copy(queue, src, dst):
    if not (src.shape == dst.shape):
        raise ValueError()
    if (src.data.size != dst.data.size):
        raise NotImplementedError('size', (src, dst))
    if (src.strides != dst.strides):
        raise NotImplementedError('strides', (src, dst))
    if (src.offset != dst.offset):
        raise NotImplementedError('offset', (src, dst))
    # XXX: only copy the parts of the buffer that are part of the logical Array
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


