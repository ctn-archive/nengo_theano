import pyopencl as cl
from plan import Plan

def plan_copy(queue, src, dst):
    # XXX: only copy the parts of the buffer that are part of the logical Array
    if not (src.size == dst.size == src.data.size == dst.data.size):
        raise NotImplementedError()
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


