import numpy as np
import pyopencl as cl
import pyopencl.array

# XXX: use cl.array.Array, which has strides, flags

class Array(cl.array.Array):
    def __init__(self, queue, shape, dtype, order='C', allocator=None,
                 data=None, strides=None, offset=0):
        try:
            np.dtype(dtype)
        except Exception, e:
            e.args = e.args + (dtype,)
            raise
        cl.array.Array.__init__(self, queue,
                                shape=tuple(map(int, shape)),
                                allocator=allocator,
                                dtype=np.dtype(dtype),
                                order=order,
                                data=data,
                                strides=tuple(map(int, strides)))
        if 0 in self.strides:
            print self.strides
        self.offset = offset

    @property
    def itemstrides(self):
        rval = []
        for si in self.strides:
            assert 0 == si % self.dtype.itemsize
            rval.append(si // self.dtype.itemsize)
        return tuple(rval)

    def empty_like(self):
        # XXX consider whether to allocate just enough for data
        # or keep using the full buffer.
        data = cl.Buffer(self.data.context,
            flags=cl.mem_flags.READ_WRITE,
            size=self.data.size)
        return Array(self.queue,
                     shape=self.shape,
                     dtype=self.dtype,
                     data=data,
                     strides=self.strides,
                     offset=self.offset,
                    )

    # TODO: call this ctype, use cl.tools.dtype_to_ctype
    @property
    def ocldtype(self):
        return ocldtype(self.dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def structure(self):
        return (self.dtype, self.shape, self.strides, self.offset)


def ocldtype(obj):
    obj = str(obj)
    if isinstance(obj, basestring):
        return {
            'float32': 'float',
            'float64': 'double',
            'int64': 'long',
        }[obj]
    else:
        raise NotImplementedError('ocldtype', obj)

# consider array.to_device
def to_device(queue, arr, flags=cl.mem_flags.READ_WRITE):
    arr = np.asarray(arr)
    buf = cl.Buffer(queue.context, flags, size=len(arr.data))
    cl.enqueue_copy(queue, buf, arr.data).wait()
    return Array(queue,
                 data=buf,
                 dtype=arr.dtype,
                 shape=arr.shape,
                 strides=arr.strides)


# consider array.empty
def empty(queue, shape, dtype, flags=cl.mem_flags.READ_WRITE,
        strides=None, order='C'):

    dtype = np.dtype(dtype)
    
    if strides is None:
        strides = [int(dtype.itemsize)]
        if order.upper() == 'C':
            for shp in reversed(shape):
                strides = [strides[0] * int(shp)] + strides
            bufsize = strides[0]
            strides = strides[1:]
        elif order.upper() == 'F':
            raise NotImplementedError()
        else:
            raise ValueError(order)
    try:
        buf = cl.Buffer(queue.context, flags, size=bufsize)
    except:
        print queue, flags, type(flags), bufsize, type(bufsize)
        raise
    return Array(queue, data=buf, dtype=dtype, shape=shape, strides=strides)


